from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .base import SegmentationOutput
from .unet3d import ConvBlock


def _inverse_permutation(order: Sequence[int]) -> tuple[int, int, int]:
    inverse = [0, 0, 0]
    for output_axis, input_axis in enumerate(order):
        inverse[input_axis] = output_axis
    return tuple(inverse)


class _TinyOCTCubeBackbone(nn.Module):
    """Small contract-compatible backbone used only by synthetic tests."""

    def __init__(self, embedding_dim: int, patch_size: int, temporal_patch_size: int) -> None:
        super().__init__()
        self.embed_dim = embedding_dim
        self.patch_embed = nn.Conv3d(
            1,
            embedding_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(embedding_dim, embedding_dim, 1),
                    nn.GELU(),
                    nn.Conv3d(embedding_dim, embedding_dim, 1),
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.GroupNorm(1, embedding_dim)

    def token_grid(self, image: torch.Tensor) -> torch.Tensor:
        current = self.patch_embed(image)
        for block in self.blocks:
            current = current + block(current)
        return self.norm(current)


class OCTCubeBackboneAdapter(nn.Module):
    """Adapts the official OCTCube ViT to spatial features for segmentation."""

    def __init__(
        self,
        *,
        backend: str,
        source_root: str,
        module: str,
        model_name: str,
        checkpoint: str,
        require_checkpoint: bool,
        minimum_checkpoint_coverage: float,
        train_mode: str,
        partial_unfreeze_blocks: int,
        input_size: int,
        num_frames: int,
        patch_size: int,
        temporal_patch_size: int,
        embedding_dim: int,
        separate_position_embedding: bool,
        class_token: bool,
    ) -> None:
        super().__init__()
        self.backend_name = backend
        self.input_size = input_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.class_token = class_token
        self.train_mode = train_mode
        self._captured_tokens: torch.Tensor | None = None

        if backend == "internal_test":
            self.backbone = _TinyOCTCubeBackbone(
                embedding_dim=embedding_dim,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
            )
        elif backend == "official":
            if source_root:
                resolved_source = Path(source_root).expanduser().resolve()
                if not resolved_source.is_dir():
                    raise FileNotFoundError(f"OCTCube source root not found: {resolved_source}")
                source_text = str(resolved_source)
                if source_text not in sys.path:
                    sys.path.insert(0, source_text)
            try:
                octcube_module = importlib.import_module(module)
            except (ImportError, ModuleNotFoundError) as error:
                raise ImportError(
                    "Cannot import the official OCTCube source. Set OCTCUBE_SOURCE_ROOT "
                    "to the cloned OCTCubeM repository or install it on PYTHONPATH."
                ) from error
            factory = getattr(octcube_module, model_name, None)
            if factory is None:
                raise ValueError(f"Official OCTCube module has no model factory {model_name!r}")
            self.backbone = factory(
                num_frames=num_frames,
                t_patch_size=temporal_patch_size,
                img_size=input_size,
                in_chans=1,
                num_classes=2,
                global_pool=False,
                sep_pos_embed=separate_position_embedding,
                cls_embed=class_token,
            )
            blocks = getattr(self.backbone, "blocks", None)
            if not blocks:
                raise ValueError("The official OCTCube model does not expose transformer blocks")
            capture_layer = getattr(self.backbone, "norm", blocks[-1])
            capture_layer.register_forward_hook(self._capture_spatial_tokens)
        else:
            raise ValueError("OCTCube backend must be 'official' or 'internal_test'")

        self.embedding_dim = int(getattr(self.backbone, "embed_dim", embedding_dim))
        if checkpoint:
            self._load_checkpoint(checkpoint, minimum_checkpoint_coverage)
        elif require_checkpoint and backend == "official":
            raise ValueError(
                "model.octcube.checkpoint is required for pretrained OCTCube training; "
                "set OCTCUBE_CHECKPOINT to the authorized local OCTCube.pth file"
            )
        self._configure_trainable_parameters(train_mode, partial_unfreeze_blocks)

    def _capture_spatial_tokens(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("Unexpected OCTCube transformer block output")
        self._captured_tokens = output

    def _load_checkpoint(self, checkpoint: str, minimum_coverage: float) -> None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"OCTCube checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            state = payload.get("model", payload.get("state_dict", payload))
        else:
            state = payload
        if not isinstance(state, dict):
            raise ValueError("OCTCube checkpoint must contain a tensor state dictionary")
        own = self.backbone.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        for raw_key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = (
                str(raw_key)
                .removeprefix("module.")
                .removeprefix("model.")
                .removeprefix("backbone.")
            )
            if key in own and own[key].shape == value.shape:
                compatible[key] = value
        eligible = {
            key: value
            for key, value in own.items()
            if not key.startswith(("head.", "fc_norm."))
        }
        loaded_elements = sum(eligible[key].numel() for key in compatible if key in eligible)
        total_elements = sum(value.numel() for value in eligible.values())
        coverage = loaded_elements / max(total_elements, 1)
        if coverage < minimum_coverage:
            raise ValueError(
                f"OCTCube checkpoint parameter coverage {coverage:.1%} is below "
                f"the required {minimum_coverage:.1%}. Check model_name and pretrained geometry."
            )
        self.backbone.load_state_dict(compatible, strict=False)

    def _configure_trainable_parameters(self, mode: str, partial_blocks: int) -> None:
        if mode not in {"frozen", "partial", "full"}:
            raise ValueError("OCTCube train_mode must be frozen, partial, or full")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "partial":
            blocks = list(getattr(self.backbone, "blocks", []))
            if partial_blocks < 1 or partial_blocks > len(blocks):
                raise ValueError(
                    f"partial_unfreeze_blocks must be between 1 and {len(blocks)} for partial tuning"
                )
            for block in blocks[-partial_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for name in ("norm", "fc_norm"):
                layer = getattr(self.backbone, name, None)
                if layer is not None:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True

    def train(self, mode: bool = True) -> OCTCubeBackboneAdapter:
        super().train(mode)
        if self.train_mode == "frozen":
            self.backbone.eval()
        return self

    def _forward_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if self.backend_name == "internal_test":
            return self.backbone.token_grid(image)  # type: ignore[union-attr]
        self._captured_tokens = None
        if hasattr(self.backbone, "forward_features"):
            self.backbone.forward_features(image)
        else:
            self.backbone(image)
        tokens = self._captured_tokens
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError("Could not capture OCTCube spatial tokens")
        temporal_tokens = self.num_frames // self.temporal_patch_size
        spatial_tokens = self.input_size // self.patch_size
        expected = temporal_tokens * spatial_tokens * spatial_tokens
        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:]
        if tokens.shape[1] != expected:
            raise ValueError(
                f"OCTCube returned {tokens.shape[1]} tokens; expected {expected} spatial tokens"
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.embedding_dim, temporal_tokens, spatial_tokens, spatial_tokens
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.train_mode == "frozen":
            with torch.no_grad():
                return self._forward_tokens(image)
        return self._forward_tokens(image)


class OCTCubePyramid(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        axis_order: Sequence[int],
        normalization: str,
        normalization_subtrahend: float,
        normalization_divisor: float,
        normalization_nonzero: bool,
        backbone: OCTCubeBackboneAdapter,
    ) -> None:
        super().__init__()
        self.channels = tuple(channels)
        self.axis_order = tuple(int(axis) for axis in axis_order)
        self.inverse_axis_order = _inverse_permutation(self.axis_order)
        self.normalization = normalization
        self.normalization_subtrahend = normalization_subtrahend
        self.normalization_divisor = normalization_divisor
        self.normalization_nonzero = normalization_nonzero
        self.backbone = backbone
        self.projections = nn.ModuleList(
            [nn.Conv3d(backbone.embedding_dim, channel, kernel_size=1) for channel in channels]
        )

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        image = image.permute(0, 1, *(axis + 2 for axis in self.axis_order))
        image = F.interpolate(
            image,
            size=(self.backbone.num_frames, self.backbone.input_size, self.backbone.input_size),
            mode="trilinear",
            align_corners=False,
        )
        if self.normalization == "octcube_fixed":
            normalized = (image - self.normalization_subtrahend) / self.normalization_divisor
            image = torch.where(image != 0, normalized, image) if self.normalization_nonzero else normalized
        elif self.normalization != "dataset":
            raise ValueError("OCTCube normalization must be 'octcube_fixed' or 'dataset'")
        return image

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        original_size = image.shape[-3:]
        oriented_size = tuple(original_size[axis] for axis in self.axis_order)
        tokens = self.backbone(self._prepare(image))
        features: list[torch.Tensor] = []
        for level, projection in enumerate(self.projections):
            size = tuple(max(dimension // (2**level), 1) for dimension in oriented_size)
            current = F.interpolate(tokens, size=size, mode="trilinear", align_corners=False)
            current = projection(current)
            current = current.permute(0, 1, *(axis + 2 for axis in self.inverse_axis_order))
            features.append(current)
        return features


class AuxiliaryPyramid(nn.Module):
    def __init__(self, channels: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        previous = 1
        for channel in channels:
            self.blocks.append(ConvBlock(previous, channel, dropout))
            previous = channel
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in channels[:-1]])

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        current = image
        for index, block in enumerate(self.blocks):
            current = block(current)
            features.append(current)
            if index < len(self.pools):
                current = self.pools[index](current)
        return features


class FusionBlock(nn.Module):
    def __init__(self, channels: int, modality_count: int, method: str, attention_heads: int) -> None:
        super().__init__()
        self.method = method
        self.modality_count = modality_count
        if method == "concatenation":
            self.refine = ConvBlock(channels * modality_count, channels)
        else:
            self.refine = ConvBlock(channels, channels)
        self.modality_score = nn.Linear(channels, 1) if method == "attention" else None
        self.cross_attention = None
        if method == "cross_modal":
            if channels % attention_heads:
                raise ValueError("Fusion channels must be divisible by attention_heads")
            self.cross_attention = nn.MultiheadAttention(
                channels, num_heads=attention_heads, batch_first=True
            )
        if method not in {"concatenation", "addition", "attention", "cross_modal"}:
            raise ValueError(f"Unknown multimodal fusion method: {method}")

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.modality_count:
            raise ValueError("Fusion block received an unexpected number of modality features")
        if self.method == "concatenation":
            return self.refine(torch.cat(list(features), dim=1))
        stacked = torch.stack(list(features), dim=1)
        if self.method == "addition":
            fused = stacked.mean(dim=1)
        elif self.method == "attention":
            descriptors = stacked.mean(dim=(-3, -2, -1))
            weights = torch.softmax(self.modality_score(descriptors), dim=1)
            fused = (stacked * weights[..., None, None, None]).sum(dim=1)
        else:
            descriptors = stacked.mean(dim=(-3, -2, -1))
            query = descriptors[:, :1]
            context, _ = self.cross_attention(query, descriptors, descriptors, need_weights=False)
            fused = stacked.mean(dim=1) + context[:, 0, :, None, None, None]
        return self.refine(fused)


class OCTCubeUNet3D(nn.Module):
    """OCTCube OCT encoder plus modality-specific 3D encoders and a U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        modalities: Sequence[str],
        out_channels: int = 1,
        base_channels: int = 16,
        levels: int = 4,
        dropout: float = 0.0,
        fusion: str = "concatenation",
        attention_heads: int = 4,
        octcube: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        if len(self.modalities) != in_channels:
            raise ValueError("The number of modalities must match in_channels")
        if "oct" not in self.modalities:
            raise ValueError("octcube_unet3d requires the OCT modality")
        if levels < 2:
            raise ValueError("octcube_unet3d requires at least two levels")
        options = dict(octcube or {})
        axis_order = options.pop("axis_order", [2, 0, 1])
        if sorted(axis_order) != [0, 1, 2]:
            raise ValueError("OCTCube axis_order must be a permutation of [0, 1, 2]")
        channels = [base_channels * (2**index) for index in range(levels)]
        backbone = OCTCubeBackboneAdapter(
            backend=options.pop("backend", "official"),
            source_root=options.pop("source_root", ""),
            module=options.pop("module", "OCTCube.models_vit_st"),
            model_name=options.pop("model_name", "vit_large_patch16"),
            checkpoint=options.pop("checkpoint", ""),
            require_checkpoint=bool(options.pop("require_checkpoint", True)),
            minimum_checkpoint_coverage=float(options.pop("minimum_checkpoint_coverage", 0.9)),
            train_mode=options.pop("train_mode", "frozen"),
            partial_unfreeze_blocks=int(options.pop("partial_unfreeze_blocks", 4)),
            input_size=int(options.pop("input_size", 256)),
            num_frames=int(options.pop("num_frames", 64)),
            patch_size=int(options.pop("patch_size", 16)),
            temporal_patch_size=int(options.pop("temporal_patch_size", 2)),
            embedding_dim=int(options.pop("embedding_dim", 1024)),
            separate_position_embedding=bool(options.pop("separate_position_embedding", True)),
            class_token=bool(options.pop("class_token", True)),
        )
        normalization = options.pop("normalization", "octcube_fixed")
        normalization_subtrahend = float(options.pop("normalization_subtrahend", 0.25))
        normalization_divisor = float(options.pop("normalization_divisor", 0.25))
        normalization_nonzero = bool(options.pop("normalization_nonzero", True))
        if normalization_divisor == 0:
            raise ValueError("OCTCube normalization_divisor cannot be zero")
        if options:
            raise ValueError(f"Unknown OCTCube options: {sorted(options)}")
        self.oct_encoder = OCTCubePyramid(
            channels,
            axis_order,
            normalization,
            normalization_subtrahend,
            normalization_divisor,
            normalization_nonzero,
            backbone,
        )
        self.auxiliary_encoders = nn.ModuleDict(
            {
                modality: AuxiliaryPyramid(channels, dropout)
                for modality in self.modalities
                if modality != "oct"
            }
        )
        self.fusion = nn.ModuleList(
            [FusionBlock(channel, len(self.modalities), fusion, attention_heads) for channel in channels]
        )
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(levels - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(channels[index], channels[index - 1], kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(channels[index], channels[index - 1], dropout))
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward_logits(self, image: torch.Tensor) -> torch.Tensor:
        modality_images = {
            modality: image[:, index : index + 1]
            for index, modality in enumerate(self.modalities)
        }
        pyramids = {"oct": self.oct_encoder(modality_images["oct"])}
        for modality, encoder in self.auxiliary_encoders.items():
            pyramids[modality] = encoder(modality_images[modality])
        fused = [
            fusion([pyramids[modality][level] for modality in self.modalities])
            for level, fusion in enumerate(self.fusion)
        ]
        current = fused[-1]
        for upconv, decoder, skip in zip(
            self.upconvs, self.decoder, reversed(fused[:-1]), strict=True
        ):
            current = upconv(current)
            if current.shape[-3:] != skip.shape[-3:]:
                current = F.interpolate(
                    current, size=skip.shape[-3:], mode="trilinear", align_corners=False
                )
            current = decoder(torch.cat([current, skip], dim=1))
        return self.head(current)

    def forward(self, image: torch.Tensor, target: torch.Tensor | None = None) -> SegmentationOutput:
        del target
        return SegmentationOutput(logits=self.forward_logits(image))
