from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .base import SegmentationOutput
from .unet3d import UNet3D


class FoundationUNet3D(nn.Module):
    """3D U-Net with an encoder initialized from an external foundation checkpoint.

    The checkpoint must contain compatible encoder tensors. Coverage is checked so
    silently loading an unrelated checkpoint is impossible.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 16,
        levels: int = 4,
        dropout: float = 0.0,
        foundation_checkpoint: str = "",
        require_foundation_checkpoint: bool = True,
        freeze_encoder: bool = True,
        minimum_checkpoint_coverage: float = 0.7,
    ) -> None:
        super().__init__()
        self.network = UNet3D(
            in_channels,
            out_channels,
            base_channels,
            levels,
            dropout,
            residual_blocks=False,
        )
        if foundation_checkpoint:
            self._load_encoder(foundation_checkpoint, minimum_checkpoint_coverage)
        elif require_foundation_checkpoint:
            raise ValueError(
                "model.foundation_checkpoint is required for foundation_unet3d; "
                "set RNV_FOUNDATION_CHECKPOINT to a compatible local checkpoint"
            )
        if freeze_encoder:
            for parameter in self.network.encoder.parameters():
                parameter.requires_grad = False

    def _load_encoder(self, checkpoint_path: str, minimum_coverage: float) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Foundation checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError("Foundation checkpoint must contain a tensor state dictionary")
        own = self.network.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        for raw_key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = raw_key.removeprefix("module.").removeprefix("network.")
            if not key.startswith("encoder."):
                key = f"encoder.{key.removeprefix('backbone.').removeprefix('encoder.')}"
            if key in own and own[key].shape == value.shape:
                compatible[key] = value
        encoder_keys = [key for key in own if key.startswith("encoder.")]
        coverage = len(compatible) / max(len(encoder_keys), 1)
        if coverage < minimum_coverage:
            raise ValueError(
                f"Foundation checkpoint encoder coverage {coverage:.1%} is below "
                f"the required {minimum_coverage:.1%}"
            )
        self.network.load_state_dict(compatible, strict=False)

    def forward(self, image: torch.Tensor, target: torch.Tensor | None = None) -> SegmentationOutput:
        return self.network(image, target)
