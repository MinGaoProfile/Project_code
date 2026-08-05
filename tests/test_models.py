import sys
import types

import pytest
import torch
from torch import nn

from rnv_seg.models import build_model


@pytest.mark.parametrize("name", ["unet3d", "diffusion_unet3d", "foundation_unet3d"])
def test_model_contract(name: str) -> None:
    config = {
        "name": name,
        "out_channels": 1,
        "base_channels": 4,
        "levels": 3,
        "dropout": 0.0,
    }
    if name == "diffusion_unet3d":
        config.update(diffusion_steps=5, diffusion_loss_weight=0.1)
    if name == "foundation_unet3d":
        config.update(require_foundation_checkpoint=False, freeze_encoder=False)
    model = build_model(config, in_channels=2)
    image = torch.randn(1, 2, 16, 16, 16)
    target = (torch.rand(1, 1, 16, 16, 16) > 0.9).float()
    model.train()
    output = model(image, target)
    assert output.logits.shape == target.shape
    (output.logits.mean() + sum(output.auxiliary_losses.values())).backward()


@pytest.mark.parametrize("fusion", ["concatenation", "addition", "attention", "cross_modal"])
def test_octcube_multimodal_model_contract(fusion: str) -> None:
    config = {
        "name": "octcube_unet3d",
        "out_channels": 1,
        "base_channels": 4,
        "levels": 3,
        "dropout": 0.0,
        "fusion": fusion,
        "attention_heads": 2,
        "octcube": {
            "backend": "internal_test",
            "require_checkpoint": False,
            "train_mode": "partial",
            "partial_unfreeze_blocks": 1,
            "input_size": 16,
            "num_frames": 8,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "embedding_dim": 8,
            "axis_order": [2, 0, 1],
            "normalization": "octcube_fixed",
        },
    }
    modalities = ["oct", "octa", "artery", "vein"]
    model = build_model(config, in_channels=4, modalities=modalities)
    image = torch.rand(1, 4, 8, 16, 16)
    target = (torch.rand(1, 1, 8, 16, 16) > 0.9).float()
    output = model(image, target)
    assert output.logits.shape == target.shape
    output.logits.mean().backward()


def test_octcube_requires_oct_modality() -> None:
    config = {
        "name": "octcube_unet3d",
        "base_channels": 4,
        "levels": 2,
        "octcube": {"backend": "internal_test", "require_checkpoint": False},
    }
    with pytest.raises(ValueError, match="requires the OCT modality"):
        build_model(config, in_channels=1, modalities=["octa"])


def test_official_octcube_adapter_token_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOfficialOCTCube(nn.Module):
        def __init__(self, **_kwargs: object) -> None:
            super().__init__()
            self.embed_dim = 8
            self.patch_embed = nn.Conv3d(1, 8, kernel_size=(2, 4, 4), stride=(2, 4, 4))
            self.blocks = nn.ModuleList([nn.Identity()])
            self.norm = nn.LayerNorm(8)

        def forward_features(self, image: torch.Tensor) -> torch.Tensor:
            grid = self.patch_embed(image)
            tokens = grid.flatten(2).transpose(1, 2)
            class_token = torch.zeros(
                tokens.shape[0], 1, tokens.shape[2], device=tokens.device, dtype=tokens.dtype
            )
            tokens = torch.cat([class_token, tokens], dim=1)
            return self.norm(self.blocks[-1](tokens))

    fake_module = types.ModuleType("fake_octcube")
    fake_module.vit_large_patch16 = lambda **kwargs: FakeOfficialOCTCube(**kwargs)
    monkeypatch.setitem(sys.modules, "fake_octcube", fake_module)
    config = {
        "name": "octcube_unet3d",
        "base_channels": 4,
        "levels": 2,
        "octcube": {
            "backend": "official",
            "module": "fake_octcube",
            "require_checkpoint": False,
            "train_mode": "frozen",
            "input_size": 16,
            "num_frames": 8,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "embedding_dim": 8,
        },
    }
    model = build_model(config, in_channels=1, modalities=["oct"])
    output = model(torch.rand(1, 1, 8, 16, 16))
    assert output.logits.shape == (1, 1, 8, 16, 16)
    assert not any(
        parameter.requires_grad
        for parameter in model.oct_encoder.backbone.backbone.parameters()
    )
