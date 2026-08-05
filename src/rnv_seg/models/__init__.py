from __future__ import annotations

from typing import Any

from rnv_seg.registry import MODELS

from .diffusion_unet3d import DiffusionUNet3D
from .foundation_unet3d import FoundationUNet3D
from .octcube_unet3d import OCTCubeUNet3D
from .unet3d import UNet3D


@MODELS.register("unet3d")
def _unet3d(**kwargs: Any) -> UNet3D:
    return UNet3D(**kwargs)


@MODELS.register("diffusion_unet3d")
def _diffusion_unet3d(**kwargs: Any) -> DiffusionUNet3D:
    return DiffusionUNet3D(**kwargs)


@MODELS.register("foundation_unet3d")
def _foundation_unet3d(**kwargs: Any) -> FoundationUNet3D:
    return FoundationUNet3D(**kwargs)


@MODELS.register("octcube_unet3d")
def _octcube_unet3d(**kwargs: Any) -> OCTCubeUNet3D:
    return OCTCubeUNet3D(**kwargs)


def build_model(config: dict[str, Any], in_channels: int, modalities: list[str] | None = None):
    options = dict(config)
    name = options.pop("name")
    options.pop("in_channels", None)
    options.pop("inference_steps", None)
    if name == "octcube_unet3d":
        if modalities is None:
            raise ValueError("modalities are required to build octcube_unet3d")
        options["modalities"] = modalities
    return MODELS.build(name, in_channels=in_channels, **options)


__all__ = ["DiffusionUNet3D", "FoundationUNet3D", "OCTCubeUNet3D", "UNet3D", "build_model"]
