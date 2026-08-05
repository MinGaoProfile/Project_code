from __future__ import annotations

from collections.abc import Sequence

import torch


@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Sequence[int],
    overlap: float = 0.5,
) -> torch.Tensor:
    """Overlap-average inference for a single [C,D,H,W] volume."""
    if image.ndim != 4:
        raise ValueError("image must have shape [C,D,H,W]")
    spatial = image.shape[-3:]
    patch = tuple(min(int(size), int(full)) for size, full in zip(patch_size, spatial, strict=True))
    strides = tuple(max(1, round(size * (1.0 - overlap))) for size in patch)

    def starts(full: int, window: int, stride: int) -> list[int]:
        values = list(range(0, max(full - window, 0) + 1, stride))
        final = max(0, full - window)
        if not values or values[-1] != final:
            values.append(final)
        return values

    model_device = next(model.parameters()).device
    output = torch.zeros((1, 1, *spatial), device="cpu", dtype=torch.float32)
    counts = torch.zeros_like(output)
    for depth in starts(spatial[0], patch[0], strides[0]):
        for height in starts(spatial[1], patch[1], strides[1]):
            for width in starts(spatial[2], patch[2], strides[2]):
                window = image[
                    None,
                    :,
                    depth : depth + patch[0],
                    height : height + patch[1],
                    width : width + patch[2],
                ]
                logits = model(window.to(model_device, non_blocking=True)).logits.float().cpu()
                output[
                    :,
                    :,
                    depth : depth + patch[0],
                    height : height + patch[1],
                    width : width + patch[2],
                ] += logits
                counts[
                    :,
                    :,
                    depth : depth + patch[0],
                    height : height + patch[1],
                    width : width + patch[2],
                ] += 1
    return output / counts.clamp_min(1)
