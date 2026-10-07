from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from scipy import ndimage


def _binary_structure(dimensions: int, connectivity: int) -> np.ndarray:
    valid = {2: {4: 1, 8: 2}, 3: {6: 1, 18: 2, 26: 3}}
    if dimensions not in valid or connectivity not in valid[dimensions]:
        choices = sorted(valid.get(dimensions, {}))
        raise ValueError(
            f"Unsupported {dimensions}D connectivity {connectivity}; choose from {choices}"
        )
    return ndimage.generate_binary_structure(dimensions, valid[dimensions][connectivity])


def _longest_consecutive_run(indices: np.ndarray) -> int:
    if not indices.size:
        return 0
    ordered = np.unique(indices.astype(np.int64, copy=False))
    breaks = np.flatnonzero(np.diff(ordered) != 1)
    boundaries = np.concatenate(([-1], breaks, [ordered.size - 1]))
    return int(np.diff(boundaries).max())


def postprocess_rnv_mask(
    mask: np.ndarray,
    *,
    minimum_2d_pixels: int = 30,
    minimum_consecutive_bscans: int = 5,
    connectivity_2d: int = 8,
    connectivity_3d: int = 26,
) -> np.ndarray:
    """Remove small 2D regions and lesions not persisting across consecutive B-scans."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError("RNV mask must have shape [depth, width, B-scan]")
    minimum_2d_pixels = int(minimum_2d_pixels)
    minimum_consecutive_bscans = int(minimum_consecutive_bscans)
    if minimum_2d_pixels < 1 or minimum_consecutive_bscans < 1:
        raise ValueError("Postprocessing thresholds must be positive integers")

    structure_2d = _binary_structure(2, int(connectivity_2d))
    cleaned = np.zeros_like(binary)
    for bscan in range(binary.shape[2]):
        labels, count = ndimage.label(binary[:, :, bscan], structure=structure_2d)
        if not count:
            continue
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep = sizes >= minimum_2d_pixels
        keep[0] = False
        cleaned[:, :, bscan] = keep[labels]

    structure_3d = _binary_structure(3, int(connectivity_3d))
    output = np.zeros_like(cleaned)
    occupied_axes = [
        np.flatnonzero(cleaned.any(axis=axes))
        for axes in ((1, 2), (0, 2), (0, 1))
    ]
    if not all(indices.size for indices in occupied_axes):
        return output.astype(np.uint8)
    bounds = tuple(
        slice(int(indices[0]), int(indices[-1]) + 1) for indices in occupied_axes
    )
    cropped = cleaned[bounds]
    labels, count = ndimage.label(cropped, structure=structure_3d)
    objects = ndimage.find_objects(labels, max_label=count)
    for component, component_bounds in enumerate(objects, start=1):
        if component_bounds is None:
            continue
        local_component = labels[component_bounds] == component
        local_bscans = np.flatnonzero(local_component.any(axis=(0, 1)))
        if _longest_consecutive_run(local_bscans) < minimum_consecutive_bscans:
            continue
        global_bounds = tuple(
            slice(
                bounds[axis].start + component_bounds[axis].start,
                bounds[axis].start + component_bounds[axis].stop,
            )
            for axis in range(3)
        )
        output[global_bounds] |= local_component
    return output.astype(np.uint8)


def postprocess_vasculature_mask(
    mask: np.ndarray,
    *,
    minimum_3d_voxels: int = 10,
    minimum_consecutive_bscans: int = 2,
    connectivity_3d: int = 26,
) -> np.ndarray:
    """Remove tiny 3D vessel fragments without applying an aggressive 2D area filter."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError("Vasculature mask must have shape [depth, width, B-scan]")
    structure = _binary_structure(3, int(connectivity_3d))
    labels, count = ndimage.label(binary, structure=structure)
    output = np.zeros_like(binary)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    for component in range(1, count + 1):
        if sizes[component] < int(minimum_3d_voxels):
            continue
        locations = np.argwhere(labels == component)
        if _longest_consecutive_run(locations[:, 2]) < int(minimum_consecutive_bscans):
            continue
        output[labels == component] = True
    return output.astype(np.uint8)


def postprocess_prediction_masks(
    raw_masks: Mapping[str, np.ndarray],
    inference_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Apply the configured test-time cleanup to all prediction heads.

    Keeping this in the inference engine ensures that checkpoint selection and
    final test inference use exactly the same postprocessing implementation.
    """
    if "rnv" not in raw_masks:
        raise KeyError("Prediction masks must contain an 'rnv' output")
    postprocessing = inference_config.get("postprocessing", {})
    enabled = bool(postprocessing.get("enabled", True))
    masks: dict[str, np.ndarray] = {}
    masks["rnv"] = (
        postprocess_rnv_mask(
            raw_masks["rnv"],
            minimum_2d_pixels=int(
                postprocessing.get("minimum_2d_component_pixels", 30)
            ),
            minimum_consecutive_bscans=int(
                postprocessing.get("minimum_consecutive_bscans", 5)
            ),
            connectivity_2d=int(postprocessing.get("connectivity_2d", 8)),
            connectivity_3d=int(postprocessing.get("connectivity_3d", 26)),
        )
        if enabled
        else np.asarray(raw_masks["rnv"], dtype=np.uint8)
    )
    if "vasculature" in raw_masks:
        vessel_post = postprocessing.get("vasculature", {})
        masks["vasculature"] = (
            postprocess_vasculature_mask(
                raw_masks["vasculature"],
                minimum_3d_voxels=int(
                    vessel_post.get("minimum_3d_component_voxels", 10)
                ),
                minimum_consecutive_bscans=int(
                    vessel_post.get("minimum_consecutive_bscans", 2)
                ),
                connectivity_3d=int(vessel_post.get("connectivity_3d", 26)),
            )
            if enabled
            else np.asarray(raw_masks["vasculature"], dtype=np.uint8)
        )
        masks["vasculature"] &= masks["rnv"]
    for name, mask in raw_masks.items():
        masks.setdefault(name, np.asarray(mask, dtype=np.uint8))
    return masks


@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Sequence[int],
    overlap: float = 0.5,
    amp: bool = False,
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
    output: torch.Tensor | None = None
    counts: torch.Tensor | None = None
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
                with torch.autocast(
                    device_type=model_device.type,
                    enabled=bool(amp and model_device.type == "cuda"),
                ):
                    logits = model(
                        window.to(
                            model_device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                    ).logits
                logits = logits.float().cpu()
                if output is None:
                    output = torch.zeros(
                        (1, logits.shape[1], *spatial), device="cpu", dtype=torch.float32
                    )
                    counts = torch.zeros_like(output)
                assert counts is not None
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
    if output is None or counts is None:
        raise RuntimeError("Sliding-window inference produced no windows")
    return output / counts.clamp_min(1)
