from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pad_to_shape(array: np.ndarray, shape: Sequence[int], value: float = 0) -> np.ndarray:
    pads: list[tuple[int, int]] = []
    for current, target in zip(array.shape[-3:], shape, strict=True):
        total = max(0, int(target) - int(current))
        pads.append((total // 2, total - total // 2))
    prefix = [(0, 0)] * (array.ndim - 3)
    return np.pad(array, prefix + pads, mode="constant", constant_values=value)


def choose_patch_slices(
    shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.Generator,
    foreground: np.ndarray | None = None,
    foreground_probability: float = 0.0,
    random_crop: bool = True,
) -> tuple[slice, slice, slice]:
    shape = tuple(int(value) for value in shape)
    patch_size = tuple(int(value) for value in patch_size)
    center: tuple[int, int, int] | None = None
    if foreground is not None and rng.random() < foreground_probability:
        locations = np.argwhere(foreground > 0)
        if locations.size:
            center = tuple(int(value) for value in locations[rng.integers(len(locations))])
    starts: list[int] = []
    for axis, (current, patch) in enumerate(zip(shape, patch_size, strict=True)):
        maximum = max(0, current - patch)
        if center is not None:
            jitter = int(rng.integers(-patch // 4, patch // 4 + 1)) if random_crop else 0
            start = min(max(0, center[axis] - patch // 2 + jitter), maximum)
        elif random_crop and maximum:
            start = int(rng.integers(0, maximum + 1))
        else:
            start = maximum // 2
        starts.append(start)
    return tuple(slice(start, start + patch) for start, patch in zip(starts, patch_size, strict=True))


def apply_random_flips(
    image: np.ndarray,
    target: np.ndarray | None,
    rng: np.random.Generator,
    probability: float = 0.5,
    axes: Sequence[int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray | None]:
    if any(axis not in {0, 1, 2} for axis in axes):
        raise ValueError("Flip axes must be selected from 0, 1, and 2")
    for spatial_axis in axes:
        if rng.random() < probability:
            image = np.flip(image, axis=spatial_axis + 1)
            if target is not None:
                target = np.flip(target, axis=spatial_axis + target.ndim - 3)
    return np.ascontiguousarray(image), None if target is None else np.ascontiguousarray(target)


def apply_intensity_augmentations(
    image: np.ndarray,
    rng: np.random.Generator,
    channels: Sequence[int],
    *,
    contrast_probability: float = 0.5,
    contrast_range: Sequence[float] = (0.8, 1.2),
    noise_probability: float = 0.3,
    noise_std_range: Sequence[float] = (0.0, 0.05),
    spatial_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply contrast and Gaussian-noise augmentation to intensity channels only."""
    output = np.asarray(image, dtype=np.float32).copy()
    contrast_low, contrast_high = (float(value) for value in contrast_range)
    noise_low, noise_high = (float(value) for value in noise_std_range)
    if not 0 <= contrast_probability <= 1 or not 0 <= noise_probability <= 1:
        raise ValueError("Augmentation probabilities must be in [0, 1]")
    if contrast_low <= 0 or contrast_high < contrast_low:
        raise ValueError("contrast_range must be positive and ordered")
    if noise_low < 0 or noise_high < noise_low:
        raise ValueError("noise_std_range must be non-negative and ordered")
    mask = None if spatial_mask is None else np.asarray(spatial_mask, dtype=bool)
    for channel in channels:
        values = output[int(channel)]
        if rng.random() < contrast_probability:
            factor = float(rng.uniform(contrast_low, contrast_high))
            selected = values if mask is None else values[mask]
            mean = float(selected.mean()) if selected.size else 0.0
            values = np.clip((values - mean) * factor + mean, 0.0, 1.0)
        if rng.random() < noise_probability:
            standard_deviation = float(rng.uniform(noise_low, noise_high))
            noise = rng.normal(0.0, standard_deviation, size=values.shape).astype(
                np.float32
            )
            values = np.clip(values + noise, 0.0, 1.0)
        if mask is not None:
            values = np.where(mask, values, 0.0)
        output[int(channel)] = values
    return np.ascontiguousarray(output)
