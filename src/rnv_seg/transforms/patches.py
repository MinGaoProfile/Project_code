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
                target = np.flip(target, axis=spatial_axis)
    return np.ascontiguousarray(image), None if target is None else np.ascontiguousarray(target)
