from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class RNVLesionSummary:
    group: str
    total_voxels: int
    largest_component_voxels: int
    component_count: int


def rnv_component_sizes(
    target: np.ndarray,
    *,
    connectivity: int = 26,
    minimum_voxels: int = 30,
) -> np.ndarray:
    """Return eligible 3D component sizes without allocating a full-volume label map."""
    if connectivity not in {6, 18, 26}:
        raise ValueError("RNV component connectivity must be 6, 18, or 26")
    if int(minimum_voxels) < 1:
        raise ValueError("minimum_voxels must be positive")
    foreground = np.asarray(target) > 0
    occupied = [
        np.flatnonzero(foreground.any(axis=axes))
        for axes in ((1, 2), (0, 2), (0, 1))
    ]
    if not all(indices.size for indices in occupied):
        return np.empty(0, dtype=np.int64)
    bounds = tuple(
        slice(int(indices[0]), int(indices[-1]) + 1) for indices in occupied
    )
    cropped = np.asarray(foreground[bounds], dtype=np.uint8)
    structure = ndimage.generate_binary_structure(
        3, {6: 1, 18: 2, 26: 3}[int(connectivity)]
    )
    labels, count = ndimage.label(cropped, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return sizes[sizes >= int(minimum_voxels)].astype(np.int64, copy=False)


def summarize_rnv_lesions(
    target: np.ndarray,
    *,
    large_component_voxels: int,
    connectivity: int = 26,
    minimum_voxels: int = 30,
) -> RNVLesionSummary:
    """Classify a labelled scan from its largest complete 3D RNV component."""
    threshold = int(large_component_voxels)
    if threshold < int(minimum_voxels):
        raise ValueError(
            "large_component_voxels must be at least minimum_voxels"
        )
    total = int(np.count_nonzero(target))
    if total == 0:
        return RNVLesionSummary("rnv_negative", 0, 0, 0)
    sizes = rnv_component_sizes(
        target,
        connectivity=connectivity,
        minimum_voxels=minimum_voxels,
    )
    largest = int(sizes.max()) if sizes.size else 0
    group = "large_containing" if largest >= threshold else "small_only"
    return RNVLesionSummary(group, total, largest, int(sizes.size))
