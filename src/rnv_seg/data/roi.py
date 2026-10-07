from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import ndimage
from scipy.io import loadmat

from .discovery import ScanRecord


@dataclass(frozen=True)
class PositivePatchCandidate:
    start: tuple[int, int, int]
    component_id: int
    component_voxels: int
    view_type: str
    component_coverage: float
    foreground_voxels: int
    is_base: bool


def curve_roi_enabled(config: dict[str, object]) -> bool:
    return bool(config.get("curve_roi", {}).get("enabled", False))


def load_retinal_surfaces(
    record: ScanRecord,
    config: dict[str, object],
    volume_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load zero-based ILM and BM surfaces on the lateral plane."""
    roi = config.get("curve_roi", {})
    if tuple(config.get("axis_order", [0, 1, 2])) != (0, 1, 2):
        raise ValueError("curve_roi currently requires data.axis_order: [0, 1, 2]")
    path = record.segmentation_path
    if path is None:
        if bool(roi.get("required", True)):
            raise FileNotFoundError(f"Curve segmentation is required for sample {record.sample_id}")
        path = record.image_path
    variables = config.get("mat_variables", {})
    name = str(roi.get("variable", variables.get("segmentation", "ManualCurveData")))
    payload = loadmat(path, variable_names=[name])
    if name not in payload:
        raise KeyError(f"Curve variable {name!r} is missing from {path}")
    curves = np.asarray(payload[name])
    ilm_layer = int(roi.get("ilm_layer", 1)) - 1
    bm_layer = int(roi.get("bm_layer", 8)) - 1
    if curves.ndim != 3 or not all(
        0 <= layer < curves.shape[1] for layer in (ilm_layer, bm_layer)
    ):
        raise ValueError(
            "Expected curve array [lateral, layers, B-scans] with "
            f"ILM layer {ilm_layer + 1} and BM layer {bm_layer + 1}; got {curves.shape}"
        )
    expected = (int(volume_shape[1]), int(volume_shape[2]))
    coordinate_base = int(roi.get("coordinate_base", 1))

    def prepare(layer: int, name: str) -> np.ndarray:
        surface = curves[:, layer, :].astype(np.float32, copy=False)
        if surface.shape != expected:
            raise ValueError(
                f"{name} surface shape {surface.shape} does not match lateral shape {expected}"
            )
        valid = np.isfinite(surface) & (surface >= coordinate_base) & (
            surface < int(volume_shape[0]) + coordinate_base
        )
        if not valid.any():
            raise ValueError(
                f"{name} surface contains no valid coordinates for sample {record.sample_id}"
            )
        if not valid.all():
            nearest = ndimage.distance_transform_edt(
                ~valid, return_distances=False, return_indices=True
            )
            surface = surface[tuple(nearest)]
        surface = np.rint(surface - coordinate_base)
        return np.clip(surface, 0, int(volume_shape[0]) - 1).astype(
            np.int32, copy=False
        )

    return prepare(ilm_layer, "ILM"), prepare(bm_layer, "BM")


def load_bm_surface(
    record: ScanRecord,
    config: dict[str, object],
    volume_shape: Sequence[int],
) -> np.ndarray:
    """Backward-compatible BM-only loader."""
    return load_retinal_surfaces(record, config, volume_shape)[1]


def shifted_curve_roi_surfaces(
    target: np.ndarray | None,
    ilm_surface: np.ndarray,
    bm_surface: np.ndarray,
    volume_depth: int,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return shifted ILM upper and BM lower-exclusive surfaces for the curved ROI."""
    roi = config.get("curve_roi", {})
    top_margin = int(roi.get("rnv_top_margin_pixels", 20))
    bottom_margin = int(roi.get("bm_bottom_margin_pixels", 100))
    ilm_shift = 0
    if target is not None and np.any(target > 0):
        rnv_depths = np.flatnonzero(np.any(target > 0, axis=(1, 2)))
        highest_depth = int(rnv_depths[0])
        highest_locations = np.argwhere(target[highest_depth] > 0)
        desired_upper = max(0, highest_depth - top_margin)
        ilm_at_highest = ilm_surface[
            highest_locations[:, 0], highest_locations[:, 1]
        ]
        ilm_shift = max(0, int(np.max(ilm_at_highest)) - desired_upper)
    upper = np.clip(ilm_surface.astype(np.int32) - ilm_shift, 0, volume_depth - 1)
    lower = np.clip(
        bm_surface.astype(np.int32) + bottom_margin + 1,
        1,
        volume_depth,
    )
    lower = np.maximum(lower, upper + 1)
    return upper, lower, ilm_shift


def curved_roi_bounds(
    upper_surface: np.ndarray,
    lower_surface: np.ndarray,
    volume_depth: int,
    minimum_depth: int = 1,
) -> tuple[int, int]:
    """Return rectangular bounds enclosing the curved ROI for fixed-size cropping."""
    top = max(0, int(upper_surface.min()))
    bottom = min(int(volume_depth), int(lower_surface.max()))
    if bottom - top < minimum_depth:
        top = max(0, bottom - minimum_depth)
        bottom = min(int(volume_depth), max(bottom, top + minimum_depth))
        top = max(0, bottom - minimum_depth)
    if bottom <= top:
        raise ValueError(f"Invalid curved ROI bounds: top={top}, bottom={bottom}")
    return top, bottom


def curved_roi_patch_mask(
    upper_surface: np.ndarray,
    lower_surface: np.ndarray,
    patch_slices: Sequence[slice],
) -> np.ndarray:
    """Return the curved ROI mask for one [depth, width, B-scan] patch."""
    depth_slice, width_slice, bscan_slice = patch_slices
    depth = np.arange(depth_slice.start, depth_slice.stop, dtype=np.int32)[:, None, None]
    upper = upper_surface[width_slice, bscan_slice][None]
    lower = lower_surface[width_slice, bscan_slice][None]
    return (depth >= upper) & (depth < lower)


def curve_roi_bounds(
    target: np.ndarray | None,
    bm_surface: np.ndarray,
    volume_depth: int,
    config: dict[str, object],
    minimum_depth: int = 1,
) -> tuple[int, int]:
    """Return a rectangular [top, bottom) ROI around RNV and the deepest BM point."""
    roi = config.get("curve_roi", {})
    top_margin = int(roi.get("rnv_top_margin_pixels", 20))
    bottom_margin = int(roi.get("bm_bottom_margin_pixels", 100))
    bottom = min(int(volume_depth), int(bm_surface.max()) + bottom_margin + 1)
    if target is not None and np.any(target > 0):
        top = max(0, int(np.flatnonzero(np.any(target > 0, axis=(1, 2)))[0]) - top_margin)
    else:
        top = 0
    if bottom - top < minimum_depth:
        top = max(0, bottom - minimum_depth)
        bottom = min(int(volume_depth), max(bottom, top + minimum_depth))
        top = max(0, bottom - minimum_depth)
    if bottom <= top:
        raise ValueError(f"Invalid curve ROI bounds: top={top}, bottom={bottom}")
    return top, bottom


def rnv_component_anchors(
    target: np.ndarray,
    patch_size: Sequence[int],
    overlap: float = 0.5,
    connectivity: int = 26,
    minimum_voxels: int = 1,
    roi_surfaces: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[tuple[int, int, int]]:
    """Return foreground anchors that cover every 3D RNV component."""
    components = _rnv_component_locations(
        target,
        connectivity=connectivity,
        minimum_voxels=minimum_voxels,
        roi_surfaces=roi_surfaces,
    )
    anchors = [
        anchor
        for _, locations in components
        for anchor in _component_anchors(locations, patch_size, overlap)
    ]
    return list(dict.fromkeys(anchors))


def _rnv_component_locations(
    target: np.ndarray,
    *,
    connectivity: int,
    minimum_voxels: int,
    roi_surfaces: tuple[np.ndarray, np.ndarray] | None,
) -> list[tuple[int, np.ndarray]]:
    """Return global foreground coordinates for eligible curved-ROI components."""
    foreground = np.asarray(target) > 0
    locations = np.argwhere(foreground)
    if roi_surfaces is not None and locations.size:
        upper, lower = roi_surfaces
        lateral_upper = upper[locations[:, 1], locations[:, 2]]
        lateral_lower = lower[locations[:, 1], locations[:, 2]]
        locations = locations[
            (locations[:, 0] >= lateral_upper) & (locations[:, 0] < lateral_lower)
        ]
    if not locations.size:
        return []
    lower = locations.min(axis=0)
    upper = locations.max(axis=0) + 1
    crop_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper, strict=True))
    cropped = foreground[crop_slices].copy()
    if roi_surfaces is not None:
        cropped &= curved_roi_patch_mask(*roi_surfaces, crop_slices)
    if connectivity not in {6, 18, 26}:
        raise ValueError("RNV component connectivity must be 6, 18, or 26")
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    labels, count = ndimage.label(cropped, structure=structure)
    objects = ndimage.find_objects(labels, max_label=count)
    components: list[tuple[int, np.ndarray]] = []
    for component, component_bounds in enumerate(objects, start=1):
        if component_bounds is None:
            continue
        local_mask = labels[component_bounds] == component
        if int(local_mask.sum()) < minimum_voxels:
            continue
        component_locations = np.argwhere(local_mask)
        component_origin = np.array(
            [
                int(lower[axis]) + int(component_bounds[axis].start)
                for axis in range(3)
            ]
        )
        components.append((component, component_locations + component_origin))
    return components


def _component_anchors(
    component_locations: np.ndarray,
    patch_size: Sequence[int],
    overlap: float,
) -> list[tuple[int, int, int]]:
    """Return one or more foreground anchors spanning a single component."""
    stride = tuple(max(1, round(int(size) * (1.0 - overlap))) for size in patch_size)
    anchors: list[tuple[int, int, int]] = []
    desired_axes: list[list[int]] = []
    for axis, patch in enumerate(patch_size):
        axis_low = int(component_locations[:, axis].min())
        axis_high = int(component_locations[:, axis].max()) + 1
        if axis_high - axis_low <= int(patch):
            desired_axes.append([(axis_low + axis_high - 1) // 2])
            continue
        values = list(range(axis_low, axis_high, stride[axis]))
        if values[-1] != axis_high - 1:
            values.append(axis_high - 1)
        desired_axes.append(values)
    for desired in product(*desired_axes):
        point = np.asarray(desired)
        nearest = int(np.square(component_locations - point).sum(axis=1).argmin())
        anchors.append(tuple(int(value) for value in component_locations[nearest]))
    return list(dict.fromkeys(anchors))


def _positive_patch_candidate(
    start: tuple[int, int, int],
    component_id: int,
    locations: np.ndarray,
    patch_size: Sequence[int],
    minimum_voxels: int,
    complete_minimum_coverage: float,
    boundary_minimum_coverage: float,
    is_base: bool,
) -> PositivePatchCandidate | None:
    lower = np.asarray(start)
    upper = lower + np.asarray(patch_size)
    inside = np.all((locations >= lower) & (locations < upper), axis=1)
    foreground_voxels = int(inside.sum())
    coverage = foreground_voxels / max(len(locations), 1)
    if foreground_voxels < minimum_voxels:
        return None
    if not is_base and coverage < boundary_minimum_coverage:
        return None
    view_type = (
        "complete"
        if is_base and coverage >= complete_minimum_coverage
        else "boundary"
    )
    return PositivePatchCandidate(
        start=start,
        component_id=component_id,
        component_voxels=int(len(locations)),
        view_type=view_type,
        component_coverage=coverage,
        foreground_voxels=foreground_voxels,
        is_base=is_base,
    )


def rnv_positive_patch_candidates(
    target: np.ndarray,
    shape: Sequence[int],
    patch_size: Sequence[int],
    depth_bounds: tuple[int, int],
    *,
    depth_anchor_fraction: float = 1.0 / 3.0,
    component_overlap: float = 0.5,
    connectivity: int = 26,
    minimum_voxels: int = 1,
    roi_surfaces: tuple[np.ndarray, np.ndarray] | None = None,
    lateral_offset_fraction: float = 0.25,
    bscan_offset_fraction: float = 0.25,
    complete_minimum_coverage: float = 0.8,
    boundary_minimum_coverage: float = 0.2,
    maximum_views_per_component: int = 6,
    maximum_views_per_case: int = 20,
) -> list[PositivePatchCandidate]:
    """Generate diverse base and shifted positive views for every RNV component."""
    shape = tuple(int(value) for value in shape)
    patch_size = tuple(int(value) for value in patch_size)
    if not 0.0 <= boundary_minimum_coverage <= complete_minimum_coverage <= 1.0:
        raise ValueError("Positive patch coverage thresholds must satisfy 0 <= boundary <= complete <= 1")
    if not 0.0 <= lateral_offset_fraction <= 1.0:
        raise ValueError("lateral_offset_fraction must be in [0, 1]")
    if not 0.0 <= bscan_offset_fraction <= 1.0:
        raise ValueError("bscan_offset_fraction must be in [0, 1]")
    if maximum_views_per_component < 1 or maximum_views_per_case < 1:
        raise ValueError("Positive patch view limits must be positive")

    components = _rnv_component_locations(
        target,
        connectivity=connectivity,
        minimum_voxels=minimum_voxels,
        roi_surfaces=roi_surfaces,
    )
    width_offset = max(1, round(patch_size[1] * lateral_offset_fraction))
    bscan_offset = max(1, round(patch_size[2] * bscan_offset_fraction))
    shifted_views = (
        (-width_offset, 0),
        (width_offset, 0),
        (0, -bscan_offset),
        (0, bscan_offset),
        (-width_offset, -bscan_offset),
        (width_offset, bscan_offset),
    )

    case_candidates: list[PositivePatchCandidate] = []
    for component_id, locations in components:
        anchors = _component_anchors(locations, patch_size, component_overlap)
        base_starts = covering_patch_starts(
            anchors,
            shape,
            patch_size,
            depth_bounds,
            depth_anchor_fraction,
        )
        seen: set[tuple[int, int, int]] = set()
        base_candidates: list[PositivePatchCandidate] = []
        extra_candidates: list[PositivePatchCandidate] = []
        candidate_specs = [(start, True) for start in base_starts]
        for width_shift, bscan_shift in shifted_views:
            candidate_specs.extend(
                (
                    (
                        base_start[0],
                        min(
                            max(0, base_start[1] + width_shift),
                            shape[1] - patch_size[1],
                        ),
                        min(
                            max(0, base_start[2] + bscan_shift),
                            shape[2] - patch_size[2],
                        ),
                    ),
                    False,
                )
                for base_start in base_starts
            )
        for start, is_base in candidate_specs:
            if start in seen:
                continue
            seen.add(start)
            candidate = _positive_patch_candidate(
                start,
                component_id,
                locations,
                patch_size,
                minimum_voxels,
                complete_minimum_coverage,
                boundary_minimum_coverage,
                is_base,
            )
            if candidate is None:
                continue
            (base_candidates if is_base else extra_candidates).append(candidate)

        target_count = max(maximum_views_per_component, len(base_candidates))
        case_candidates.extend(
            base_candidates
            + extra_candidates[: max(0, target_count - len(base_candidates))]
        )

    unique: dict[tuple[int, int, int], PositivePatchCandidate] = {}
    for candidate in case_candidates:
        existing = unique.get(candidate.start)
        if existing is None or (candidate.is_base and not existing.is_base):
            unique[candidate.start] = candidate
    candidates = list(unique.values())
    base_candidates = [candidate for candidate in candidates if candidate.is_base]
    extra_candidates = [candidate for candidate in candidates if not candidate.is_base]
    target_count = max(maximum_views_per_case, len(base_candidates))
    return base_candidates + extra_candidates[: max(0, target_count - len(base_candidates))]


def covering_patch_starts(
    anchors: Sequence[tuple[int, int, int]],
    shape: Sequence[int],
    patch_size: Sequence[int],
    depth_bounds: tuple[int, int],
    depth_anchor_fraction: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Greedily merge nearby lesion anchors into the fewest practical windows."""
    if not 0.0 < depth_anchor_fraction < 1.0:
        raise ValueError("depth_anchor_fraction must be between 0 and 1")
    remaining = set(anchors)
    starts: list[tuple[int, int, int]] = []
    while remaining:
        anchor = min(remaining, key=lambda point: (point[2], point[1], point[0]))
        current_start: list[int] = []
        for axis, (current, patch) in enumerate(zip(shape, patch_size, strict=True)):
            lower = depth_bounds[0] if axis == 0 else 0
            upper = depth_bounds[1] if axis == 0 else int(current)
            maximum = max(lower, upper - int(patch))
            anchor_offset = (
                round(int(patch) * depth_anchor_fraction)
                if axis == 0
                else int(patch) // 2
            )
            current_start.append(
                min(max(lower, anchor[axis] - anchor_offset), maximum)
            )
        start = tuple(current_start)
        starts.append(start)
        updated = {
            point
            for point in remaining
            if not all(
                start[axis] <= point[axis] < start[axis] + int(patch_size[axis])
                for axis in range(3)
            )
        }
        if len(updated) == len(remaining):
            raise ValueError(
                f"Patch at {start} cannot cover RNV anchor {anchor} inside depth bounds "
                f"{depth_bounds}"
            )
        remaining = updated
    return starts


def roi_sliding_window_starts(
    shape: Sequence[int],
    patch_size: Sequence[int],
    depth_bounds: tuple[int, int],
    *,
    overlap: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Return deterministic 3D windows distributed across the rectangular ROI.

    With zero overlap, full-size windows are spaced across each available axis without
    intersecting one another. This represents the entire spatial extent without adding
    an overlapping end window when an axis length is not divisible by the patch size.
    """
    shape = tuple(int(value) for value in shape)
    patch_size = tuple(int(value) for value in patch_size)
    if len(shape) != 3 or len(patch_size) != 3:
        raise ValueError("shape and patch_size must contain three values")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    regions = (
        (int(depth_bounds[0]), int(depth_bounds[1])),
        (0, shape[1]),
        (0, shape[2]),
    )
    axes: list[list[int]] = []
    for axis in range(3):
        lower, upper = regions[axis]
        extent = upper - lower
        if extent < patch_size[axis]:
            raise ValueError(
                f"ROI extent {extent} on axis {axis} is smaller than patch size "
                f"{patch_size[axis]}"
            )
        maximum = upper - patch_size[axis]
        if overlap == 0.0:
            count = max(1, extent // patch_size[axis])
            if count == 1:
                values = [lower + (extent - patch_size[axis]) // 2]
            else:
                values = [
                    int(round(value))
                    for value in np.linspace(lower, maximum, count)
                ]
            if any(
                right - left < patch_size[axis]
                for left, right in zip(values, values[1:], strict=False)
            ):
                raise RuntimeError("Zero-overlap ROI windows unexpectedly intersect")
            axes.append(values)
            continue
        stride = max(1, round(patch_size[axis] * (1.0 - overlap)))
        values = list(range(lower, maximum + 1, stride))
        if not values or values[-1] != maximum:
            values.append(maximum)
        axes.append(values)
    return [tuple(int(value) for value in start) for start in product(*axes)]


def roi_stratified_window_candidates(
    shape: Sequence[int],
    patch_size: Sequence[int],
    depth_bounds: tuple[int, int],
    *,
    spatial_phases: Sequence[float] = (0.0, 0.5, 1.0),
) -> list[tuple[tuple[int, int, int], tuple[int, int]]]:
    """Return alternative depth/width positions for non-overlapping B-scan slots.

    Each returned item is ``(start, slot)``. Candidates sharing a slot are spatial
    alternatives and must not be selected together. Candidates from different slots
    have disjoint B-scan intervals, so one candidate per slot remains non-overlapping.
    """
    shape = tuple(int(value) for value in shape)
    patch_size = tuple(int(value) for value in patch_size)
    if len(shape) != 3 or len(patch_size) != 3:
        raise ValueError("shape and patch_size must contain three values")
    phases = tuple(float(value) for value in spatial_phases)
    if not phases or any(not 0.0 <= value <= 1.0 for value in phases):
        raise ValueError("spatial_phases must be non-empty values in [0, 1]")

    def phased_starts(lower: int, upper: int, patch: int) -> list[int]:
        maximum = upper - patch
        if maximum < lower:
            raise ValueError(f"Region [{lower}, {upper}) is smaller than patch {patch}")
        return sorted(
            {
                int(round(lower + phase * (maximum - lower)))
                for phase in phases
            }
        )

    depth_starts = phased_starts(
        int(depth_bounds[0]), int(depth_bounds[1]), patch_size[0]
    )
    width_starts = phased_starts(0, shape[1], patch_size[1])
    base = roi_sliding_window_starts(
        shape, patch_size, depth_bounds, overlap=0.0
    )
    bscan_starts = sorted({start[2] for start in base})
    candidates: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for slot_index, bscan_start in enumerate(bscan_starts):
        slot = (slot_index, bscan_start)
        candidates.extend(
            ((depth_start, width_start, bscan_start), slot)
            for depth_start, width_start in product(depth_starts, width_starts)
        )
    return candidates


def choose_curve_roi_patch_slices(
    shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.Generator,
    depth_bounds: tuple[int, int],
    foreground_anchor: tuple[int, int, int] | None = None,
    foreground: np.ndarray | None = None,
    foreground_probability: float = 0.0,
    random_crop: bool = True,
    depth_anchor_fraction: float = 0.5,
) -> tuple[slice, slice, slice]:
    """Sample a 3D patch inside the scan ROI, optionally centred on an RNV voxel."""
    shape = tuple(int(value) for value in shape)
    patch_size = tuple(int(value) for value in patch_size)
    if not 0.0 < depth_anchor_fraction < 1.0:
        raise ValueError("depth_anchor_fraction must be between 0 and 1")
    center = foreground_anchor
    if center is None and foreground is not None and rng.random() < foreground_probability:
        locations = np.argwhere(foreground > 0)
        if locations.size:
            selected = locations[rng.integers(len(locations))]
            center = tuple(int(value) for value in selected)
    starts: list[int] = []
    for axis, (current, patch) in enumerate(zip(shape, patch_size, strict=True)):
        lower = depth_bounds[0] if axis == 0 else 0
        upper = depth_bounds[1] if axis == 0 else current
        maximum = max(lower, upper - patch)
        if center is not None:
            axis_center = center[axis]
            if axis == 0:
                anchor_offset = round(patch * depth_anchor_fraction)
                jitter = 0
            else:
                anchor_offset = patch // 2
                jitter = int(rng.integers(-patch // 4, patch // 4 + 1)) if random_crop else 0
            start = min(max(lower, axis_center - anchor_offset + jitter), maximum)
        elif random_crop and maximum > lower:
            start = int(rng.integers(lower, maximum + 1))
        else:
            start = lower + (maximum - lower) // 2
        starts.append(start)
    return tuple(
        slice(start, start + patch)
        for start, patch in zip(starts, patch_size, strict=True)
    )
