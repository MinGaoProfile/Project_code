from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat
from torch.utils.data import Dataset

from rnv_seg.registry import DATASETS
from rnv_seg.transforms import (
    apply_intensity_augmentations,
    apply_random_flips,
    choose_patch_slices,
    pad_to_shape,
)

from .discovery import ScanRecord
from .roi import (
    choose_curve_roi_patch_slices,
    covering_patch_starts,
    curve_roi_enabled,
    curved_roi_bounds,
    load_retinal_surfaces,
    rnv_component_anchors,
    rnv_positive_patch_candidates,
    roi_stratified_window_candidates,
    shifted_curve_roi_surfaces,
)


def _locations_intersect_patch(
    locations: np.ndarray,
    start: Sequence[int],
    patch_size: Sequence[int],
    margin: Sequence[int] = (0, 0, 0),
) -> bool:
    """Return whether sparse foreground coordinates intersect a (possibly expanded) patch."""
    if locations.size == 0:
        return False
    lower = np.asarray(start, dtype=np.int64) - np.asarray(margin, dtype=np.int64)
    upper = (
        np.asarray(start, dtype=np.int64)
        + np.asarray(patch_size, dtype=np.int64)
        + np.asarray(margin, dtype=np.int64)
    )
    return bool(np.any(np.all((locations >= lower) & (locations < upper), axis=1)))


def _negative_patch_category(
    start: Sequence[int],
    patch_size: Sequence[int],
    rnv_locations: np.ndarray,
    adjacent_margin: Sequence[int],
) -> str:
    """Separate lesion-adjacent negatives from scan-wide anatomical coverage.

    Macula cannot be identified reliably from local retinal thickness because thin
    temporal retina can satisfy the same rule.  Non-adjacent patches therefore remain
    a neutral coverage category and are diversified by scan record and spatial slot.
    """
    if _locations_intersect_patch(
        rnv_locations,
        start,
        patch_size,
        adjacent_margin,
    ):
        return "adjacent"
    return "anatomical_coverage"


def _normalization_statistics(
    array: np.ndarray, method: str, percentiles: Sequence[float]
) -> tuple[float, float] | None:
    """Compute normalization statistics from a complete raw case volume."""
    if method == "none":
        return None
    if method == "minmax":
        return float(array.min()), float(array.max())
    if method == "percentile":
        low, high = np.percentile(array, percentiles)
        return float(low), float(high)
    if method == "zscore":
        return float(array.mean()), float(array.std())
    raise ValueError(f"Unknown normalization method: {method}")


def _normalize(
    array: np.ndarray,
    method: str,
    percentiles: Sequence[float],
    statistics: tuple[float, float] | None = None,
) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if method == "none":
        return array
    if statistics is None:
        statistics = _normalization_statistics(array, method, percentiles)
    if statistics is None:
        return array
    low, high = statistics
    if method == "zscore":
        return ((array - low) / max(high, 1e-6)).astype(np.float32, copy=False)
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def target_names_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(str(name) for name in config.get("target_names", ["rnv"]))
    if not names or len(set(names)) != len(names):
        raise ValueError("data.target_names must contain unique target names")
    return names


def _target_variable(config: dict[str, Any], name: str) -> str:
    variables = config["mat_variables"]
    key = "target" if name == "rnv" else f"{name}_target"
    if key not in variables:
        raise KeyError(f"data.mat_variables.{key} is required for target {name!r}")
    return str(variables[key])


def _load_target_stack(record: ScanRecord, config: dict[str, Any]) -> np.ndarray | None:
    if record.target_path is None:
        return None
    names = target_names_from_config(config)
    variable_names = [_target_variable(config, name) for name in names]
    payload = loadmat(record.target_path, variable_names=variable_names)
    missing = [name for name in variable_names if name not in payload]
    if missing:
        raise KeyError(
            f"Missing target variable(s) {missing} in {record.target_path} for sample {record.sample_id}"
        )
    axis_order = tuple(config.get("axis_order", [0, 1, 2]))
    targets = [np.transpose(np.asarray(payload[name]), axis_order) for name in variable_names]
    if len({target.shape for target in targets}) != 1:
        raise ValueError(f"Target shapes do not match in {record.target_path}")
    stack = np.stack(targets, axis=0)
    if config.get("require_vasculature_within_rnv", False) and "vasculature" in names:
        rnv = stack[names.index("rnv")] > 0
        vasculature = stack[names.index("vasculature")] > 0
        outside = int(np.logical_and(vasculature, ~rnv).sum())
        if outside:
            raise ValueError(
                f"vasculature_mask has {outside} voxels outside RNV mask in sample {record.sample_id}"
            )
    return stack


def _center_pad_or_crop_2d(array: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    result = array
    for axis, desired in enumerate(target):
        current = result.shape[axis]
        if current > desired:
            start = (current - desired) // 2
            slices = [slice(None), slice(None)]
            slices[axis] = slice(start, start + desired)
            result = result[tuple(slices)]
    pads = []
    for current, desired in zip(result.shape, target, strict=True):
        total = max(0, desired - current)
        pads.append((total // 2, total - total // 2))
    return np.pad(result, pads, mode="constant")


def _align_vessel_plane(
    array: np.ndarray, target: tuple[int, int], alignment: str
) -> np.ndarray:
    if alignment == "exact":
        if array.shape != target:
            raise ValueError(
                f"Vessel prior shape {array.shape} does not exactly match lateral volume shape {target}"
            )
        return array
    if alignment == "center_pad_or_crop":
        return _center_pad_or_crop_2d(array, target)
    raise ValueError(f"Unknown vessel-prior alignment method: {alignment}")


def _combine_vessel_channels(
    rgb: np.ndarray,
    channels: Sequence[int],
    threshold: float = 0.0,
) -> np.ndarray:
    """Combine color-coded vessel classes as a binary union."""
    selected = tuple(int(channel) for channel in channels)
    if not selected or any(channel not in {0, 1, 2} for channel in selected):
        raise ValueError("Vessel RGB channels must be a non-empty subset of [0, 1, 2]")
    combined = np.max(rgb[..., list(selected)], axis=-1)
    return (combined > float(threshold)).astype(np.float32)


def _remove_small_vessel_components(
    mask: np.ndarray,
    minimum_pixels: int,
    connectivity: int = 8,
) -> np.ndarray:
    """Remove small connected regions from a 2D binary vessel mask."""
    minimum_pixels = int(minimum_pixels)
    binary = np.asarray(mask) > 0
    if minimum_pixels <= 1:
        return binary.astype(np.float32)
    if connectivity not in {4, 8}:
        raise ValueError("Vessel component connectivity must be 4 or 8")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = ndimage.label(binary, structure=structure)
    if not count:
        return np.zeros_like(mask, dtype=np.float32)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels].astype(np.float32)


def _prepare_vessel_plane(
    rgb: np.ndarray,
    channels: Sequence[int],
    config: dict[str, Any],
) -> np.ndarray:
    binary = _combine_vessel_channels(
        rgb,
        channels,
        float(config.get("binary_threshold", 0.0)),
    )
    return _remove_small_vessel_components(
        binary,
        int(config.get("minimum_component_pixels", 1)),
        int(config.get("component_connectivity", 8)),
    )


def _vessel_channels(config: dict[str, Any], modality: str) -> tuple[int, ...]:
    combined_key = f"{modality}_channels"
    if combined_key in config:
        return tuple(int(channel) for channel in config[combined_key])
    return (int(config[f"{modality}_channel"]),)


def _vessel_plane_for_modality(
    rgb: np.ndarray,
    modality: str,
    config: dict[str, Any],
) -> np.ndarray:
    if modality == "large_vessel":
        artery = _prepare_vessel_plane(rgb, _vessel_channels(config, "artery"), config)
        vein = _prepare_vessel_plane(rgb, _vessel_channels(config, "vein"), config)
        return np.maximum(artery, vein)
    return _prepare_vessel_plane(rgb, _vessel_channels(config, modality), config)


def _vessel_volume_patch(
    path: Path,
    modality: str,
    volume_shape: tuple[int, int, int],
    patch_slices: tuple[slice, slice, slice],
    depth_axis: int,
    alignment: str,
    vessel_config: dict[str, Any],
) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    lateral_axes = [axis for axis in range(3) if axis != depth_axis]
    target_plane = (volume_shape[lateral_axes[0]], volume_shape[lateral_axes[1]])
    plane = _align_vessel_plane(
        _vessel_plane_for_modality(rgb, modality, vessel_config),
        target_plane,
        alignment,
    )
    plane_slices = (patch_slices[lateral_axes[0]], patch_slices[lateral_axes[1]])
    plane_patch = plane[plane_slices]
    expanded = np.expand_dims(plane_patch, axis=depth_axis)
    return np.repeat(expanded, patch_slices[depth_axis].stop - patch_slices[depth_axis].start, axis=depth_axis)


@DATASETS.register("rnv_mat")
class RNVMatDataset(Dataset[dict[str, Any]]):
    """Lazy MATLAB reader that crops before float conversion to control memory use."""

    def __init__(
        self,
        records: Sequence[ScanRecord],
        config: dict[str, Any],
        training: bool,
    ) -> None:
        self.records = list(records)
        self.config = config
        self.training = training
        self.modalities = list(config["modalities"])
        self.target_names = target_names_from_config(config)
        self.variables = config["mat_variables"]
        self.axis_order = tuple(config.get("axis_order", [0, 1, 2]))
        self.patch_size = tuple(int(value) for value in config["patch_size"])
        self.samples_per_volume = int(config.get("samples_per_volume", 1)) if training else 1
        self.foreground_probability = float(config.get("foreground_patch_probability", 0.0))
        self.normalize = config.get("normalize", "percentile")
        self.percentiles = config.get("percentile_range", [0.5, 99.5])
        self.vessel = config.get("vessel_prior", {})
        self.random_flip_axes = tuple(int(axis) for axis in config.get("random_flip_axes", [0, 1, 2]))
        self.random_flip_probability = float(config.get("random_flip_probability", 0.5))
        self.intensity_augmentation = config.get("intensity_augmentation", {})
        self.intensity_channels = tuple(
            index
            for index, modality in enumerate(self.modalities)
            if modality in {"oct", "octa"}
        )
        self._normalization_statistics_cache: dict[
            tuple[str, str], tuple[float, float] | None
        ] = {}
        self.use_curve_roi = curve_roi_enabled(config)
        self.depth_anchor_fraction = float(
            config.get("curve_roi", {}).get("rnv_depth_position_fraction", 0.40)
        )
        if self.use_curve_roi and 0 in self.random_flip_axes:
            raise ValueError("curve_roi preserves vitreous depth, so random_flip_axes must not include 0")
        self.patch_requests: list[tuple[int, tuple[int, int, int] | None]] | None = None
        self.patch_request_types: list[str] | None = None
        self.patch_request_slots: list[tuple[int, int, int] | None] | None = None
        self.patch_request_lesions: list[tuple[int, str, int] | None] | None = None
        self.patch_request_lesion_sizes: list[int | None] | None = None
        self.patch_request_scan_groups: list[str | None] | None = None
        self.positive_indices: list[int] = []
        self.positive_indices_by_category: dict[str, list[int]] = {
            "complete": [],
            "boundary": [],
        }
        self.positive_indices_by_scan_group: dict[str, list[int]] = {
            "small_only": [],
            "large_containing": [],
        }
        self.negative_indices_by_category: dict[str, list[int]] = {
            "adjacent": [],
            "anatomical_coverage": [],
        }
        lesion_sampling = config.get("lesion_sampling", {})
        balanced_sampling = config.get("balanced_patch_sampling", {})
        index_validation = bool(balanced_sampling.get("validation_enabled", False))
        if (
            (self.training or index_validation)
            and self.use_curve_roi
            and lesion_sampling.get("enabled", False)
        ):
            positive_requests: list[tuple[int, tuple[int, int, int] | None]] = []
            positive_types: list[str] = []
            positive_lesions: list[tuple[int, str, int]] = []
            positive_lesion_sizes: list[int] = []
            negative_requests: list[tuple[int, tuple[int, int, int]]] = []
            negative_types: list[str] = []
            negative_slots: list[tuple[int, int, int]] = []
            sliding_config = config.get("roi_sliding_window", {})
            spatial_phases = balanced_sampling.get("spatial_phases", [0.0, 0.5, 1.0])
            adjacent_margin = tuple(
                int(value)
                for value in balanced_sampling.get(
                    "adjacent_margin_pixels", [0, self.patch_size[1] // 2, self.patch_size[2]]
                )
            )
            total_records = len(self.records)
            for record_index, record in enumerate(self.records):
                targets = self._load_targets(record)
                if targets is None:
                    continue
                target = targets[self.target_names.index("rnv")]
                ilm_surface, bm_surface = load_retinal_surfaces(
                    record, self.config, target.shape
                )
                upper_surface, lower_surface, ilm_shift = shifted_curve_roi_surfaces(
                    target,
                    ilm_surface,
                    bm_surface,
                    target.shape[0],
                    self.config,
                )
                bounds = curved_roi_bounds(
                    upper_surface,
                    lower_surface,
                    target.shape[0],
                    minimum_depth=self.patch_size[0],
                )
                guidance_names = lesion_sampling.get("guidance_targets", ["rnv"])
                unknown_guidance = set(guidance_names) - set(self.target_names)
                if unknown_guidance:
                    raise ValueError(
                        f"Unknown lesion_sampling.guidance_targets: {sorted(unknown_guidance)}"
                    )
                case_positive: dict[
                    tuple[int, int, int],
                    tuple[str, tuple[int, str, int], bool, int],
                ] = {}
                positive_views = lesion_sampling.get("positive_views", {})
                for guidance_name in guidance_names:
                    guidance = targets[self.target_names.index(guidance_name)]
                    if self.training and positive_views.get("enabled", True):
                        candidates = rnv_positive_patch_candidates(
                            guidance,
                            target.shape,
                            self.patch_size,
                            bounds,
                            depth_anchor_fraction=self.depth_anchor_fraction,
                            component_overlap=float(
                                lesion_sampling.get("component_overlap", 0.5)
                            ),
                            connectivity=int(lesion_sampling.get("connectivity", 26)),
                            minimum_voxels=int(
                                lesion_sampling.get("minimum_voxels", 1)
                            ),
                            roi_surfaces=(upper_surface, lower_surface),
                            lateral_offset_fraction=float(
                                positive_views.get("lateral_offset_fraction", 0.25)
                            ),
                            bscan_offset_fraction=float(
                                positive_views.get("bscan_offset_fraction", 0.25)
                            ),
                            complete_minimum_coverage=float(
                                positive_views.get("complete_minimum_coverage", 0.8)
                            ),
                            boundary_minimum_coverage=float(
                                positive_views.get("boundary_minimum_coverage", 0.2)
                            ),
                            maximum_views_per_component=int(
                                positive_views.get("maximum_views_per_component", 6)
                            ),
                            maximum_views_per_case=int(
                                positive_views.get("maximum_views_per_case", 20)
                            ),
                        )
                        for candidate in candidates:
                            existing = case_positive.get(candidate.start)
                            if existing is None or (candidate.is_base and not existing[2]):
                                case_positive[candidate.start] = (
                                    candidate.view_type,
                                    (record_index, guidance_name, candidate.component_id),
                                    candidate.is_base,
                                    candidate.component_voxels,
                                )
                    else:
                        anchors = rnv_component_anchors(
                            guidance,
                            self.patch_size,
                            overlap=float(
                                lesion_sampling.get("component_overlap", 0.5)
                            ),
                            connectivity=int(lesion_sampling.get("connectivity", 26)),
                            minimum_voxels=int(
                                lesion_sampling.get("minimum_voxels", 1)
                            ),
                            roi_surfaces=(upper_surface, lower_surface),
                        )
                        starts = covering_patch_starts(
                            anchors,
                            target.shape,
                            self.patch_size,
                            bounds,
                            self.depth_anchor_fraction,
                        )
                        for component_id, start in enumerate(starts, start=1):
                            case_positive.setdefault(
                                start,
                                (
                                    "complete",
                                    (record_index, guidance_name, component_id),
                                    True,
                                    0,
                                ),
                            )
                starts = sorted(case_positive)
                for start in starts:
                    view_type, lesion_group, _, component_voxels = case_positive[start]
                    positive_requests.append((record_index, start))
                    positive_types.append(f"positive_{view_type}")
                    positive_lesions.append(lesion_group)
                    positive_lesion_sizes.append(component_voxels)
                positive_counts = {
                    name: sum(
                        view_type == f"positive_{name}"
                        for view_type in positive_types[-len(starts) :]
                    )
                    if starts
                    else 0
                    for name in ("complete", "boundary")
                }
                category_counts = {
                    "adjacent": 0,
                    "anatomical_coverage": 0,
                }
                if sliding_config.get("enabled", True):
                    candidates = roi_stratified_window_candidates(
                        target.shape,
                        self.patch_size,
                        bounds,
                        spatial_phases=spatial_phases,
                    )
                    rnv_locations = np.argwhere(target > 0)
                    seen: set[tuple[int, int, int]] = set()
                    for start, slot in candidates:
                        if start in seen or _locations_intersect_patch(
                            rnv_locations, start, self.patch_size
                        ):
                            continue
                        seen.add(start)
                        category = _negative_patch_category(
                            start,
                            self.patch_size,
                            rnv_locations,
                            adjacent_margin,
                        )
                        negative_requests.append((record_index, start))
                        negative_types.append(f"negative_{category}")
                        negative_slots.append((record_index, *slot))
                        category_counts[category] += 1
                print(
                    f"Indexing {'/'.join(guidance_names)} patches "
                    f"[{record_index + 1}/{total_records}] "
                    f"{record.case_id}: {len(starts)} positive views "
                    f"(complete={positive_counts['complete']}, "
                    f"boundary={positive_counts['boundary']}) + "
                    f"{sum(category_counts.values())} pure-negative candidates "
                    f"(adjacent={category_counts['adjacent']}, "
                    f"anatomical_coverage={category_counts['anatomical_coverage']}), "
                    f"ILM shift={ilm_shift}px",
                    flush=True,
                )
            self.patch_requests = positive_requests + negative_requests
            self.patch_request_types = (
                positive_types + negative_types
            )
            self.patch_request_slots = [None] * len(positive_requests) + negative_slots
            self.patch_request_lesions = positive_lesions + [None] * len(
                negative_requests
            )
            self.patch_request_lesion_sizes = positive_lesion_sizes + [None] * len(
                negative_requests
            )
            size_groups = config.get("lesion_size_groups", {})
            large_threshold = int(size_groups.get("large_component_voxels", 36_586))
            largest_by_record: dict[int, int] = {}
            for lesion, size in zip(
                positive_lesions, positive_lesion_sizes, strict=True
            ):
                record_index = int(lesion[0])
                largest_by_record[record_index] = max(
                    largest_by_record.get(record_index, 0), int(size)
                )
            positive_scan_groups = [
                (
                    "large_containing"
                    if largest_by_record.get(int(lesion[0]), 0) >= large_threshold
                    else "small_only"
                )
                for lesion in positive_lesions
            ]
            self.patch_request_scan_groups = positive_scan_groups + [None] * len(
                negative_requests
            )
            self.positive_indices = list(range(len(positive_requests)))
            for index, sample_type in enumerate(positive_types):
                self.positive_indices_by_category[
                    sample_type.removeprefix("positive_")
                ].append(index)
                self.positive_indices_by_scan_group[
                    positive_scan_groups[index]
                ].append(index)
            for offset, category in enumerate(negative_types, start=len(positive_requests)):
                self.negative_indices_by_category[category.removeprefix("negative_")].append(
                    offset
                )
            if not self.patch_requests:
                raise ValueError("lesion_sampling found no RNV components")
            negative_total = sum(len(values) for values in self.negative_indices_by_category.values())
            print(
                f"Indexed {len(positive_requests)} multi-view foreground-guided, "
                f"{negative_total} pure-negative candidates; balanced samplers select "
                "positive:negative=1:1 per epoch",
                flush=True,
            )

    def __len__(self) -> int:
        if self.patch_requests is not None:
            return len(self.patch_requests)
        return len(self.records) * self.samples_per_volume

    def _load_targets(self, record: ScanRecord) -> np.ndarray | None:
        return _load_target_stack(record, self.config)

    def __getitem__(self, index: int) -> dict[str, Any]:
        indexed_start = None
        if self.patch_requests is not None:
            record_index, indexed_start = self.patch_requests[index]
            record = self.records[record_index]
        else:
            record = self.records[index // self.samples_per_volume]
        draw = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(draw + index)
        target = self._load_targets(record)
        primary_target = None if target is None else target[self.target_names.index("rnv")]

        variable_names = [self.variables[name] for name in self.modalities if name in {"oct", "octa"}]
        image_data = loadmat(record.image_path, variable_names=variable_names)
        reference = primary_target
        if reference is None:
            reference = np.transpose(np.asarray(image_data[variable_names[0]]), self.axis_order)
        original_shape = tuple(int(value) for value in reference.shape)
        upper_surface = None
        lower_surface = None
        if self.use_curve_roi:
            ilm_surface, bm_surface = load_retinal_surfaces(
                record, self.config, original_shape
            )
            if any(patch > current for patch, current in zip(self.patch_size, original_shape, strict=True)):
                raise ValueError("curve_roi patch dimensions cannot exceed the volume dimensions")
            upper_surface, lower_surface, _ = shifted_curve_roi_surfaces(
                primary_target,
                ilm_surface,
                bm_surface,
                original_shape[0],
                self.config,
            )
            bounds = curved_roi_bounds(
                upper_surface,
                lower_surface,
                original_shape[0],
                minimum_depth=self.patch_size[0],
            )
            if indexed_start is not None:
                slices = tuple(
                    slice(start, start + patch)
                    for start, patch in zip(indexed_start, self.patch_size, strict=True)
                )
            else:
                slices = choose_curve_roi_patch_slices(
                    original_shape,
                    self.patch_size,
                    rng,
                    bounds,
                    foreground=(
                        primary_target
                        if primary_target is not None
                        and (not self.training or self.patch_requests is None)
                        else None
                    ),
                    foreground_probability=(
                        self.foreground_probability
                        if self.training and self.patch_requests is None
                        else 1.0
                        if not self.training and primary_target is not None
                        else 0.0
                    ),
                    random_crop=self.training,
                    depth_anchor_fraction=self.depth_anchor_fraction,
                )
        else:
            padded_reference = pad_to_shape(reference, self.patch_size)
            shape = tuple(int(value) for value in padded_reference.shape)
            slices = choose_patch_slices(
                shape,
                self.patch_size,
                rng,
                foreground=padded_reference if primary_target is not None else None,
                foreground_probability=self.foreground_probability if self.training else 0.0,
                random_crop=self.training,
            )

        channels: list[np.ndarray] = []
        for modality in self.modalities:
            if modality in {"oct", "octa"}:
                full_volume = np.transpose(
                    np.asarray(image_data[self.variables[modality]]), self.axis_order
                )
                cache_key = (record.sample_id, modality)
                if cache_key not in self._normalization_statistics_cache:
                    self._normalization_statistics_cache[cache_key] = (
                        _normalization_statistics(
                            full_volume, self.normalize, self.percentiles
                        )
                    )
                volume = pad_to_shape(full_volume, self.patch_size)[slices]
                channels.append(
                    _normalize(
                        volume,
                        self.normalize,
                        self.percentiles,
                        self._normalization_statistics_cache[cache_key],
                    )
                )
            elif modality in {"artery", "vein", "large_vessel"}:
                if record.vessel_map_path is None:
                    raise FileNotFoundError(f"Vessel map is required for sample {record.sample_id}")
                channels.append(
                    _vessel_volume_patch(
                        record.vessel_map_path,
                        modality,
                        original_shape,
                        slices,
                        int(self.vessel.get("depth_axis", 0)),
                        self.vessel.get("alignment", "center_pad_or_crop"),
                        self.vessel,
                    ).astype(np.float32)
                )

        image = np.stack(channels, axis=0)
        target_patch = None
        if target is not None:
            target_patch = pad_to_shape(target, self.patch_size)[(slice(None), *slices)]
            target_patch = (target_patch > 0).astype(np.float32)
        if self.training:
            augmentation = self.intensity_augmentation
            if augmentation.get("enabled", False):
                contrast = augmentation.get("contrast", {})
                noise = augmentation.get("gaussian_noise", {})
                image = apply_intensity_augmentations(
                    image,
                    rng,
                    self.intensity_channels,
                    contrast_probability=float(contrast.get("probability", 0.5)),
                    contrast_range=contrast.get("range", [0.8, 1.2]),
                    noise_probability=float(noise.get("probability", 0.3)),
                    noise_std_range=noise.get("std_range", [0.0, 0.05]),
                )
            image, target_patch = apply_random_flips(
                image,
                target_patch,
                rng,
                probability=self.random_flip_probability,
                axes=self.random_flip_axes,
            )
        sample: dict[str, Any] = {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "sample_id": record.sample_id,
        }
        if target_patch is not None:
            sample["target"] = torch.from_numpy(target_patch)
        if self.patch_request_types is not None:
            sample["sampling_type"] = self.patch_request_types[index]
        return sample


def build_dataset(
    records: Sequence[ScanRecord], config: dict[str, Any], training: bool
) -> RNVMatDataset:
    name = config.get("dataset_name", "rnv_mat")
    return DATASETS.build(name, records=records, config=config, training=training)


def load_full_case(record: ScanRecord, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load one complete case for sliding-window inference.

    Full 768x600x600 volumes are memory intensive; callers should process one case
    at a time and keep sliding-window outputs on CPU.
    """
    modalities = list(config["modalities"])
    variables = config["mat_variables"]
    axis_order = tuple(config.get("axis_order", [0, 1, 2]))
    names = [variables[name] for name in modalities if name in {"oct", "octa"}]
    payload = loadmat(record.image_path, variable_names=names)
    volumes: list[np.ndarray] = []
    shape: tuple[int, int, int] | None = None
    for modality in modalities:
        if modality in {"oct", "octa"}:
            array = np.transpose(np.asarray(payload[variables[modality]]), axis_order)
            shape = tuple(int(value) for value in array.shape)
            volumes.append(
                _normalize(array, config.get("normalize", "percentile"), config.get("percentile_range", [0.5, 99.5]))
            )
        else:
            if shape is None:
                raise ValueError("OCT or OCTA must precede 2D vessel priors in data.modalities")
            if record.vessel_map_path is None:
                raise FileNotFoundError(f"Vessel map is required for sample {record.sample_id}")
            with Image.open(record.vessel_map_path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
            vessel = config.get("vessel_prior", {})
            depth_axis = int(vessel.get("depth_axis", 0))
            lateral_axes = [axis for axis in range(3) if axis != depth_axis]
            plane = _align_vessel_plane(
                _vessel_plane_for_modality(rgb, modality, vessel),
                (shape[lateral_axes[0]], shape[lateral_axes[1]]),
                vessel.get("alignment", "center_pad_or_crop"),
            )
            volumes.append(np.repeat(np.expand_dims(plane, depth_axis), shape[depth_axis], axis=depth_axis))
    target = _load_target_stack(record, config)
    if target is not None:
        target = torch.from_numpy((target > 0).astype(np.float32, copy=False))
    return torch.from_numpy(np.stack(volumes).astype(np.float32, copy=False)), target
