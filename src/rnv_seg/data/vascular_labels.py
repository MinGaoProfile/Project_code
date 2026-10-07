from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage


def _triplet(values: Sequence[float], name: str) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain three non-negative values")
    return result


def _robust_unit(array: np.ndarray, mask: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    values = array[mask]
    if not values.size:
        return np.zeros_like(array, dtype=np.float32)
    low, high = (float(value) for value in np.percentile(values, percentiles))
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def _bounding_slices(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coordinates = np.nonzero(mask)
    starts = [max(0, int(axis.min()) - margin) for axis in coordinates]
    stops = [min(mask.shape[index], int(axis.max()) + margin + 1) for index, axis in enumerate(coordinates)]
    return tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    if not values.size or float(values.max()) <= float(values.min()):
        return float(values.max()) if values.size else 1.0
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64) / max(int(histogram.sum()), 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * centers)
    total_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    variance = np.zeros_like(denominator)
    valid = denominator > 0
    variance[valid] = (
        total_mean * cumulative_probability[valid] - cumulative_mean[valid]
    ) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(variance))])


def generate_vasculature_mask(
    octa: np.ndarray,
    structural_mask: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate an OCTA-guided vascular pseudo-label inside a structural RNV mask."""
    options = config or {}
    signal = np.asarray(octa)
    structural = np.asarray(structural_mask) > 0
    if signal.shape != structural.shape or signal.ndim != 3:
        raise ValueError("OCTA and structural mask must have matching 3D shapes")
    output = np.zeros(structural.shape, dtype=np.uint8)
    if not structural.any():
        return output, {"structural_voxels": 0, "vasculature_voxels": 0, "components": 0}

    margin = int(options.get("crop_margin_pixels", 8))
    crop_slices = _bounding_slices(structural, margin)
    local_mask = structural[crop_slices]
    local_signal = signal[crop_slices].astype(np.float32, copy=False)
    denoise_sigma = _triplet(options.get("denoise_sigma", [0.8, 0.6, 0.6]), "denoise_sigma")
    background_sigma = _triplet(
        options.get("background_sigma", [4.0, 2.0, 2.0]), "background_sigma"
    )
    smoothed = ndimage.gaussian_filter(local_signal, sigma=denoise_sigma)
    background = ndimage.gaussian_filter(smoothed, sigma=background_sigma)
    contrast = np.maximum(smoothed - background, 0.0)
    normalization = options.get("normalization_percentiles", [5.0, 99.0])
    intensity_score = _robust_unit(smoothed, local_mask, normalization)
    contrast_score = _robust_unit(contrast, local_mask, normalization)
    intensity_weight = float(options.get("intensity_weight", 0.7))
    if not 0.0 <= intensity_weight <= 1.0:
        raise ValueError("intensity_weight must be between 0 and 1")
    score = intensity_weight * intensity_score + (1.0 - intensity_weight) * contrast_score

    structure = ndimage.generate_binary_structure(3, 3)
    structural_labels, structural_count = ndimage.label(local_mask, structure=structure)
    threshold_method = str(options.get("threshold_method", "otsu_hysteresis"))
    low_quantile = float(options.get("low_quantile", 0.60))
    high_quantile = float(options.get("high_quantile", 0.85))
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("low_quantile and high_quantile must satisfy 0 <= low < high <= 1")
    minimum_structural = int(options.get("minimum_structural_component_voxels", 10))
    minimum_vascular = int(options.get("minimum_vascular_component_voxels", 5))
    local_output = np.zeros_like(local_mask)
    component_rows: list[dict[str, Any]] = []

    for component in range(1, structural_count + 1):
        component_mask = structural_labels == component
        component_size = int(component_mask.sum())
        if component_size < minimum_structural:
            continue
        values = score[component_mask]
        otsu = _otsu_threshold(values)
        if threshold_method == "otsu_hysteresis":
            low = max(
                float(options.get("minimum_low_score", 0.12)),
                otsu * float(options.get("low_otsu_scale", 1.0)),
            )
        elif threshold_method == "quantile_hysteresis":
            low = float(np.quantile(values, low_quantile))
        else:
            raise ValueError(
                "threshold_method must be otsu_hysteresis or quantile_hysteresis"
            )
        high = max(
            low,
            float(options.get("minimum_high_score", 0.25)),
            float(np.quantile(values, high_quantile)),
        )
        candidates = component_mask & (score >= low)
        seeds = component_mask & (score >= high)
        candidate_labels, candidate_count = ndimage.label(candidates, structure=structure)
        if candidate_count:
            seed_labels = np.unique(candidate_labels[seeds])
            seed_labels = seed_labels[seed_labels > 0]
            sizes = np.bincount(candidate_labels.ravel(), minlength=candidate_count + 1)
            keep = np.zeros(candidate_count + 1, dtype=bool)
            keep[seed_labels] = sizes[seed_labels] >= minimum_vascular
            local_output |= keep[candidate_labels]
        component_rows.append(
            {
                "structural_voxels": component_size,
                "otsu_threshold": otsu,
                "low_threshold": low,
                "high_threshold": high,
                "vascular_voxels": int((local_output & component_mask).sum()),
            }
        )

    local_output &= local_mask
    output[crop_slices] = local_output.astype(np.uint8)
    vascular_voxels = int(output.sum())
    structural_voxels = int(structural.sum())
    return output, {
        "structural_voxels": structural_voxels,
        "vasculature_voxels": vascular_voxels,
        "vascular_fraction": vascular_voxels / structural_voxels,
        "components": len(component_rows),
        "component_statistics": component_rows,
        "crop_shape": list(local_mask.shape),
    }
