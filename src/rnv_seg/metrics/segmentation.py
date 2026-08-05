from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage


def _binary(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=bool)


def voxel_metrics(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1.0e-8) -> dict[str, float]:
    prediction = _binary(prediction)
    target = _binary(target)
    tp = float(np.logical_and(prediction, target).sum())
    fp = float(np.logical_and(prediction, ~target).sum())
    fn = float(np.logical_and(~prediction, target).sum())
    tn = float(np.logical_and(~prediction, ~target).sum())
    return {
        "dice": (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon),
        "iou": (tp + epsilon) / (tp + fp + fn + epsilon),
        "precision": (tp + epsilon) / (tp + fp + epsilon),
        "recall": (tp + epsilon) / (tp + fn + epsilon),
        "specificity": (tn + epsilon) / (tn + fp + epsilon),
    }


def volume_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    epsilon: float = 1.0e-8,
) -> dict[str, float]:
    voxel_volume = float(np.prod(voxel_spacing_mm))
    predicted = float(_binary(prediction).sum()) * voxel_volume
    actual = float(_binary(target).sum()) * voxel_volume
    absolute_error = abs(predicted - actual)
    relative_error = absolute_error / max(actual, epsilon)
    return {
        "predicted_volume_mm3": predicted,
        "target_volume_mm3": actual,
        "absolute_volume_error_mm3": absolute_error,
        "relative_volume_error": relative_error,
        "scan_presence_correct": float((predicted > 0) == (actual > 0)),
    }


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: Sequence[float],
    tolerance_mm: float,
) -> dict[str, float]:
    prediction = _binary(prediction)
    target = _binary(target)
    if not prediction.any() and not target.any():
        return {"surface_dice": 1.0, "hausdorff95_mm": 0.0, "assd_mm": 0.0}
    if not prediction.any() or not target.any():
        return {"surface_dice": 0.0, "hausdorff95_mm": float("inf"), "assd_mm": float("inf")}
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = np.logical_xor(prediction, ndimage.binary_erosion(prediction, structure))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target, structure))
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing_mm)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_mm)
    pred_distances = distance_to_target[pred_surface]
    target_distances = distance_to_prediction[target_surface]
    all_distances = np.concatenate([pred_distances, target_distances])
    within = float((pred_distances <= tolerance_mm).sum() + (target_distances <= tolerance_mm).sum())
    denominator = float(pred_distances.size + target_distances.size)
    return {
        "surface_dice": within / max(denominator, 1.0),
        "hausdorff95_mm": float(np.percentile(all_distances, 95)),
        "assd_mm": float(all_distances.mean()),
    }

