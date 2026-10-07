from __future__ import annotations

import numpy as np
from scipy import ndimage


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    if connectivity == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError("Connectivity must be 6, 18, or 26")


def lesion_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    connectivity: int = 26,
    minimum_voxels: int = 3,
    match_iou: float = 0.1,
) -> dict[str, float]:
    structure = _structure(connectivity)
    predicted_labels, predicted_count = ndimage.label(
        np.asarray(prediction, dtype=bool), structure=structure
    )
    target_labels, target_count = ndimage.label(np.asarray(target, dtype=bool), structure=structure)
    predicted_sizes = np.bincount(predicted_labels.ravel(), minlength=predicted_count + 1)
    target_sizes = np.bincount(target_labels.ravel(), minlength=target_count + 1)
    valid_predicted = set(
        int(index) for index in np.flatnonzero(predicted_sizes[1:] >= minimum_voxels) + 1
    )
    valid_target = set(int(index) for index in np.flatnonzero(target_sizes[1:] >= minimum_voxels) + 1)

    candidates: list[tuple[float, int, int]] = []
    overlap = (predicted_labels > 0) & (target_labels > 0)
    if overlap.any():
        multiplier = int(target_count) + 1
        pairs = (
            predicted_labels[overlap].astype(np.int64) * multiplier
            + target_labels[overlap].astype(np.int64)
        )
        pair_codes, intersections = np.unique(pairs, return_counts=True)
        for code, intersection_count in zip(pair_codes, intersections, strict=True):
            pred_index = int(code // multiplier)
            target_index = int(code % multiplier)
            if pred_index not in valid_predicted or target_index not in valid_target:
                continue
            intersection = float(intersection_count)
            union = float(predicted_sizes[pred_index] + target_sizes[target_index] - intersection_count)
            iou = intersection / union if union else 0.0
            if iou >= match_iou:
                candidates.append((iou, pred_index, target_index))
    matched_pred: set[int] = set()
    matched_target: set[int] = set()
    for _, pred_index, target_index in sorted(candidates, reverse=True):
        if pred_index not in matched_pred and target_index not in matched_target:
            matched_pred.add(pred_index)
            matched_target.add(target_index)
    tp = len(matched_target)
    fp = len(valid_predicted) - tp
    fn = len(valid_target) - tp
    sensitivity = tp / len(valid_target) if valid_target else float(not valid_predicted)
    precision = tp / len(valid_predicted) if valid_predicted else float(not valid_target)
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "lesion_sensitivity": float(sensitivity),
        "lesion_precision": float(precision),
        "lesion_f1": float(f1),
        "false_positive_lesions": float(fp),
        "detected_lesions": float(tp),
        "target_lesions": float(len(valid_target)),
        "missed_lesions": float(fn),
    }
