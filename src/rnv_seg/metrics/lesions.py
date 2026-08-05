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


def _components(mask: np.ndarray, connectivity: int, minimum_voxels: int) -> list[np.ndarray]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=_structure(connectivity))
    return [labels == index for index in range(1, count + 1) if int((labels == index).sum()) >= minimum_voxels]


def lesion_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    connectivity: int = 26,
    minimum_voxels: int = 3,
    match_iou: float = 0.1,
) -> dict[str, float]:
    predicted = _components(prediction, connectivity, minimum_voxels)
    actual = _components(target, connectivity, minimum_voxels)
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for target_index, truth in enumerate(actual):
            intersection = float(np.logical_and(pred, truth).sum())
            union = float(np.logical_or(pred, truth).sum())
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
    fp = len(predicted) - tp
    fn = len(actual) - tp
    sensitivity = tp / len(actual) if actual else float(not predicted)
    precision = tp / len(predicted) if predicted else float(not actual)
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "lesion_sensitivity": float(sensitivity),
        "lesion_precision": float(precision),
        "lesion_f1": float(f1),
        "false_positive_lesions": float(fp),
        "detected_lesions": float(tp),
        "target_lesions": float(len(actual)),
        "missed_lesions": float(fn),
    }

