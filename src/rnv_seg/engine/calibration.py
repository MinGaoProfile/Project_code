from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product
from typing import Any

import numpy as np

from .evaluator import evaluate_binary_case
from .inference import postprocess_rnv_mask


def _validated_grid(
    values: Sequence[float | int],
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> tuple[float | int, ...]:
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise ValueError(f"{name} must contain at least one value")
    if any(float(value) < minimum for value in unique):
        raise ValueError(f"{name} values must be >= {minimum}")
    if maximum is not None and any(float(value) > maximum for value in unique):
        raise ValueError(f"{name} values must be <= {maximum}")
    return unique


def select_calibration_result(
    rows: Sequence[dict[str, Any]],
    minimum_lesion_sensitivity: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select the best sensitivity-qualified row and the best unconstrained row."""

    minimum_lesion_sensitivity = float(minimum_lesion_sensitivity)

    def rank(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row["selection_score"]),
            -float(row["false_positive_lesions"]),
            float(row["lesion_sensitivity"]),
            float(row["positive_dice"]),
            float(row["lesion_f1"]),
        )

    finite = [
        row
        for row in rows
        if np.isfinite(float(row.get("selection_score", float("nan"))))
        and np.isfinite(float(row.get("lesion_sensitivity", float("nan"))))
    ]
    if not finite:
        return None, None
    unconstrained = max(finite, key=rank)
    eligible = [
        row
        for row in finite
        if float(row["lesion_sensitivity"]) >= minimum_lesion_sensitivity
    ]
    return (max(eligible, key=rank) if eligible else None), unconstrained


def calibrate_postprocessing_grid(
    cases: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    thresholds: Sequence[float],
    minimum_areas: Sequence[int],
    consecutive_bscans: Sequence[int],
    metric_config: dict[str, Any],
    minimum_lesion_sensitivity: float = 0.82,
    positive_dice_weight: float = 0.5,
    lesion_f1_weight: float = 0.5,
    connectivity_2d: int = 8,
    connectivity_3d: int = 26,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Evaluate a postprocessing grid from one inference pass per validation case."""

    thresholds = tuple(
        float(value)
        for value in _validated_grid(
            thresholds, name="thresholds", minimum=0.0, maximum=1.0
        )
    )
    minimum_areas = tuple(
        int(value)
        for value in _validated_grid(
            minimum_areas, name="minimum_areas", minimum=1.0
        )
    )
    consecutive_bscans = tuple(
        int(value)
        for value in _validated_grid(
            consecutive_bscans, name="consecutive_bscans", minimum=1.0
        )
    )
    positive_dice_weight = float(positive_dice_weight)
    lesion_f1_weight = float(lesion_f1_weight)
    weight_total = positive_dice_weight + lesion_f1_weight
    if positive_dice_weight < 0.0 or lesion_f1_weight < 0.0 or weight_total <= 0.0:
        raise ValueError("Calibration metric weights must be non-negative and sum positive")
    positive_dice_weight /= weight_total
    lesion_f1_weight /= weight_total

    keys = list(product(thresholds, minimum_areas, consecutive_bscans))
    accumulators = {
        key: {
            "positive_dice": [],
            "detected_lesions": 0.0,
            "target_lesions": 0.0,
            "false_positive_lesions": 0.0,
            "negative_scans": 0,
            "clean_negative_scans": 0,
        }
        for key in keys
    }
    case_count = 0
    for probability, target in cases:
        probability = np.asarray(probability, dtype=np.float32)
        target = np.asarray(target, dtype=bool)
        if probability.shape != target.shape or probability.ndim != 3:
            raise ValueError("Calibration probability and target must share a 3D shape")
        case_count += 1
        target_positive = bool(target.any())
        for key in keys:
            threshold, minimum_area, minimum_frames = key
            prediction = postprocess_rnv_mask(
                probability >= threshold,
                minimum_2d_pixels=minimum_area,
                minimum_consecutive_bscans=minimum_frames,
                connectivity_2d=connectivity_2d,
                connectivity_3d=connectivity_3d,
            )
            metrics = evaluate_binary_case(prediction, target, metric_config)
            accumulator = accumulators[key]
            if target_positive:
                accumulator["positive_dice"].append(float(metrics["dice"]))
            else:
                accumulator["negative_scans"] += 1
                accumulator["clean_negative_scans"] += int(not prediction.any())
            for name in (
                "detected_lesions",
                "target_lesions",
                "false_positive_lesions",
            ):
                accumulator[name] += float(metrics[name])
    if case_count == 0:
        raise ValueError("Calibration requires at least one validation case")

    rows: list[dict[str, Any]] = []
    for threshold, minimum_area, minimum_frames in keys:
        accumulator = accumulators[(threshold, minimum_area, minimum_frames)]
        detected = float(accumulator["detected_lesions"])
        target_lesions = float(accumulator["target_lesions"])
        false_positives = float(accumulator["false_positive_lesions"])
        sensitivity = detected / target_lesions if target_lesions else float("nan")
        precision_denominator = detected + false_positives
        precision = detected / precision_denominator if precision_denominator else 1.0
        lesion_f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if np.isfinite(sensitivity) and precision + sensitivity
            else float("nan")
        )
        dice_values = accumulator["positive_dice"]
        positive_dice = float(np.mean(dice_values)) if dice_values else float("nan")
        selection_score = (
            positive_dice_weight * positive_dice + lesion_f1_weight * lesion_f1
            if np.isfinite(positive_dice) and np.isfinite(lesion_f1)
            else float("nan")
        )
        negative_scans = int(accumulator["negative_scans"])
        rows.append(
            {
                "threshold": threshold,
                "minimum_2d_component_pixels": minimum_area,
                "minimum_consecutive_bscans": minimum_frames,
                "positive_dice": positive_dice,
                "lesion_f1": lesion_f1,
                "lesion_sensitivity": sensitivity,
                "lesion_precision": precision,
                "false_positive_lesions": int(false_positives),
                "detected_lesions": int(detected),
                "target_lesions": int(target_lesions),
                "negative_scan_clean_rate": (
                    float(accumulator["clean_negative_scans"]) / negative_scans
                    if negative_scans
                    else float("nan")
                ),
                "selection_score": selection_score,
                "eligible": bool(
                    np.isfinite(sensitivity)
                    and sensitivity >= float(minimum_lesion_sensitivity)
                ),
            }
        )
    selected, unconstrained = select_calibration_result(
        rows, minimum_lesion_sensitivity
    )
    return rows, selected, unconstrained
