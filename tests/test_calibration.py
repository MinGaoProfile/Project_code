import numpy as np

from rnv_seg.engine.calibration import (
    calibrate_postprocessing_grid,
    select_calibration_result,
)


def _metric_config() -> dict:
    return {
        "lesion": {
            "enabled": True,
            "connectivity": 26,
            "minimum_voxels": 3,
            "match_iou": 0.1,
        },
        "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
        "surface": {"enabled": False},
    }


def test_calibration_scans_24_combinations_and_enforces_sensitivity_floor() -> None:
    target = np.zeros((24, 24, 10), dtype=np.uint8)
    target[1:6, 1:7, 0:7] = 1
    target[10:15, 10:16, 0:5] = 1
    probability = np.full(target.shape, 0.01, dtype=np.float32)
    probability[target > 0] = 0.70
    probability[10:15, 10:16, 0:5] = 0.58
    probability[1:6, 12:16, 0:7] = 0.70  # Persistent 20-pixel false positive.

    rows, selected, unconstrained = calibrate_postprocessing_grid(
        [(probability, target)],
        thresholds=[0.50, 0.55, 0.60, 0.65],
        minimum_areas=[10, 20, 30],
        consecutive_bscans=[5, 7],
        metric_config=_metric_config(),
        minimum_lesion_sensitivity=0.82,
    )

    assert len(rows) == 24
    assert selected is not None
    assert unconstrained is not None
    assert selected["lesion_sensitivity"] >= 0.82
    assert selected["threshold"] in {0.50, 0.55}
    assert selected["minimum_2d_component_pixels"] == 30
    assert selected["minimum_consecutive_bscans"] == 5


def test_calibration_does_not_select_an_ineligible_combination() -> None:
    rows = [
        {
            "selection_score": 0.95,
            "false_positive_lesions": 0,
            "lesion_sensitivity": 0.7,
            "positive_dice": 0.99,
            "lesion_f1": 0.91,
        },
        {
            "selection_score": 0.8,
            "false_positive_lesions": 1,
            "lesion_sensitivity": 0.85,
            "positive_dice": 0.8,
            "lesion_f1": 0.8,
        },
    ]

    selected, unconstrained = select_calibration_result(rows, 0.82)

    assert selected is rows[1]
    assert unconstrained is rows[0]
