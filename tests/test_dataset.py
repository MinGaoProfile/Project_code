from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from rnv_seg.data.dataset import RNVMatDataset
from rnv_seg.data.discovery import ScanRecord


def test_multimodal_dataset_shape(tmp_path: Path) -> None:
    volume = np.arange(16 * 20 * 20, dtype=np.uint8).reshape(16, 20, 20)
    target = np.zeros_like(volume)
    target[5:9, 8:12, 8:12] = 1
    image_path = tmp_path / "synthetic_oct_octa.mat"
    target_path = tmp_path / "synthetic_mask.mat"
    vessel_path = tmp_path / "synthetic_vessel.png"
    savemat(image_path, {"imgMat": volume, "imgMatAg": volume[::-1]})
    savemat(target_path, {"mask": target})
    vessel = np.zeros((18, 18, 3), dtype=np.uint8)
    vessel[..., 0] = 255
    Image.fromarray(vessel).save(vessel_path)
    record = ScanRecord("safe", "SYNTHETIC", "SYNTHETIC_OD", tmp_path, image_path, target_path, None, vessel_path)
    config = {
        "modalities": ["oct", "octa", "artery", "vein"],
        "mat_variables": {"oct": "imgMat", "octa": "imgMatAg", "target": "mask"},
        "axis_order": [0, 1, 2],
        "patch_size": [8, 16, 16],
        "samples_per_volume": 1,
        "foreground_patch_probability": 1.0,
        "normalize": "minmax",
        "percentile_range": [0.5, 99.5],
        "vessel_prior": {"artery_channel": 0, "vein_channel": 1, "depth_axis": 0},
    }
    sample = RNVMatDataset([record], config, training=True)[0]
    assert sample["image"].shape == (4, 8, 16, 16)
    assert sample["target"].shape == (1, 8, 16, 16)
    assert sample["target"].sum() > 0


def test_exact_vessel_alignment_rejects_mismatched_shape(tmp_path: Path) -> None:
    volume = np.zeros((8, 16, 16), dtype=np.uint8)
    image_path = tmp_path / "synthetic_oct_octa.mat"
    target_path = tmp_path / "synthetic_mask.mat"
    vessel_path = tmp_path / "synthetic_vessel.png"
    savemat(image_path, {"imgMat": volume})
    savemat(target_path, {"mask": volume})
    Image.fromarray(np.zeros((15, 16, 3), dtype=np.uint8)).save(vessel_path)
    record = ScanRecord(
        "safe", "SYNTHETIC", "SYNTHETIC_OD", tmp_path, image_path, target_path, None, vessel_path
    )
    config = {
        "modalities": ["oct", "artery"],
        "mat_variables": {"oct": "imgMat", "target": "mask"},
        "patch_size": [8, 16, 16],
        "normalize": "minmax",
        "percentile_range": [0.5, 99.5],
        "vessel_prior": {
            "artery_channel": 0,
            "depth_axis": 0,
            "alignment": "exact",
        },
    }
    with pytest.raises(ValueError, match="does not exactly match"):
        RNVMatDataset([record], config, training=False)[0]
