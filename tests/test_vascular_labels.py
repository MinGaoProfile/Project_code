import numpy as np

from rnv_seg.data.vascular_labels import generate_vasculature_mask


def test_vascular_pseudolabel_is_bright_connected_and_inside_structural_mask() -> None:
    structural = np.zeros((16, 16, 8), dtype=np.uint8)
    structural[3:13, 3:13, 1:7] = 1
    octa = np.full(structural.shape, 10, dtype=np.uint8)
    octa[8, 5:11, 2:6] = 220
    octa[0, :, :] = 255  # Bright signal outside structural RNV must never enter the label.
    mask, statistics = generate_vasculature_mask(
        octa,
        structural,
        {
            "denoise_sigma": [0.0, 0.0, 0.0],
            "background_sigma": [1.0, 1.0, 1.0],
            "threshold_method": "quantile_hysteresis",
            "low_quantile": 0.8,
            "high_quantile": 0.9,
            "minimum_vascular_component_voxels": 3,
        },
    )
    assert mask.shape == structural.shape
    assert mask.dtype == np.uint8
    assert np.all(mask <= structural)
    assert mask[8, 5:11, 2:6].sum() > 0
    assert mask[0].sum() == 0
    assert statistics["vasculature_voxels"] == int(mask.sum())


def test_empty_structural_mask_produces_empty_vascular_label() -> None:
    octa = np.ones((4, 4, 4), dtype=np.uint8)
    mask, statistics = generate_vasculature_mask(octa, np.zeros_like(octa))
    assert not mask.any()
    assert statistics["structural_voxels"] == 0
