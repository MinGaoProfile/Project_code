from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat

from rnv_seg.data.dataset import (
    RNVMatDataset,
    _combine_vessel_channels,
    _negative_patch_category,
    _normalization_statistics,
    _normalize,
    _remove_small_vessel_components,
    _vessel_plane_for_modality,
)
from rnv_seg.data.discovery import ScanRecord
from rnv_seg.data.preview import _preview_indices, save_sample_preview
from rnv_seg.data.roi import (
    covering_patch_starts,
    curved_roi_patch_mask,
    rnv_component_anchors,
    rnv_positive_patch_candidates,
    roi_sliding_window_starts,
    roi_stratified_window_candidates,
    shifted_curve_roi_surfaces,
)
from rnv_seg.transforms import apply_intensity_augmentations


def test_negative_patch_categories_do_not_use_retinal_thickness_as_macula_proxy() -> None:
    patch_size = (8, 4, 4)
    rnv_locations = np.asarray([[2, 5, 5]], dtype=np.int64)

    assert _negative_patch_category(
        (0, 0, 0),
        patch_size,
        rnv_locations,
        (0, 2, 2),
    ) == "adjacent"
    assert _negative_patch_category(
        (0, 0, 0),
        patch_size,
        np.empty((0, 3), dtype=np.int64),
        (0, 2, 2),
    ) == "anatomical_coverage"
    assert _negative_patch_category(
        (0, 4, 0),
        patch_size,
        np.empty((0, 3), dtype=np.int64),
        (0, 2, 2),
    ) == "anatomical_coverage"


def test_patch_percentile_normalization_reuses_full_case_statistics() -> None:
    volume = np.arange(10 * 8 * 6, dtype=np.float32).reshape(10, 8, 6)
    patch_slices = (slice(2, 7), slice(1, 6), slice(2, 5))
    statistics = _normalization_statistics(volume, "percentile", [10.0, 90.0])

    normalized_full = _normalize(volume, "percentile", [10.0, 90.0])
    normalized_patch = _normalize(
        volume[patch_slices],
        "percentile",
        [10.0, 90.0],
        statistics,
    )
    patch_local_normalization = _normalize(
        volume[patch_slices], "percentile", [10.0, 90.0]
    )

    np.testing.assert_allclose(normalized_patch, normalized_full[patch_slices])
    assert not np.allclose(normalized_patch, patch_local_normalization)


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
        "modalities": ["oct", "octa", "large_vessel"],
        "mat_variables": {"oct": "imgMat", "octa": "imgMatAg", "target": "mask"},
        "axis_order": [0, 1, 2],
        "patch_size": [8, 16, 16],
        "samples_per_volume": 1,
        "foreground_patch_probability": 1.0,
        "normalize": "percentile",
        "percentile_range": [0.5, 99.5],
        "vessel_prior": {
            "artery_channels": [0, 2],
            "vein_channels": [1, 2],
            "depth_axis": 0,
        },
    }
    sample = RNVMatDataset([record], config, training=True)[0]
    assert sample["image"].shape == (3, 8, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["target"].shape == (1, 8, 16, 16)
    assert sample["target"].sum() > 0


def test_multitask_dataset_reads_aligned_vasculature_target(tmp_path: Path) -> None:
    volume = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    rnv = np.zeros_like(volume, dtype=np.uint8)
    vasculature = np.zeros_like(volume, dtype=np.uint8)
    rnv[2:6, 2:6, 2:6] = 1
    vasculature[3:5, 3:5, 2:6] = 1
    image_path = tmp_path / "dual_oct_octa.mat"
    target_path = tmp_path / "dual_mask.mat"
    savemat(image_path, {"imgMat": volume})
    savemat(target_path, {"mask": rnv, "vasculature_mask": vasculature})
    record = ScanRecord(
        "dual", "SYNTHETIC", "SYNTHETIC_OD", tmp_path, image_path, target_path, None, None
    )
    dataset = RNVMatDataset(
        [record],
        {
            "modalities": ["oct"],
            "target_names": ["rnv", "vasculature"],
            "mat_variables": {
                "oct": "imgMat",
                "target": "mask",
                "vasculature_target": "vasculature_mask",
            },
            "axis_order": [0, 1, 2],
            "patch_size": [8, 8, 8],
            "normalize": "minmax",
            "random_flip_probability": 0.0,
            "require_vasculature_within_rnv": True,
        },
        training=True,
    )
    sample = dataset[0]
    assert sample["target"].shape == (2, 8, 8, 8)
    assert sample["target"][0].sum() == 64
    assert sample["target"][1].sum() == 16
    output = save_sample_preview(
        sample,
        tmp_path / "preview",
        0,
        modalities=["oct"],
        target_names=["rnv", "vasculature"],
    )
    assert (output / "label_rnv_bscans.png").is_file()
    assert (output / "label_vasculature_bscans.png").is_file()
    assert (output / "overlay_vasculature_bscans.png").is_file()


def test_vessel_channels_combine_blue_into_artery_and_vein() -> None:
    rgb = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    artery = _combine_vessel_channels(rgb, [0, 2])
    vein = _combine_vessel_channels(rgb, [1, 2])
    np.testing.assert_array_equal(artery, [[1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(vein, [[0.0, 1.0, 1.0]])
    assert set(np.unique(artery)) <= {0.0, 1.0}
    assert set(np.unique(vein)) <= {0.0, 1.0}


def test_large_vessel_is_union_of_cleaned_artery_and_vein() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgb[1:3, 1:3, 0] = 1
    rgb[4:6, 4:6, 1] = 1
    rgb[6:8, 1:3, 2] = 1
    config = {
        "artery_channels": [0, 2],
        "vein_channels": [1, 2],
        "minimum_component_pixels": 2,
        "component_connectivity": 8,
    }
    large_vessel = _vessel_plane_for_modality(rgb, "large_vessel", config)
    assert large_vessel.sum() == 12
    assert set(np.unique(large_vessel)) == {0.0, 1.0}


def test_small_vessel_components_are_removed() -> None:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[1, 1] = 1
    mask[4:6, 4:7] = 1
    cleaned = _remove_small_vessel_components(mask, minimum_pixels=4)
    assert cleaned[1, 1] == 0
    assert cleaned[4:6, 4:7].sum() == 6
    assert set(np.unique(cleaned)) == {0.0, 1.0}


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


def test_curve_roi_centres_training_patch_on_rnv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    volume = np.broadcast_to(np.arange(16, dtype=np.uint8)[:, None, None], (16, 8, 8)).copy()
    target = np.zeros_like(volume)
    target[4, 3, 3] = 1
    curves = np.zeros((8, 8, 8), dtype=np.uint16)
    curves[:, 0, :] = 7  # MATLAB coordinate: zero-based ILM depth is 6.
    curves[:, 7, :] = 10  # MATLAB coordinate: zero-based BM depth is 9.
    image_path = tmp_path / "synthetic_oct_octa.mat"
    target_path = tmp_path / "synthetic_mask.mat"
    segmentation_path = tmp_path / "synthetic_seg.mat"
    savemat(image_path, {"imgMat": volume})
    savemat(target_path, {"mask": target})
    savemat(segmentation_path, {"ManualCurveData": curves})
    record = ScanRecord(
        "safe",
        "SYNTHETIC",
        "SYNTHETIC_OD",
        tmp_path,
        image_path,
        target_path,
        segmentation_path,
        None,
    )
    config = {
        "modalities": ["oct"],
        "mat_variables": {
            "oct": "imgMat",
            "target": "mask",
            "segmentation": "ManualCurveData",
        },
        "axis_order": [0, 1, 2],
        "patch_size": [8, 4, 4],
        "foreground_patch_probability": 1.0,
        "normalize": "percentile",
        "percentile_range": [0.5, 99.5],
        "random_flip_axes": [1, 2],
        "random_flip_probability": 0.0,
        "curve_roi": {
            "enabled": True,
            "bm_layer": 8,
            "coordinate_base": 1,
            "rnv_top_margin_pixels": 2,
            "bm_bottom_margin_pixels": 3,
        },
        "lesion_sampling": {"enabled": True},
        "roi_sliding_window": {"enabled": True, "overlap": 0.0},
    }
    dataset = RNVMatDataset([record], config, training=True)
    indexing_output = capsys.readouterr().out
    sample = dataset[0]
    assert "[1/1] synthetic:" in indexing_output
    assert "[1/1] safe:" not in indexing_output
    assert sample["image"].shape == (1, 8, 4, 4)
    assert sample["target"].sum() == 1


def test_curve_roi_coordinates_crop_unmasked_original_patch(tmp_path: Path) -> None:
    volume = np.full((8, 4, 4), 0.25, dtype=np.float32)
    volume[0] = 0.75
    volume[7] = 0.9
    target = np.zeros_like(volume, dtype=np.uint8)
    target[3, 2, 2] = 1
    curves = np.zeros((4, 8, 4), dtype=np.uint16)
    curves[:, 0, :] = 4  # zero-based ILM depth 3
    curves[:, 7, :] = 5  # zero-based BM depth 4
    image_path = tmp_path / "unmasked_oct_octa.mat"
    target_path = tmp_path / "unmasked_mask.mat"
    segmentation_path = tmp_path / "unmasked_seg.mat"
    savemat(image_path, {"imgMat": volume})
    savemat(target_path, {"mask": target})
    savemat(segmentation_path, {"ManualCurveData": curves})
    record = ScanRecord(
        "unmasked",
        "SYNTHETIC",
        "SYNTHETIC_OD",
        tmp_path,
        image_path,
        target_path,
        segmentation_path,
        None,
    )
    config = {
        "modalities": ["oct"],
        "mat_variables": {
            "oct": "imgMat",
            "target": "mask",
            "segmentation": "ManualCurveData",
        },
        "axis_order": [0, 1, 2],
        "patch_size": [8, 4, 4],
        "foreground_patch_probability": 1.0,
        "normalize": "none",
        "random_flip_axes": [1, 2],
        "random_flip_probability": 0.0,
        "curve_roi": {
            "enabled": True,
            "ilm_layer": 1,
            "bm_layer": 8,
            "coordinate_base": 1,
            "rnv_top_margin_pixels": 0,
            "bm_bottom_margin_pixels": 0,
        },
    }
    sample = RNVMatDataset([record], config, training=True)[0]
    assert torch.all(sample["image"][0, 0] == 0.75)
    assert torch.allclose(sample["image"][0, 7], torch.full((4, 4), 0.9))
    assert sample["target"].sum() == 1


def test_sample_preview_writes_arrays_and_bscan_images(tmp_path: Path) -> None:
    oct_image = torch.linspace(0, 1, 8 * 8 * 4).reshape(1, 8, 8, 4)
    octa = torch.flip(oct_image, dims=(1,))
    large_vessel = torch.zeros_like(oct_image)
    large_vessel[:, :, 2:5, 1:3] = 1
    image = torch.cat([oct_image, octa, large_vessel], dim=0)
    target = torch.zeros_like(oct_image)
    target[:, 2:5, 3:6, 2] = 1
    output = save_sample_preview(
        {"image": image, "target": target, "sample_id": "synthetic"},
        tmp_path,
        0,
        bscan_slices=3,
        modalities=["oct", "octa", "large_vessel"],
    )
    assert (output / "sample.npz").is_file()
    assert not (output / "input_bscans.png").exists()
    assert (output / "label_bscans.png").is_file()
    assert (output / "overlay_bscans.png").is_file()
    assert (output / "input_oct_bscans.png").is_file()
    assert (output / "input_octa_bscans.png").is_file()
    assert not (output / "input_large_vessel_bscans.png").exists()
    assert (output / "input_large_vessel_enface.png").is_file()
    assert (output / "metadata.json").is_file()


def test_preview_selection_uses_distinct_lesions_and_includes_negative() -> None:
    class PreviewPool:
        patch_requests = [
            (0, (0, 0, 0)),
            (0, (0, 4, 0)),
            (0, (0, 8, 0)),
            (1, (0, 0, 0)),
            (2, (0, 0, 0)),
            (3, (0, 0, 0)),
            (4, (0, 0, 0)),
        ]
        positive_indices = [0, 1, 2, 3, 4]
        patch_request_lesions = [
            (0, "rnv", 1),
            (0, "rnv", 1),
            (0, "rnv", 1),
            (1, "rnv", 1),
            (2, "rnv", 1),
            None,
            None,
        ]
        patch_request_types = [
            "positive_boundary",
            "positive_complete",
            "positive_boundary",
            "positive_complete",
            "positive_complete",
            "negative_anatomical_coverage",
            "negative_adjacent",
        ]
        negative_indices_by_category = {
            "adjacent": [6],
            "anatomical_coverage": [5],
        }

        def __len__(self) -> int:
            return len(self.patch_requests)

    dataset = PreviewPool()
    selected = _preview_indices(
        dataset,
        count=4,
        seed=42,
        positive_count=3,
        negative_count=1,
    )
    assert len(selected) == len(set(selected)) == 4
    selected_positive_cases = {
        dataset.patch_requests[index][0] for index in selected[:3]
    }
    assert selected_positive_cases == {0, 1, 2}
    assert selected[3] == 5


def test_shifted_curve_roi_uses_rnv_margin_and_shifted_bm() -> None:
    target = np.zeros((32, 2, 2), dtype=np.uint8)
    target[8, 0, 1] = 1
    ilm = np.array([[10, 12], [11, 13]], dtype=np.int32)
    bm = np.array([[20, 22], [21, 23]], dtype=np.int32)
    upper, lower, shift = shifted_curve_roi_surfaces(
        target,
        ilm,
        bm,
        32,
        {
            "curve_roi": {
                "rnv_top_margin_pixels": 2,
                "bm_bottom_margin_pixels": 5,
            }
        },
    )
    assert shift == 6
    np.testing.assert_array_equal(upper, ilm - 6)
    np.testing.assert_array_equal(lower, bm + 6)
    mask = curved_roi_patch_mask(upper, lower, (slice(0, 32), slice(0, 2), slice(0, 2)))
    assert mask[8, 0, 1]
    assert not mask[3, 0, 1]
    assert not mask[28, 0, 1]


def test_intensity_augmentation_does_not_change_binary_vessel_channel() -> None:
    image = np.zeros((2, 4, 4, 4), dtype=np.float32)
    image[0] = 0.5
    image[1, :, 1:3, 1:3] = 1
    mask = np.ones((4, 4, 4), dtype=bool)
    augmented = apply_intensity_augmentations(
        image,
        np.random.default_rng(7),
        channels=[0],
        contrast_probability=1.0,
        contrast_range=[1.2, 1.2],
        noise_probability=1.0,
        noise_std_range=[0.05, 0.05],
        spatial_mask=mask,
    )
    assert not np.array_equal(augmented[0], image[0])
    np.testing.assert_array_equal(augmented[1], image[1])


def test_component_anchors_cover_multiple_and_long_rnv_lesions() -> None:
    target = np.zeros((32, 32, 32), dtype=np.uint8)
    target[4:7, 4:7, 2:5] = 1
    target[20:23, 20:23, 24:27] = 1
    target[12:14, 12:14, 4:28] = 1
    anchors = rnv_component_anchors(target, (16, 16, 8), overlap=0.5)
    assert all(target[anchor] == 1 for anchor in anchors)
    assert any(anchor[0] < 8 and anchor[1] < 8 for anchor in anchors)
    assert any(anchor[0] > 18 and anchor[1] > 18 for anchor in anchors)
    long_component_anchors = [anchor for anchor in anchors if 11 <= anchor[0] <= 14]
    assert len(long_component_anchors) > 1


def test_positive_patch_candidates_add_complete_and_shifted_boundary_views() -> None:
    target = np.zeros((64, 64, 32), dtype=np.uint8)
    target[20:24, 30:34, 14:18] = 1
    candidates = rnv_positive_patch_candidates(
        target,
        target.shape,
        patch_size=(32, 32, 16),
        depth_bounds=(0, 64),
        depth_anchor_fraction=0.33,
        minimum_voxels=1,
        lateral_offset_fraction=0.25,
        bscan_offset_fraction=0.25,
        complete_minimum_coverage=0.8,
        boundary_minimum_coverage=0.2,
        maximum_views_per_component=6,
        maximum_views_per_case=20,
    )
    assert len(candidates) == 6
    assert sum(candidate.view_type == "complete" for candidate in candidates) == 1
    assert sum(candidate.view_type == "boundary" for candidate in candidates) == 5
    assert len({candidate.start for candidate in candidates}) == len(candidates)
    assert all(candidate.foreground_voxels == int(target.sum()) for candidate in candidates)


def test_positive_patch_case_limit_never_drops_required_base_views() -> None:
    target = np.zeros((32, 96, 32), dtype=np.uint8)
    target[10:13, 4:7, 4:7] = 1
    target[10:13, 44:47, 14:17] = 1
    target[10:13, 84:87, 24:27] = 1
    candidates = rnv_positive_patch_candidates(
        target,
        target.shape,
        patch_size=(32, 32, 16),
        depth_bounds=(0, 32),
        minimum_voxels=1,
        maximum_views_per_component=6,
        maximum_views_per_case=2,
    )
    assert {candidate.component_id for candidate in candidates} == {1, 2, 3}
    assert all(candidate.is_base for candidate in candidates)


def test_covering_patch_starts_merge_nearby_rnv_and_slide_to_distant_rnv() -> None:
    anchors = [(5, 5, 2), (7, 7, 4), (22, 22, 25)]
    patch_size = (16, 16, 8)
    starts = covering_patch_starts(
        anchors,
        shape=(32, 32, 32),
        patch_size=patch_size,
        depth_bounds=(0, 32),
    )
    assert len(starts) == 2
    assert all(
        any(
            all(
                start[axis] <= anchor[axis] < start[axis] + patch_size[axis]
                for axis in range(3)
            )
            for start in starts
        )
        for anchor in anchors
    )


def test_rnv_depth_anchor_is_above_patch_center_to_reduce_vitreous() -> None:
    starts = covering_patch_starts(
        [(150, 200, 16)],
        shape=(400, 400, 32),
        patch_size=(304, 304, 32),
        depth_bounds=(0, 400),
        depth_anchor_fraction=0.33,
    )
    assert starts == [(50, 48, 0)]
    assert 150 - starts[0][0] == 100


def test_roi_representative_windows_have_no_overlap() -> None:
    starts = roi_sliding_window_starts(
        shape=(32, 32, 32),
        patch_size=(16, 16, 8),
        depth_bounds=(0, 32),
        overlap=0.0,
    )
    assert starts[0] == (0, 0, 0)
    assert starts[-1] == (16, 16, 24)
    assert sorted({start[0] for start in starts}) == [0, 16]
    assert sorted({start[1] for start in starts}) == [0, 16]
    assert sorted({start[2] for start in starts}) == [0, 8, 16, 24]
    assert len(starts) == 2 * 2 * 4


def test_roi_representative_windows_distribute_non_divisible_remainder() -> None:
    starts = roi_sliding_window_starts(
        shape=(35, 34, 18),
        patch_size=(16, 16, 8),
        depth_bounds=(3, 34),
        overlap=0.0,
    )
    assert starts == [(10, 0, 0), (10, 0, 10), (10, 18, 0), (10, 18, 10)]
    patch_size = (16, 16, 8)
    for index, left in enumerate(starts):
        for right in starts[index + 1 :]:
            assert any(
                left[axis] + patch_size[axis] <= right[axis]
                or right[axis] + patch_size[axis] <= left[axis]
                for axis in range(3)
            )


def test_roi_stratified_candidates_rotate_spatial_phase_per_bscan_slot() -> None:
    candidates = roi_stratified_window_candidates(
        shape=(35, 34, 18),
        patch_size=(16, 16, 8),
        depth_bounds=(3, 34),
        spatial_phases=(0.0, 0.5, 1.0),
    )
    slots = {slot for _, slot in candidates}
    assert slots == {(0, 0), (1, 10)}
    assert len(candidates) == 18
    for slot in slots:
        starts = [start for start, candidate_slot in candidates if candidate_slot == slot]
        assert {start[0] for start in starts} == {3, 10, 18}
        assert {start[1] for start in starts} == {0, 9, 18}
        assert len({start[2] for start in starts}) == 1
