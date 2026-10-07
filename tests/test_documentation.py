from pathlib import Path

from rnv_seg.config import load_config


def _compact_list(values: list[object]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def test_data_processing_defaults_are_synchronized_in_documentation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "base.yaml")
    multimodal = load_config(root / "configs" / "experiments" / "unet3d_multimodal.yaml")
    readme = (root / "README.md").read_text(encoding="utf-8").replace(" ", "")
    data_document = (root / "docs" / "DATA_PROCESSING.md").read_text(
        encoding="utf-8"
    ).replace(" ", "")
    documents = (readme, data_document)

    expected_lists = (
        config["data"]["patch_size"],
        config["inference"]["patch_size"],
        multimodal["data"]["modalities"],
        config["metrics"]["volume"]["voxel_spacing_mm"],
        config["data"]["intensity_augmentation"]["contrast"]["range"],
        config["data"]["intensity_augmentation"]["gaussian_noise"]["std_range"],
    )
    for document in documents:
        for values in expected_lists:
            assert _compact_list(values) in document
        assert "docs/DATA_PROCESSING.md" in document
        assert "effectivebatchsize4" in document
    assert "unmaskedoriginalimage" in readme
    assert "未遮罩" in data_document
    depth_fraction = str(config["data"]["curve_roi"]["rnv_depth_position_fraction"])
    assert depth_fraction in readme
    assert depth_fraction in data_document
    postprocessing = config["inference"]["postprocessing"]
    for document in documents:
        assert str(postprocessing["minimum_2d_component_pixels"]) in document
        assert str(postprocessing["minimum_consecutive_bscans"]) in document
        calibration = config["inference"]["calibration"]
        assert _compact_list(calibration["thresholds"]) in document
        assert _compact_list(calibration["minimum_2d_component_pixels"]) in document
        assert _compact_list(calibration["minimum_consecutive_bscans"]) in document
        assert str(calibration["minimum_lesion_sensitivity"]) in document
        assert "calibrated_config.yaml" in document
        assert "predicts/<case_id>" in document
        assert "best_full_volume.pt" in document
        assert "best_loss.pt" in document
        assert "best_dice.pt" in document
        assert "v3_subject_lesion_size_stratified" in document
        assert "rnv_voxels" in document
        assert "patch_plus_full_volume_v2" in document
        assert "negative_patch_clean_rate" in document
        assert "preview_dataset_index" in document
        assert str(
            config["training"]["full_volume_checkpoint"]["every_n_epochs"]
        ) in document
        assert str(config["training"]["early_stopping_patience"]) in document

    agent_rules = (root / "AGENTS.md").read_text(encoding="utf-8")
    assert "README.md" in agent_rules
    assert "docs/DATA_PROCESSING.md" in agent_rules
