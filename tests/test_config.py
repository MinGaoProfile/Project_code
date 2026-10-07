from pathlib import Path

import pytest
import yaml

from rnv_seg.config import load_config


def test_defaults_merge_and_environment_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project": {"experiment_name": "base"},
                "data": {"dataset_root": "${RNV_TEST_ROOT}", "modalities": ["oct"], "patch_size": [8, 8, 8]},
                "model": {"levels": 4},
                "loss": {},
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    child.write_text("defaults: [base.yaml]\nproject:\n  experiment_name: child\n", encoding="utf-8")
    monkeypatch.setenv("RNV_TEST_ROOT", str(tmp_path))
    config = load_config(child)
    assert config["project"]["experiment_name"] == "child"
    assert config["data"]["dataset_root"] == str(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "octcube_unet_oct_frozen.yaml",
        "octcube_unet_oct_finetune.yaml",
        "octcube_unet_oct_octa.yaml",
        "octcube_unet_multimodal.yaml",
        "octcube_unet_multimodal_attention.yaml",
    ],
)
def test_required_octcube_experiment_configs_load(name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "experiments" / name)
    assert config["model"]["name"] == "octcube_unet3d"
    assert "oct" in config["data"]["modalities"]


@pytest.mark.parametrize(
    "name",
    [
        "unet3d_oct.yaml",
        "unet3d_oct_octa.yaml",
        "unet3d_multimodal.yaml",
        "unet3d_multimodal_vasculature.yaml",
        "diffusion_unet_oct_octa.yaml",
    ],
)
def test_unet_experiments_use_regularized_residual_defaults(name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "experiments" / name)
    assert config["model"]["base_channels"] == 24
    assert config["model"]["levels"] == 5
    assert config["model"]["dropout"] == 0.1
    assert config["optimizer"]["weight_decay"] == 1.0e-5


@pytest.mark.parametrize(
    "experiment",
    sorted((Path(__file__).resolve().parents[1] / "configs" / "experiments").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_all_experiments_share_segmentation_strategy(experiment: Path) -> None:
    config = load_config(experiment)
    assert config["data"]["split_group"] == "subject"
    assert config["data"]["split_version"] == "v3_subject_lesion_size_stratified"
    assert config["data"]["validation_stratification"] == {
        "enabled": True,
        "field_code_regex": r"(?:^|[-_])(F\d{3})(?:[_-]|$)",
        "large_component_voxels": 36586,
        "connectivity": 26,
        "minimum_voxels": 30,
        "search_trials": 10000,
        "required_validation_strata": ["presence:negative"],
    }
    assert config["data"]["patch_size"] == [480, 224, 24]
    assert "foreground_patch_probability" not in config["data"]
    assert "context_patch_sampling" not in config["data"]
    assert config["data"]["roi_sliding_window"] == {
        "enabled": True,
        "overlap": 0.0,
    }
    assert config["data"]["balanced_patch_sampling"] == {
        "enabled": True,
        "validation_enabled": False,
        "positive_fraction": 0.5,
        "positive_category_fractions": {
            "complete": 0.55,
            "boundary": 0.45,
        },
        "positive_scan_group_fractions": {
            "small_only": 0.7,
            "large_containing": 0.3,
        },
        "maximum_positive_views_per_lesion_per_epoch": 2,
        "negative_category_fractions": {
            "adjacent": 0.3,
            "anatomical_coverage": 0.7,
        },
        "adjacent_margin_pixels": [0, 152, 24],
        "spatial_phases": [0.0, 0.5, 1.0],
        "hard_negative_fraction": 0.5,
        "hard_positive_fraction": 0.3,
        "hard_positive_mining": {
            "enabled": True,
            "warmup_epochs": 3,
            "refresh_every": 1,
            "candidate_limit": 256,
            "scoring_batch_size": 1,
            "recheck_fraction": 0.5,
            "probability_threshold": 0.45,
            "minimum_missed_voxels": 10,
            "minimum_missed_fraction": 0.05,
            "contrast_ring_radius": 4,
            "missed_fraction_weight": 0.5,
            "confidence_shortfall_weight": 0.3,
            "low_contrast_weight": 0.2,
            "small_only_bonus": 0.15,
            "boundary_bonus": 0.10,
            "maximum_pool_size": 256,
            "maximum_per_case": 12,
            "maximum_per_lesion": 3,
            "maximum_views_per_lesion_per_epoch": 3,
            "score_decay": 0.9,
            "minimum_retained_score": 0.15,
        },
        "hard_negative_mining": {
            "enabled": True,
            "strategy_version": "patch_plus_full_volume_v2",
            "warmup_epochs": 8,
            "refresh_every": 2,
            "candidate_limit": 512,
            "scoring_batch_size": 1,
            "recheck_fraction": 0.25,
            "probability_threshold": 0.4,
            "minimum_false_positive_voxels": 20,
            "use_inference_postprocessing": True,
            "maximum_pool_size": 768,
            "maximum_per_case": 16,
            "top_probability_voxels": 4096,
            "voxel_count_weight": 0.35,
            "score_decay": 0.9,
            "minimum_retained_score": 0.35,
            "full_volume_refresh": {
                "enabled": True,
                "warmup_epochs": 8,
                "refresh_every": 4,
                "cases_per_refresh": 4,
                "probability_threshold": 0.5,
            },
        },
    }
    assert config["loss"]["focal_tversky_alpha"] == 0.6
    assert config["loss"]["focal_tversky_beta"] == 0.4
    assert config["scheduler"] == {
        "name": "reduce_on_plateau",
        "monitor": "validation_loss",
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "smoothing_window": 3,
        "threshold": 2.0e-3,
        "threshold_mode": "abs",
        "cooldown": 0,
        "minimum_learning_rate": 1.0e-6,
    }
    assert config["training"]["full_volume_checkpoint"] == {
        "enabled": True,
        "every_n_epochs": 1,
        "metric": "validation_loss",
        "mode": "min",
        "preferred_metric": "validation_loss",
        "preferred_mode": "min",
        "preferred_min_delta": 2.0e-3,
        "selection_metric_weights": {
            "small_only_dice": 0.35,
            "small_only_lesion_f1": 0.30,
            "large_containing_dice": 0.25,
            "negative_clean_rate": 0.10,
        },
        "evaluate_at_training_end": True,
    }
    assert config["training"]["batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert config["training"]["amp"] is True
    assert config["training"]["early_stopping_patience"] == 18
    assert config["training"]["checkpoint_metric"] == "validation_loss"
    assert config["training"]["checkpoint_strategy"] == "full_volume_loss_v3"
    assert config["training"]["checkpoint_mode"] == "min"
    assert config["training"]["checkpoint_min_delta"] == 2.0e-3
    assert config["data"]["random_flip_axes"] == [1, 2]
    assert config["data"]["intensity_augmentation"] == {
        "enabled": True,
        "contrast": {"probability": 0.5, "range": [0.8, 1.2]},
        "gaussian_noise": {"probability": 0.3, "std_range": [0.0, 0.05]},
    }
    assert config["data"]["sample_preview"] == {
        "enabled": True,
        "count": 4,
        "positive_count": 3,
        "negative_count": 1,
        "bscan_slices": 4,
    }
    assert config["data"]["curve_roi"] == {
        "enabled": True,
        "variable": "ManualCurveData",
        "ilm_layer": 1,
        "bm_layer": 8,
        "coordinate_base": 1,
        "rnv_top_margin_pixels": 20,
        "rnv_depth_position_fraction": 0.40,
        "bm_bottom_margin_pixels": 100,
        "required": True,
    }
    expected_sampling = {
        "enabled": True,
        "connectivity": 26,
        "minimum_voxels": 30,
        "component_overlap": 0.5,
        "positive_views": {
            "enabled": True,
            "lateral_offset_fraction": 0.25,
            "bscan_offset_fraction": 0.25,
            "complete_minimum_coverage": 0.8,
            "boundary_minimum_coverage": 0.2,
            "maximum_views_per_component": 6,
            "maximum_views_per_case": 20,
        },
    }
    for key, value in expected_sampling.items():
        assert config["data"]["lesion_sampling"][key] == value
    assert config["inference"]["patch_size"] == [768, 600, 24]
    assert config["inference"]["threshold"] == 0.45
    assert config["inference"]["save_probabilities"] is False
    assert config["inference"]["probability_heatmaps"] == {
        "enabled": True,
        "bscan_count": 6,
        "overlay_alpha": 0.70,
    }
    assert config["inference"]["output_format"] == "mat"
    assert config["inference"]["mat_variable"] == "mask"
    expected_postprocessing = {
        "enabled": True,
        "minimum_2d_component_pixels": 30,
        "minimum_consecutive_bscans": 5,
        "connectivity_2d": 8,
        "connectivity_3d": 26,
    }
    for key, value in expected_postprocessing.items():
        assert config["inference"]["postprocessing"][key] == value
    assert config["inference"]["calibration"] == {
        "thresholds": [0.45, 0.50, 0.55, 0.60, 0.65],
        "minimum_2d_component_pixels": [5, 10, 20, 30],
        "minimum_consecutive_bscans": [3, 5, 7],
        "minimum_lesion_sensitivity": 0.82,
        "selection_metric_weights": {
            "positive_dice": 0.5,
            "lesion_f1": 0.5,
        },
    }
    assert config["metrics"]["volume"]["voxel_spacing_mm"] == [
        0.003,
        0.015,
        0.015,
    ]
    assert config["loss"]["components"]["focal_tversky"] == 1.0
    if "multimodal" in experiment.stem:
        assert config["data"]["modalities"] == ["oct", "octa", "large_vessel"]
