from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Environment variable {name!r} is required by the configuration")

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, recursively merge defaults, and expand environment variables."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    defaults = current.pop("defaults", [])
    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = (config_path.parent / default).resolve()
        merged = _deep_merge(merged, load_config(default_path))
    merged = _deep_merge(merged, current)
    merged = _expand_env(merged)
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    required = ("project", "data", "model", "loss", "training")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    modalities = config["data"].get("modalities", [])
    supported = {"oct", "octa", "artery", "vein", "large_vessel"}
    unknown = set(modalities) - supported
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if not modalities:
        raise ValueError("At least one input modality is required")
    stratification = config["data"].get("validation_stratification", {})
    if stratification.get("enabled", False):
        if config["data"].get("split_group") != "subject":
            raise ValueError(
                "validation_stratification requires data.split_group=subject"
            )
        if int(stratification.get("minimum_voxels", 30)) < 1:
            raise ValueError(
                "validation_stratification.minimum_voxels must be positive"
            )
        if int(stratification.get("large_component_voxels", 0)) < int(
            stratification.get("minimum_voxels", 30)
        ):
            raise ValueError(
                "validation_stratification.large_component_voxels must be at "
                "least minimum_voxels"
            )
        if int(stratification.get("connectivity", 26)) not in {6, 18, 26}:
            raise ValueError(
                "validation_stratification.connectivity must be 6, 18, or 26"
            )
        if int(stratification.get("search_trials", 0)) < 1:
            raise ValueError(
                "validation_stratification.search_trials must be positive"
            )
    target_names = config["data"].get("target_names", ["rnv"])
    if "rnv" not in target_names or len(set(target_names)) != len(target_names):
        raise ValueError("data.target_names must contain rnv and have no duplicates")
    if config["model"].get("name") == "multitask_unet3d":
        heads = config["model"].get("output_heads", [])
        if heads != target_names:
            raise ValueError("model.output_heads must exactly match data.target_names")
        tasks = list(config["loss"].get("tasks", {}))
        if tasks != target_names:
            raise ValueError("loss.tasks must exactly match data.target_names")
    patch_size = config["data"].get("patch_size", [])
    if len(patch_size) != 3 or any(int(size) <= 0 for size in patch_size):
        raise ValueError("data.patch_size must contain three positive integers")
    curve_roi = config["data"].get("curve_roi", {})
    if curve_roi.get("enabled", False):
        depth_fraction = float(
            curve_roi.get("rnv_depth_position_fraction", 0.40)
        )
        if not 0.0 < depth_fraction < 1.0:
            raise ValueError(
                "data.curve_roi.rnv_depth_position_fraction must be in (0, 1)"
            )
    lesion_groups = config["data"].get("lesion_size_groups", {})
    if lesion_groups:
        lesion_minimum = int(lesion_groups.get("minimum_voxels", 30))
        if lesion_minimum < 1:
            raise ValueError("data.lesion_size_groups.minimum_voxels must be positive")
        if int(lesion_groups.get("large_component_voxels", 0)) < lesion_minimum:
            raise ValueError(
                "data.lesion_size_groups.large_component_voxels must be at least "
                "minimum_voxels"
            )
        if int(lesion_groups.get("connectivity", 26)) not in {6, 18, 26}:
            raise ValueError(
                "data.lesion_size_groups.connectivity must be 6, 18, or 26"
            )
    balanced = config["data"].get("balanced_patch_sampling", {})
    hard_positive_fraction = float(balanced.get("hard_positive_fraction", 0.3))
    if not 0.0 <= hard_positive_fraction <= 1.0:
        raise ValueError("hard_positive_fraction must be in [0, 1]")
    positive_categories = balanced.get("positive_category_fractions", {})
    if positive_categories:
        if set(positive_categories) != {"complete", "boundary"}:
            raise ValueError(
                "positive_category_fractions must define complete and boundary"
            )
        if any(float(value) < 0.0 for value in positive_categories.values()) or sum(
            float(value) for value in positive_categories.values()
        ) <= 0.0:
            raise ValueError(
                "positive_category_fractions must be non-negative and sum positive"
            )
    group_fractions = balanced.get("positive_scan_group_fractions", {})
    if group_fractions:
        if set(group_fractions) != {"small_only", "large_containing"}:
            raise ValueError(
                "positive_scan_group_fractions must define small_only and "
                "large_containing"
            )
        if any(float(value) < 0.0 for value in group_fractions.values()) or sum(
            float(value) for value in group_fractions.values()
        ) <= 0.0:
            raise ValueError(
                "positive_scan_group_fractions must be non-negative and sum positive"
            )
    hard_negative = (
        config["data"]
        .get("balanced_patch_sampling", {})
        .get("hard_negative_mining", {})
    )
    hard_positive = balanced.get("hard_positive_mining", {})
    if hard_positive.get("enabled", False):
        for name in (
            "refresh_every",
            "candidate_limit",
            "scoring_batch_size",
            "minimum_missed_voxels",
            "contrast_ring_radius",
            "maximum_pool_size",
            "maximum_per_case",
            "maximum_per_lesion",
            "maximum_views_per_lesion_per_epoch",
        ):
            if int(hard_positive.get(name, 1)) < 1:
                raise ValueError(f"hard_positive_mining.{name} must be positive")
        for name, default in (
            ("probability_threshold", 0.45),
            ("minimum_missed_fraction", 0.05),
            ("recheck_fraction", 0.5),
            ("score_decay", 0.9),
        ):
            value = float(hard_positive.get(name, default))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"hard_positive_mining.{name} must be in [0, 1]")
        score_weights = [
            float(hard_positive.get("missed_fraction_weight", 0.5)),
            float(hard_positive.get("confidence_shortfall_weight", 0.3)),
            float(hard_positive.get("low_contrast_weight", 0.2)),
        ]
        if any(value < 0.0 for value in score_weights) or sum(score_weights) <= 0.0:
            raise ValueError(
                "hard_positive_mining score weights must be non-negative and sum positive"
            )
        for name in ("small_only_bonus", "boundary_bonus"):
            if float(hard_positive.get(name, 0.0)) < 0.0:
                raise ValueError(f"hard_positive_mining.{name} must be non-negative")
        if float(hard_positive.get("minimum_retained_score", 0.15)) < 0.0:
            raise ValueError(
                "hard_positive_mining.minimum_retained_score must be non-negative"
            )
        ordinary_views = int(
            balanced.get("maximum_positive_views_per_lesion_per_epoch", 2)
        )
        if int(
            hard_positive.get(
                "maximum_views_per_lesion_per_epoch", ordinary_views + 1
            )
        ) < ordinary_views:
            raise ValueError(
                "hard-positive maximum_views_per_lesion_per_epoch cannot be "
                "smaller than the ordinary positive-view limit"
            )
    if hard_negative.get("enabled", False):
        if hard_negative.get("strategy_version") not in {
            None,
            "patch_plus_full_volume_v2",
        }:
            raise ValueError("Unknown hard-negative mining strategy_version")
        for name in (
            "refresh_every",
            "candidate_limit",
            "scoring_batch_size",
            "minimum_false_positive_voxels",
            "maximum_pool_size",
            "maximum_per_case",
        ):
            if int(hard_negative.get(name, 1)) < 1:
                raise ValueError(f"hard_negative_mining.{name} must be positive")
        probability_threshold = float(
            hard_negative.get("probability_threshold", 0.5)
        )
        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError(
                "hard_negative_mining.probability_threshold must be in [0, 1]"
            )
        full_refresh = hard_negative.get("full_volume_refresh", {})
        if full_refresh.get("enabled", False):
            for name in ("refresh_every", "cases_per_refresh"):
                if int(full_refresh.get(name, 1)) < 1:
                    raise ValueError(
                        "hard_negative_mining.full_volume_refresh."
                        f"{name} must be positive"
                    )
            full_threshold = float(
                full_refresh.get(
                    "probability_threshold",
                    config.get("inference", {}).get("threshold", 0.45),
                )
            )
            if not 0.0 <= full_threshold <= 1.0:
                raise ValueError(
                    "hard_negative_mining.full_volume_refresh."
                    "probability_threshold must be in [0, 1]"
                )
    inference = config.get("inference", {})
    heatmaps = inference.get("probability_heatmaps", {})
    if heatmaps.get("enabled", False):
        if int(heatmaps.get("bscan_count", 6)) < 1:
            raise ValueError("inference.probability_heatmaps.bscan_count must be positive")
        overlay_alpha = float(heatmaps.get("overlay_alpha", 0.70))
        if not 0.0 <= overlay_alpha <= 1.0:
            raise ValueError(
                "inference.probability_heatmaps.overlay_alpha must be in [0, 1]"
            )
    calibration = inference.get("calibration", {})
    if calibration:
        thresholds = calibration.get("thresholds", [])
        if not thresholds or any(
            not 0.0 <= float(value) <= 1.0 for value in thresholds
        ):
            raise ValueError(
                "inference.calibration.thresholds must be values in [0, 1]"
            )
        for name in (
            "minimum_2d_component_pixels",
            "minimum_consecutive_bscans",
        ):
            values = calibration.get(name, [])
            if not values or any(int(value) < 1 for value in values):
                raise ValueError(
                    f"inference.calibration.{name} must contain positive integers"
                )
        sensitivity = float(
            calibration.get("minimum_lesion_sensitivity", 0.82)
        )
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(
                "inference.calibration.minimum_lesion_sensitivity must be in [0, 1]"
            )
        weights = calibration.get("selection_metric_weights", {})
        positive_weight = float(weights.get("positive_dice", 0.5))
        lesion_weight = float(weights.get("lesion_f1", 0.5))
        if (
            positive_weight < 0.0
            or lesion_weight < 0.0
            or positive_weight + lesion_weight <= 0.0
        ):
            raise ValueError(
                "inference.calibration.selection_metric_weights must be "
                "non-negative and sum positive"
            )
    levels = int(config["model"].get("levels", 4))
    divisor = 2 ** (levels - 1)
    if any(int(size) < divisor for size in patch_size):
        raise ValueError(
            f"Patch dimensions must each be at least {divisor} for {levels} levels"
        )
    full_volume = config["training"].get("full_volume_checkpoint", {})
    if full_volume.get("enabled", False):
        if int(full_volume.get("every_n_epochs", 5)) < 1:
            raise ValueError(
                "training.full_volume_checkpoint.every_n_epochs must be positive"
            )
        if str(full_volume.get("mode", "max")) not in {"min", "max"}:
            raise ValueError(
                "training.full_volume_checkpoint.mode must be 'min' or 'max'"
            )
        if not str(full_volume.get("metric", "rnv_dice")).strip():
            raise ValueError(
                "training.full_volume_checkpoint.metric must not be empty"
            )
        if int(full_volume.get("every_n_epochs", 1)) != 1:
            raise ValueError(
                "Full-volume validation must run every epoch when it controls "
                "checkpointing, LR scheduling, and early stopping"
            )
        if full_volume.get("metric") != config["training"].get(
            "checkpoint_metric"
        ) or full_volume.get("mode") != config["training"].get(
            "checkpoint_mode"
        ):
            raise ValueError(
                "Full-volume and training checkpoint metric/mode must match"
            )
        if full_volume.get("metric") != "validation_loss" or full_volume.get(
            "mode"
        ) != "min":
            raise ValueError(
                "Model selection, LR scheduling, and early stopping must use "
                "validation_loss in min mode"
            )
        scheduler = config.get("scheduler", {})
        if str(scheduler.get("name", "none")).lower() != "reduce_on_plateau":
            raise ValueError(
                "Full-volume validation-loss selection requires "
                "scheduler.name=reduce_on_plateau"
            )
        if scheduler.get("monitor") != full_volume.get("metric") or scheduler.get(
            "mode"
        ) != full_volume.get("mode"):
            raise ValueError(
                "Scheduler monitor/mode must match the full-volume checkpoint "
                "metric/mode"
            )
        if full_volume.get("preferred_metric") != "validation_loss":
            raise ValueError(
                "training.full_volume_checkpoint.preferred_metric must be "
                "validation_loss"
            )
        if full_volume.get("preferred_mode") != "min":
            raise ValueError(
                "training.full_volume_checkpoint.preferred_mode must be min"
            )
        for name in ("checkpoint_min_delta",):
            if float(config["training"].get(name, 0.0)) < 0.0:
                raise ValueError(f"training.{name} must be non-negative")
        if float(full_volume.get("preferred_min_delta", 0.0)) < 0.0:
            raise ValueError(
                "training.full_volume_checkpoint.preferred_min_delta must be "
                "non-negative"
            )
        if float(full_volume.get("preferred_min_delta", 0.0)) != float(
            config["training"].get("checkpoint_min_delta", 0.0)
        ):
            raise ValueError(
                "Preferred and default loss checkpoints must use the same min_delta"
            )
        selection_weights = full_volume.get("selection_metric_weights", {})
        if selection_weights and (
            any(float(value) < 0.0 for value in selection_weights.values())
            or sum(float(value) for value in selection_weights.values()) <= 0.0
        ):
            raise ValueError(
                "training.full_volume_checkpoint.selection_metric_weights must "
                "be non-negative and sum positive"
            )
        if config["training"].get("checkpoint_strategy") not in {
            "dual_full_volume_loss_dice_v1",
            "size_stratified_full_volume_v2",
            "full_volume_loss_v3",
        }:
            raise ValueError(
                "Unknown training.checkpoint_strategy"
            )
    scheduler = config.get("scheduler", {})
    if int(scheduler.get("smoothing_window", 1)) < 1:
        raise ValueError("scheduler.smoothing_window must be positive")
    if config["model"].get("name") == "octcube_unet3d":
        if "oct" not in modalities:
            raise ValueError("octcube_unet3d requires data.modalities to include oct")
        octcube = config["model"].get("octcube", {})
        axis_order = octcube.get("axis_order", [2, 0, 1])
        if sorted(axis_order) != [0, 1, 2]:
            raise ValueError("model.octcube.axis_order must be a permutation of [0, 1, 2]")
        input_size = int(octcube.get("input_size", 256))
        spatial_patch = int(octcube.get("patch_size", 16))
        num_frames = int(octcube.get("num_frames", 64))
        temporal_patch = int(octcube.get("temporal_patch_size", 2))
        if input_size % spatial_patch or num_frames % temporal_patch:
            raise ValueError("OCTCube input dimensions must be divisible by its patch dimensions")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def modality_count(config: dict[str, Any]) -> int:
    return len(config["data"]["modalities"])
