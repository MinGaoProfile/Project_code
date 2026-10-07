from __future__ import annotations

import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.io import savemat
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rnv_seg.config import load_config, modality_count
from rnv_seg.data import (
    assert_no_group_leakage,
    build_validation_strata,
    discover_scans,
    split_train_validation,
    summarize_rnv_lesions,
    vessel_map_dir_for_split,
)
from rnv_seg.data.dataset import build_dataset, load_full_case, target_names_from_config
from rnv_seg.data.preview import save_dataset_previews
from rnv_seg.data.sampling import BalancedPatchBatchSampler
from rnv_seg.data.splits import save_split_manifest
from rnv_seg.engine import (
    calibrate_postprocessing_grid,
    evaluate_binary_case,
    postprocess_prediction_masks,
    save_probability_heatmaps,
    sliding_window_logits,
    train,
)
from rnv_seg.losses import build_loss
from rnv_seg.models import build_model
from rnv_seg.utils import create_run_directory, load_checkpoint, seed_everything


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _records(config: dict[str, Any], manifest_path: Path | None = None):
    data = config["data"]
    record_metadata = None
    train_records = discover_scans(
        data["dataset_root"],
        data["train_split"],
        vessel_map_dir_for_split(data, data["train_split"]),
        data.get("vessel_prior", {}).get("filename_suffix", "_inner_octa_mask.png"),
        require_target=True,
        strict=bool(data.get("strict_discovery", True)),
    )
    if data.get("validation_split"):
        validation_records = discover_scans(
            data["dataset_root"],
            data["validation_split"],
            vessel_map_dir_for_split(data, data["validation_split"]),
            data.get("vessel_prior", {}).get("filename_suffix", "_inner_octa_mask.png"),
            require_target=True,
            strict=bool(data.get("strict_discovery", True)),
        )
        assert_no_group_leakage(
            {"train": train_records, "validation": validation_records}, data.get("split_group", "subject")
        )
    else:
        stratification = data.get("validation_stratification", {})
        strata_by_sample = None
        if stratification.get("enabled", False):
            strata_by_sample, record_metadata = build_validation_strata(
                train_records,
                str(data["mat_variables"]["target"]),
                field_code_regex=str(
                    stratification.get(
                        "field_code_regex",
                        r"(?:^|[-_])(F\d{3})(?:[_-]|$)",
                    )
                ),
                large_component_voxels=int(
                    stratification.get("large_component_voxels", 36_586)
                ),
                connectivity=int(stratification.get("connectivity", 26)),
                minimum_voxels=int(stratification.get("minimum_voxels", 30)),
                progress=True,
            )
        train_records, validation_records = split_train_validation(
            train_records,
            float(data.get("validation_fraction", 0.2)),
            int(config["seed"]),
            data.get("split_group", "subject"),
            strata_by_sample=strata_by_sample,
            search_trials=int(stratification.get("search_trials", 4096)),
            required_validation_strata=stratification.get(
                "required_validation_strata", []
            ),
        )
    if manifest_path is not None:
        save_split_manifest(
            manifest_path,
            train_records,
            validation_records,
            int(config["seed"]),
            data.get("split_version", "v1"),
            data.get("split_group", "subject"),
            data["dataset_root"],
            record_metadata=record_metadata,
        )
    return train_records, validation_records


def _loader(
    records,
    data_config: dict[str, Any],
    training: bool,
    batch_size: int,
    seed: int,
):
    dataset = build_dataset(records, data_config, training)
    loader_options = {
        "num_workers": int(data_config.get("num_workers", 0)),
        "pin_memory": bool(data_config.get("pin_memory", False)),
        "persistent_workers": bool(data_config.get("num_workers", 0) > 0),
    }
    balanced = data_config.get("balanced_patch_sampling", {})
    use_balanced = bool(
        balanced.get("enabled", False)
        and dataset.positive_indices
        and (training or balanced.get("validation_enabled", True))
    )
    if use_balanced:
        batch_sampler = BalancedPatchBatchSampler(
            dataset,
            batch_size,
            balanced,
            seed=seed,
            training=training,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **loader_options)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        **loader_options,
    )


def _optimizer(model: torch.nn.Module, config: dict[str, Any]):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = config.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(config["learning_rate"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _scheduler(optimizer, config: dict[str, Any], epochs: int):
    name = config.get("name", "none").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(config.get("minimum_learning_rate", 0.0))
        )
    if name == "reduce_on_plateau":
        mode = str(config.get("mode", "min"))
        if mode not in {"min", "max"}:
            raise ValueError("reduce_on_plateau scheduler mode must be 'min' or 'max'")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 5)),
            threshold=float(config.get("threshold", 1.0e-4)),
            threshold_mode=str(config.get("threshold_mode", "rel")),
            cooldown=int(config.get("cooldown", 0)),
            min_lr=float(config.get("minimum_learning_rate", 0.0)),
        )
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def _assert_resume_compatible(
    current: dict[str, Any], checkpoint: dict[str, Any]
) -> None:
    comparisons = {
        "seed": (current.get("seed"), checkpoint.get("seed")),
        "data.modalities": (
            current["data"].get("modalities"),
            checkpoint["data"].get("modalities"),
        ),
        "data.patch_size": (
            current["data"].get("patch_size"),
            checkpoint["data"].get("patch_size"),
        ),
        "data.split_version": (
            current["data"].get("split_version"),
            checkpoint["data"].get("split_version"),
        ),
        "data.validation_stratification": (
            current["data"].get("validation_stratification"),
            checkpoint["data"].get("validation_stratification"),
        ),
        "data.lesion_size_groups": (
            current["data"].get("lesion_size_groups"),
            checkpoint["data"].get("lesion_size_groups"),
        ),
        "data.curve_roi": (
            current["data"].get("curve_roi"),
            checkpoint["data"].get("curve_roi"),
        ),
        "data.balanced_patch_sampling": (
            current["data"].get("balanced_patch_sampling"),
            checkpoint["data"].get("balanced_patch_sampling"),
        ),
        "model": (current.get("model"), checkpoint.get("model")),
        "loss": (current.get("loss"), checkpoint.get("loss")),
        "optimizer": (current.get("optimizer"), checkpoint.get("optimizer")),
        "scheduler": (current.get("scheduler"), checkpoint.get("scheduler")),
        "training.checkpoint_metric": (
            current.get("training", {}).get("checkpoint_metric"),
            checkpoint.get("training", {}).get("checkpoint_metric"),
        ),
        "training.checkpoint_mode": (
            current.get("training", {}).get("checkpoint_mode"),
            checkpoint.get("training", {}).get("checkpoint_mode"),
        ),
        "training.checkpoint_strategy": (
            current.get("training", {}).get("checkpoint_strategy"),
            checkpoint.get("training", {}).get("checkpoint_strategy"),
        ),
        "training.batch_size": (
            current.get("training", {}).get("batch_size"),
            checkpoint.get("training", {}).get("batch_size"),
        ),
        "training.gradient_accumulation_steps": (
            current.get("training", {}).get("gradient_accumulation_steps"),
            checkpoint.get("training", {}).get("gradient_accumulation_steps"),
        ),
        "training.amp": (
            current.get("training", {}).get("amp"),
            checkpoint.get("training", {}).get("amp"),
        ),
    }
    checkpoint_full_volume = checkpoint.get("training", {}).get(
        "full_volume_checkpoint"
    )
    if checkpoint_full_volume is not None:
        comparisons["training.full_volume_checkpoint"] = (
            current.get("training", {}).get("full_volume_checkpoint"),
            checkpoint_full_volume,
        )
        comparisons["inference"] = (
            current.get("inference"),
            checkpoint.get("inference"),
        )
    mismatched = [name for name, (left, right) in comparisons.items() if left != right]
    if mismatched:
        raise ValueError(
            "Resume checkpoint is incompatible with the current training config: "
            + ", ".join(mismatched)
        )


def train_from_config(
    config_path: str | Path,
    resume: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    resume_path = Path(resume).expanduser().resolve() if resume else None
    resume_header: dict[str, Any] | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_header = torch.load(resume_path, map_location="cpu", weights_only=False)
        _assert_resume_compatible(config, resume_header["config"])
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    repository = _repository_root()
    run_dir = resume_path.parent.parent if resume_path else create_run_directory(config, repository)
    train_records, validation_records = _records(config, run_dir / "splits" / "manifest.json")
    batch_size = int(config["training"].get("batch_size", 1))
    train_loader = _loader(
        train_records, config["data"], True, batch_size, int(config["seed"])
    )
    preview_config = config["data"].get("sample_preview", {})
    if preview_config.get("enabled", False) and resume_path is None:
        previews = save_dataset_previews(
            train_loader.dataset,
            run_dir / "data_samples",
            int(preview_config.get("count", 3)),
            int(preview_config.get("bscan_slices", 4)),
            int(config["seed"]),
            positive_count=preview_config.get("positive_count"),
            negative_count=int(preview_config.get("negative_count", 0)),
        )
        print(f"Saved {len(previews)} training sample previews to {run_dir / 'data_samples'}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"], modality_count(config), config["data"]["modalities"]
    ).to(device)
    criterion = build_loss(config["loss"])
    optimizer = _optimizer(model, config["optimizer"])
    scheduler = _scheduler(optimizer, config["scheduler"], int(config["training"]["epochs"]))
    resume_payload = None
    if resume_path is not None:
        resume_payload = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            device=device,
        )
        if scheduler is not None and "scheduler_state" not in resume_payload:
            resumed_epoch = int(resume_payload["epoch"])
            scheduler.last_epoch = resumed_epoch
            scheduler._step_count = resumed_epoch + 1
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        print(
            f"Resuming run {run_dir.name} from {resume_path.name} at "
            f"epoch {resume_payload['epoch']}",
            flush=True,
        )
    full_volume_config = config["training"].get("full_volume_checkpoint", {})

    def full_volume_evaluator(
        current_model: torch.nn.Module, epoch: int
    ) -> dict[str, float]:
        return evaluate_full_volume_records(
            current_model,
            validation_records,
            config,
            device,
            criterion=criterion,
            description=f"Epoch {epoch} full-volume validation",
        )

    summary = train(
        model,
        train_loader,
        None,
        criterion,
        optimizer,
        scheduler,
        config,
        run_dir,
        device,
        resume_payload=resume_payload,
        full_volume_evaluator=(
            full_volume_evaluator
            if full_volume_config.get("enabled", False)
            else None
        ),
    )
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def preview_from_config(
    config_path: str | Path,
    output_dir: str | Path,
    count: int | None = None,
) -> Path:
    config = load_config(config_path)
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    train_records, _ = _records(config)
    dataset = build_dataset(train_records, config["data"], training=True)
    preview_config = config["data"].get("sample_preview", {})
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    previews = save_dataset_previews(
        dataset,
        destination,
        int(count if count is not None else preview_config.get("count", 3)),
        int(preview_config.get("bscan_slices", 4)),
        int(config["seed"]),
        positive_count=preview_config.get("positive_count"),
        negative_count=int(preview_config.get("negative_count", 0)),
    )
    print(f"Saved {len(previews)} training sample previews to {destination}", flush=True)
    return destination


def _inference_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(config["model"])
    if model_config.get("name") == "foundation_unet3d":
        model_config["foundation_checkpoint"] = ""
        model_config["require_foundation_checkpoint"] = False
    if model_config.get("name") == "octcube_unet3d":
        model_config["octcube"] = dict(model_config["octcube"])
        model_config["octcube"]["checkpoint"] = ""
        model_config["octcube"]["require_checkpoint"] = False
    return model_config


@torch.no_grad()
def evaluate_full_volume_records(
    model: torch.nn.Module,
    records,
    config: dict[str, Any],
    device: torch.device,
    *,
    criterion: torch.nn.Module | None = None,
    description: str = "Full-volume evaluation",
) -> dict[str, float]:
    """Evaluate complete scans with the exact test-time windowing and cleanup."""
    model.to(device).eval()
    threshold = float(config["inference"].get("threshold", 0.45))
    patch_size = config["inference"].get(
        "patch_size", config["data"]["patch_size"]
    )
    overlap = float(config["inference"].get("sliding_window_overlap", 0.5))
    amp = bool(config["training"].get("amp", True))
    rows: defaultdict[str, list[float]] = defaultdict(list)
    rnv_group_counts: defaultdict[str, int] = defaultdict(int)
    validation_losses: list[float] = []
    target_names = target_names_from_config(config["data"])
    progress = tqdm(records, desc=description, unit="scan", leave=False)
    for record in progress:
        image, target = load_full_case(record, config["data"])
        if target is None:
            raise ValueError(
                f"Validation target is missing for sample {record.sample_id}"
            )
        logits = sliding_window_logits(
            model, image, patch_size, overlap, amp=amp
        )
        if criterion is not None:
            validation_losses.append(
                float(criterion(logits, target[None]).detach())
            )
        probabilities = torch.sigmoid(logits)[0].numpy()
        raw_masks = {
            name: (probabilities[channel] >= threshold).astype(np.uint8)
            for channel, name in enumerate(target_names)
        }
        prediction = postprocess_prediction_masks(raw_masks, config["inference"])
        target_np = target.numpy() > 0.5
        for channel, task_name in enumerate(target_names):
            rnv_group = None
            if task_name == "rnv":
                grouping = config["data"].get("lesion_size_groups", {})
                rnv_group = summarize_rnv_lesions(
                    target_np[channel],
                    large_component_voxels=int(
                        grouping.get("large_component_voxels", 36_586)
                    ),
                    connectivity=int(grouping.get("connectivity", 26)),
                    minimum_voxels=int(grouping.get("minimum_voxels", 30)),
                ).group
                rnv_group_counts[rnv_group] += 1
            metrics = evaluate_binary_case(
                prediction[task_name], target_np[channel], config["metrics"]
            )
            target_positive = bool(target_np[channel].any())
            for name, value in metrics.items():
                key = name if len(target_names) == 1 else f"{task_name}_{name}"
                if np.isfinite(value):
                    rows[key].append(float(value))
                    if task_name == "rnv" and target_positive:
                        positive_key = (
                            f"positive_{name}"
                            if len(target_names) == 1
                            else f"rnv_positive_{name}"
                        )
                        rows[positive_key].append(float(value))
                    if task_name == "rnv" and rnv_group is not None:
                        group_prefix = "" if len(target_names) == 1 else "rnv_"
                        rows[f"{group_prefix}{rnv_group}_{name}"].append(
                            float(value)
                        )
            if task_name == "rnv" and not target_positive:
                prefix = "" if len(target_names) == 1 else "rnv_"
                predicted_mask = prediction[task_name]
                false_positive_voxels = float(predicted_mask.sum())
                rows[f"{prefix}negative_false_positive_voxels"].append(
                    false_positive_voxels
                )
                rows[f"{prefix}negative_false_positive_fraction"].append(
                    float(predicted_mask.mean())
                )
                rows[f"{prefix}negative_max_probability"].append(
                    float(probabilities[channel].max())
                )
                rows[f"{prefix}negative_patch_clean_rate"].append(
                    float(false_positive_voxels == 0)
                )
        dice_key = "dice" if len(target_names) == 1 else "rnv_dice"
        status = {}
        if validation_losses:
            status["val_loss"] = f"{np.mean(validation_losses):.4f}"
        if rows[dice_key]:
            status["val_dice"] = f"{np.mean(rows[dice_key]):.4f}"
        progress.set_postfix(status)
    results = {
        name: float(np.mean(values)) for name, values in rows.items() if values
    }
    if len(target_names) > 1 and {
        "rnv_dice",
        "vasculature_dice",
    } <= results.keys():
        results["multitask_score"] = (
            0.7 * results["rnv_dice"] + 0.3 * results["vasculature_dice"]
        )
    if validation_losses:
        results["validation_loss"] = float(np.mean(validation_losses))
    group_prefix = "" if len(target_names) == 1 else "rnv_"
    for group, count in rnv_group_counts.items():
        results[f"{group_prefix}{group}_case_count"] = float(count)
    selection = config["training"].get("full_volume_checkpoint", {}).get(
        "selection_metric_weights", {}
    )
    weighted_metrics = {
        f"{group_prefix}small_only_dice": float(
            selection.get("small_only_dice", 0.35)
        ),
        f"{group_prefix}small_only_lesion_f1": float(
            selection.get("small_only_lesion_f1", 0.30)
        ),
        f"{group_prefix}large_containing_dice": float(
            selection.get("large_containing_dice", 0.25)
        ),
        f"{group_prefix}negative_patch_clean_rate": float(
            selection.get("negative_clean_rate", 0.10)
        ),
    }
    available = [
        (float(results[name]), weight)
        for name, weight in weighted_metrics.items()
        if weight > 0.0 and name in results and np.isfinite(results[name])
    ]
    if available:
        weight_total = sum(weight for _, weight in available)
        results["stratified_rnv_score"] = sum(
            value * weight for value, weight in available
        ) / weight_total
    return results


def evaluate_from_config(config_path: str | Path, checkpoint: str | Path) -> dict[str, float]:
    config = load_config(config_path)
    _, validation_records = _records(config)
    model_config = _inference_model_config(config)
    model = build_model(model_config, modality_count(config), config["data"]["modalities"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(checkpoint, model, device=device)
    criterion = build_loss(config["loss"])
    return evaluate_full_volume_records(
        model, validation_records, config, device, criterion=criterion
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@torch.no_grad()
def calibrate_from_config(
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    thresholds: list[float] | None = None,
    minimum_areas: list[int] | None = None,
    consecutive_bscans: list[int] | None = None,
    minimum_lesion_sensitivity: float | None = None,
) -> dict[str, Any]:
    """Calibrate probability threshold and RNV cleanup on full validation scans."""

    config = load_config(config_path)
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    _, validation_records = _records(config)
    model_config = _inference_model_config(config)
    model = build_model(model_config, modality_count(config), config["data"]["modalities"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(checkpoint, model, device=device)
    model.to(device).eval()

    calibration = config["inference"].get("calibration", {})
    thresholds = list(
        thresholds if thresholds is not None else calibration.get("thresholds", [0.5, 0.55, 0.6, 0.65])
    )
    minimum_areas = list(
        minimum_areas
        if minimum_areas is not None
        else calibration.get("minimum_2d_component_pixels", [10, 20, 30])
    )
    consecutive_bscans = list(
        consecutive_bscans
        if consecutive_bscans is not None
        else calibration.get("minimum_consecutive_bscans", [5, 7])
    )
    sensitivity_floor = float(
        minimum_lesion_sensitivity
        if minimum_lesion_sensitivity is not None
        else calibration.get("minimum_lesion_sensitivity", 0.82)
    )
    weights = calibration.get(
        "selection_metric_weights", {"positive_dice": 0.5, "lesion_f1": 0.5}
    )
    postprocessing = config["inference"].get("postprocessing", {})
    target_names = target_names_from_config(config["data"])
    rnv_channel = target_names.index("rnv")
    inference_patch = config["inference"].get(
        "patch_size", config["data"]["patch_size"]
    )
    overlap = float(config["inference"].get("sliding_window_overlap", 0.5))
    amp = bool(config["training"].get("amp", True))

    def probability_cases():
        progress = tqdm(
            validation_records,
            desc="Calibrating validation",
            unit="scan",
        )
        for record in progress:
            image, target = load_full_case(record, config["data"])
            if target is None:
                raise ValueError(
                    f"Validation target is missing for sample {record.sample_id}"
                )
            logits = sliding_window_logits(
                model, image, inference_patch, overlap, amp=amp
            )
            probability = torch.sigmoid(logits)[0, rnv_channel].numpy()
            yield probability, target[rnv_channel].numpy() > 0.5

    rows, selected, unconstrained = calibrate_postprocessing_grid(
        probability_cases(),
        thresholds=thresholds,
        minimum_areas=minimum_areas,
        consecutive_bscans=consecutive_bscans,
        metric_config=config["metrics"],
        minimum_lesion_sensitivity=sensitivity_floor,
        positive_dice_weight=float(weights.get("positive_dice", 0.5)),
        lesion_f1_weight=float(weights.get("lesion_f1", 0.5)),
        connectivity_2d=int(postprocessing.get("connectivity_2d", 8)),
        connectivity_3d=int(postprocessing.get("connectivity_3d", 26)),
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "calibration_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    calibrated_config_path: Path | None = None
    if selected is not None:
        calibrated = copy.deepcopy(config)
        calibrated["inference"]["threshold"] = float(selected["threshold"])
        calibrated_post = calibrated["inference"].setdefault("postprocessing", {})
        calibrated_post["minimum_2d_component_pixels"] = int(
            selected["minimum_2d_component_pixels"]
        )
        calibrated_post["minimum_consecutive_bscans"] = int(
            selected["minimum_consecutive_bscans"]
        )
        calibrated_config_path = destination / "calibrated_config.yaml"
        calibrated_config_path.write_text(
            yaml.safe_dump(calibrated, sort_keys=False), encoding="utf-8"
        )

    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "source_config": str(Path(config_path).resolve()),
        "validation_cases": len(validation_records),
        "minimum_lesion_sensitivity": sensitivity_floor,
        "selection_rule": (
            f"{float(weights.get('positive_dice', 0.5))} * positive_dice + "
            f"{float(weights.get('lesion_f1', 0.5))} * lesion_f1, then fewer "
            "false-positive lesions"
        ),
        "grid": {
            "thresholds": thresholds,
            "minimum_2d_component_pixels": minimum_areas,
            "minimum_consecutive_bscans": consecutive_bscans,
        },
        "selected": selected,
        "best_unconstrained": unconstrained,
        "results_csv": str(results_path.resolve()),
        "calibrated_config": (
            str(calibrated_config_path.resolve())
            if calibrated_config_path is not None
            else None
        ),
    }
    safe_summary = _json_safe(summary)
    (destination / "calibration_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    if selected is None:
        print(
            "No calibration combination satisfied the lesion-sensitivity floor; "
            "no calibrated_config.yaml was written.",
            flush=True,
        )
    return safe_summary


def infer_from_config(
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    max_scans: int | None = None,
) -> None:
    config = load_config(config_path)
    data = config["data"]
    records = discover_scans(
        data["dataset_root"],
        data["test_split"],
        vessel_map_dir_for_split(data, data["test_split"]),
        data.get("vessel_prior", {}).get("filename_suffix", "_inner_octa_mask.png"),
        require_target=False,
        strict=bool(data.get("strict_discovery", True)),
    )
    if max_scans is not None:
        records = records[: max(0, int(max_scans))]
    model_config = _inference_model_config(config)
    model = build_model(model_config, modality_count(config), config["data"]["modalities"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(checkpoint, model, device=device)
    model.to(device).eval()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    threshold = float(config["inference"].get("threshold", 0.45))
    output_format = str(config["inference"].get("output_format", "npz")).lower()
    target_names = target_names_from_config(data)
    for record in tqdm(records, desc="Test inference", unit="scan"):
        image, _ = load_full_case(record, data)
        inference_patch = config["inference"].get("patch_size", data["patch_size"])
        logits = sliding_window_logits(
            model,
            image,
            inference_patch,
            float(config["inference"].get("sliding_window_overlap", 0.5)),
            amp=bool(config["training"].get("amp", True)),
        )
        probabilities = torch.sigmoid(logits)[0].numpy()
        raw_masks = {
            name: (probabilities[channel] >= threshold).astype(np.uint8)
            for channel, name in enumerate(target_names)
        }
        masks = postprocess_prediction_masks(raw_masks, config["inference"])
        case_id = record.scan_dir.name
        case_destination = destination / case_id
        case_destination.mkdir(parents=True, exist_ok=True)
        if output_format == "mat":
            configured_variables = config["inference"].get("mat_variables", {})
            output_variables = {
                name: str(
                    configured_variables.get(
                        name,
                        config["inference"].get("mat_variable", data["mat_variables"]["target"])
                        if name == "rnv"
                        else data["mat_variables"].get(f"{name}_target", f"{name}_mask"),
                    )
                )
                for name in target_names
            }
            filename = record.image_path.name.replace("_oct_octa.mat", "_mask.mat")
            payload = {output_variables[name]: masks[name] for name in target_names}
            if config["inference"].get("save_probabilities", False):
                payload.update(
                    {
                        f"{output_variables[name]}_probability": probabilities[channel]
                        for channel, name in enumerate(target_names)
                    }
                )
            savemat(
                case_destination / filename,
                payload,
                do_compression=bool(config["inference"].get("mat_compression", True)),
            )
        elif output_format == "npz":
            payload = dict(masks)
            if config["inference"].get("save_probabilities", False):
                payload.update(
                    {f"{name}_probability": probabilities[channel] for channel, name in enumerate(target_names)}
                )
            np.savez_compressed(case_destination / "prediction.npz", **payload)
        else:
            raise ValueError(f"Unknown inference output format: {output_format}")
        heatmap_config = config["inference"].get("probability_heatmaps", {})
        if heatmap_config.get("enabled", False):
            image_numpy = image.numpy()
            for channel, name in enumerate(target_names):
                save_probability_heatmaps(
                    probabilities[channel],
                    image_numpy,
                    data["modalities"],
                    case_destination,
                    name,
                    threshold,
                    bscan_count=int(heatmap_config.get("bscan_count", 6)),
                    overlay_alpha=float(heatmap_config.get("overlay_alpha", 0.70)),
                )
        tqdm.write(
            f"Saved {case_id}: "
            + ", ".join(
                f"{name}={int(raw_masks[name].sum())}->{int(masks[name].sum())} voxels"
                for name in target_names
            )
            + f" -> {case_destination}"
        )
