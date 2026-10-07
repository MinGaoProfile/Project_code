from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from rnv_seg.data.dataset import load_full_case
from rnv_seg.engine.evaluator import evaluate_loader
from rnv_seg.engine.inference import (
    postprocess_prediction_masks,
    sliding_window_logits,
)
from rnv_seg.utils.checkpoint import save_checkpoint
from rnv_seg.utils.experiment import append_jsonl


def _replace_checkpoint_alias(source: Path, destination: Path) -> None:
    """Atomically make destination reference source, copying only if links fail."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def _false_positive_patch_score(
    probability: torch.Tensor,
    retained_mask: np.ndarray,
    hard_config: dict[str, Any],
) -> float | None:
    """Rank a pure-negative patch by persistent confidence and spatial extent."""
    retained = torch.from_numpy(np.asarray(retained_mask, dtype=bool))
    false_positive_probabilities = probability[retained]
    false_positive_voxels = int(false_positive_probabilities.numel())
    minimum_false_positive_voxels = max(
        1, int(hard_config.get("minimum_false_positive_voxels", 10))
    )
    if false_positive_voxels < minimum_false_positive_voxels:
        return None
    top_voxels = max(1, int(hard_config.get("top_probability_voxels", 4096)))
    count = min(top_voxels, false_positive_voxels)
    confidence = float(
        torch.topk(false_positive_probabilities, count).values.mean()
    )
    voxel_count_weight = float(hard_config.get("voxel_count_weight", 0.25))
    if voxel_count_weight < 0.0:
        raise ValueError("hard_negative_mining.voxel_count_weight must be non-negative")
    extent = math.log1p(false_positive_voxels) / math.log1p(probability.numel())
    return confidence + voxel_count_weight * extent


def _hard_positive_patch_score(
    probability: torch.Tensor,
    target: torch.Tensor,
    oct_image: torch.Tensor,
    hard_config: dict[str, Any],
) -> float | None:
    """Score labelled RNV that remains below threshold, including low contrast."""
    reference = target > 0.5
    positive_voxels = int(reference.sum())
    if positive_voxels == 0:
        return None
    threshold = float(hard_config.get("probability_threshold", 0.45))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("hard_positive_mining.probability_threshold must be in [0, 1]")
    missed = reference & (probability < threshold)
    missed_voxels = int(missed.sum())
    missed_fraction = missed_voxels / positive_voxels
    if missed_voxels < int(hard_config.get("minimum_missed_voxels", 10)):
        return None
    if missed_fraction < float(hard_config.get("minimum_missed_fraction", 0.05)):
        return None

    missed_probability = probability[missed]
    confidence_shortfall = float(
        ((threshold - missed_probability).clamp_min(0.0) / max(threshold, 1.0e-6)).mean()
    )
    radius = max(1, int(hard_config.get("contrast_ring_radius", 4)))
    coordinates = torch.nonzero(reference, as_tuple=False)
    lower = torch.clamp(coordinates.min(dim=0).values - radius, min=0)
    upper = torch.minimum(
        coordinates.max(dim=0).values + radius + 1,
        torch.as_tensor(reference.shape, device=coordinates.device),
    )
    context_slices = tuple(
        slice(int(begin), int(end)) for begin, end in zip(lower, upper, strict=True)
    )
    context_reference = reference[context_slices]
    ring_values = oct_image[context_slices][~context_reference]
    if ring_values.numel():
        lesion_mean = oct_image[reference].float().mean()
        ring_values = ring_values.float()
        standardized_contrast = torch.abs(lesion_mean - ring_values.mean()) / (
            ring_values.std(unbiased=False) + 0.05
        )
        low_contrast = float(1.0 / (1.0 + standardized_contrast))
    else:
        low_contrast = 0.0

    weights = {
        "missed_fraction": float(hard_config.get("missed_fraction_weight", 0.5)),
        "confidence_shortfall": float(
            hard_config.get("confidence_shortfall_weight", 0.3)
        ),
        "low_contrast": float(hard_config.get("low_contrast_weight", 0.2)),
    }
    if any(value < 0.0 for value in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError("hard_positive_mining score weights must be non-negative and sum positive")
    return (
        weights["missed_fraction"] * missed_fraction
        + weights["confidence_shortfall"] * confidence_shortfall
        + weights["low_contrast"] * low_contrast
    ) / sum(weights.values())


@torch.no_grad()
def _refresh_hard_positive_scores(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    amp_enabled: bool,
) -> dict[int, float]:
    """Mine low-confidence labelled RNV patches for replay in the next epoch."""
    sampler = getattr(train_loader, "batch_sampler", None)
    if not hasattr(sampler, "hard_positive_candidates"):
        return {}
    balanced_config = (
        config.get("data", {}).get("balanced_patch_sampling", {})
    )
    hard_config = balanced_config.get("hard_positive_mining", {})
    if not hard_config.get("enabled", False):
        return {}
    warmup = int(hard_config.get("warmup_epochs", 3))
    refresh_every = max(1, int(hard_config.get("refresh_every", 1)))
    if epoch < warmup or (epoch - warmup) % refresh_every:
        return {}
    refresh_index = (epoch - warmup) // refresh_every
    indices = sampler.hard_positive_candidates(
        int(hard_config.get("candidate_limit", 256)),
        refresh_index,
        float(hard_config.get("recheck_fraction", 0.5)),
    )
    if not indices:
        return {}

    dataset = train_loader.dataset
    was_training = bool(dataset.training)
    dataset.training = False
    scoring_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=max(1, int(hard_config.get("scoring_batch_size", 2))),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
    )
    target_names = list(config.get("data", {}).get("target_names", ["rnv"]))
    rnv_channel = target_names.index("rnv")
    modalities = list(config.get("data", {}).get("modalities", ["oct"]))
    oct_channel = modalities.index("oct")
    scores: dict[int, float] = {}
    model.eval()
    progress = tqdm(
        scoring_loader,
        total=len(scoring_loader),
        desc=f"Epoch {epoch} hard-positive mining",
        unit="batch",
        leave=False,
    )
    try:
        offset = 0
        for batch in progress:
            batch_indices = indices[offset : offset + int(batch["image"].shape[0])]
            offset += len(batch_indices)
            image = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            target = batch["target"][:, rnv_channel].to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                probabilities = torch.sigmoid(model(image).logits[:, rnv_channel])
            probabilities = probabilities.float().cpu()
            target = target.float().cpu()
            oct_images = image[:, oct_channel].float().cpu()
            for index, probability, reference, oct_image in zip(
                batch_indices,
                probabilities,
                target,
                oct_images,
                strict=True,
            ):
                score = _hard_positive_patch_score(
                    probability, reference, oct_image, hard_config
                )
                if score is None:
                    continue
                sample_type = dataset.patch_request_types[index]
                scan_group = dataset.patch_request_scan_groups[index]
                if sample_type == "positive_boundary":
                    score += float(hard_config.get("boundary_bonus", 0.10))
                if scan_group == "small_only":
                    score += float(hard_config.get("small_only_bonus", 0.15))
                scores[int(index)] = score
    finally:
        dataset.training = was_training
    sampler.update_hard_positive_scores(
        scores,
        indices,
        int(hard_config.get("maximum_pool_size", 256)),
        score_decay=float(hard_config.get("score_decay", 0.9)),
        minimum_retained_score=float(
            hard_config.get("minimum_retained_score", 0.15)
        ),
        maximum_per_case=int(hard_config.get("maximum_per_case", 12)),
        maximum_per_lesion=int(hard_config.get("maximum_per_lesion", 3)),
    )
    return scores


@torch.no_grad()
def _refresh_hard_negative_scores(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    amp_enabled: bool,
) -> dict[int, float]:
    sampler = getattr(train_loader, "batch_sampler", None)
    if not hasattr(sampler, "hard_negative_candidates"):
        return {}
    hard_config = (
        config.get("data", {})
        .get("balanced_patch_sampling", {})
        .get("hard_negative_mining", {})
    )
    if not hard_config.get("enabled", False):
        return {}
    warmup = int(hard_config.get("warmup_epochs", 10))
    refresh_every = max(1, int(hard_config.get("refresh_every", 5)))
    if epoch < warmup or (epoch - warmup) % refresh_every:
        return {}
    refresh_index = (epoch - warmup) // refresh_every
    indices = sampler.hard_negative_candidates(
        int(hard_config.get("candidate_limit", 256)),
        refresh_index,
        float(hard_config.get("recheck_fraction", 0.25)),
    )
    if not indices:
        return {}
    dataset = train_loader.dataset
    was_training = bool(dataset.training)
    dataset.training = False
    scoring_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=max(1, int(hard_config.get("scoring_batch_size", 2))),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(device.type == "cuda"),
    )
    model.eval()
    scores: dict[int, float] = {}
    probability_threshold = float(hard_config.get("probability_threshold", 0.5))
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("hard_negative_mining.probability_threshold must be in [0, 1]")
    use_inference_postprocessing = bool(
        hard_config.get("use_inference_postprocessing", True)
    )
    progress = tqdm(
        scoring_loader,
        total=len(scoring_loader),
        desc=f"Epoch {epoch} hard-negative mining",
        unit="batch",
        leave=False,
    )
    try:
        offset = 0
        for batch in progress:
            batch_indices = indices[offset : offset + int(batch["image"].shape[0])]
            offset += len(batch_indices)
            image = batch["image"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            target = batch["target"][:, 0]
            if torch.count_nonzero(target).item():
                raise ValueError(
                    "Hard-negative mining received a candidate containing RNV voxels"
                )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                probabilities = torch.sigmoid(model(image).logits[:, 0])
            probabilities = probabilities.float().cpu()
            for index, probability in zip(
                batch_indices, probabilities, strict=True
            ):
                raw_mask = (probability >= probability_threshold).numpy()
                if use_inference_postprocessing:
                    retained_mask = postprocess_prediction_masks(
                        {"rnv": raw_mask}, config.get("inference", {})
                    )["rnv"].astype(bool)
                else:
                    retained_mask = raw_mask
                score = _false_positive_patch_score(
                    probability, retained_mask, hard_config
                )
                if score is not None:
                    scores[int(index)] = score
    finally:
        dataset.training = was_training
    if hasattr(sampler, "update_hard_negative_scores"):
        sampler.update_hard_negative_scores(
            scores,
            indices,
            int(hard_config.get("maximum_pool_size", 512)),
            score_decay=float(hard_config.get("score_decay", 0.9)),
            minimum_retained_score=float(
                hard_config.get("minimum_retained_score", 0.35)
            ),
            maximum_per_case=int(hard_config.get("maximum_per_case", 12)),
        )
    else:
        sampler.set_hard_negative_scores(scores)
    return scores


@torch.no_grad()
def _refresh_full_volume_hard_negative_scores(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    amp_enabled: bool,
) -> dict[int, float]:
    """Mine persistent false positives from complete labelled training scans.

    Validation scans are deliberately inaccessible here. False-positive components
    found with test-time sliding windows are mapped back to the already indexed
    pure-negative training patches, so no new crop can accidentally contain RNV.
    """
    sampler = getattr(train_loader, "batch_sampler", None)
    if not hasattr(sampler, "update_hard_negative_scores"):
        return {}
    hard_config = (
        config.get("data", {})
        .get("balanced_patch_sampling", {})
        .get("hard_negative_mining", {})
    )
    full_config = hard_config.get("full_volume_refresh", {})
    if not full_config.get("enabled", False):
        return {}
    warmup = int(full_config.get("warmup_epochs", 8))
    refresh_every = max(1, int(full_config.get("refresh_every", 4)))
    if epoch < warmup or (epoch - warmup) % refresh_every:
        return {}

    dataset = train_loader.dataset
    records = getattr(dataset, "records", None)
    patch_requests = getattr(dataset, "patch_requests", None)
    if not records or patch_requests is None:
        return {}
    by_case: dict[int, list[int]] = {}
    for index in sampler.all_negative_indices:
        case_index = int(patch_requests[index][0])
        by_case.setdefault(case_index, []).append(int(index))
    if not by_case:
        return {}

    cases = sorted(by_case)
    cases_per_refresh = max(1, int(full_config.get("cases_per_refresh", 2)))
    cases_per_refresh = min(cases_per_refresh, len(cases))
    fixed_rng = np.random.default_rng(int(config.get("seed", 0)) + 104729)
    fixed_rng.shuffle(cases)
    refresh_index = (epoch - warmup) // refresh_every
    start = (refresh_index * cases_per_refresh) % len(cases)
    selected_cases = [
        cases[(start + offset) % len(cases)]
        for offset in range(cases_per_refresh)
    ]

    probability_threshold = float(
        full_config.get(
            "probability_threshold",
            config.get("inference", {}).get("threshold", 0.45),
        )
    )
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError(
            "hard_negative_mining.full_volume_refresh.probability_threshold "
            "must be in [0, 1]"
        )
    patch_size = tuple(int(value) for value in dataset.patch_size)
    inference_config = config.get("inference", {})
    inference_patch_size = inference_config.get(
        "patch_size", config["data"]["patch_size"]
    )
    overlap = float(inference_config.get("sliding_window_overlap", 0.5))
    scores: dict[int, float] = {}
    evaluated_indices: list[int] = []
    was_training = bool(dataset.training)
    dataset.training = False
    model.eval()
    progress = tqdm(
        selected_cases,
        desc=f"Epoch {epoch} full-volume hard-negative mining",
        unit="scan",
        leave=False,
    )
    try:
        for case_index in progress:
            image, target = load_full_case(records[case_index], config["data"])
            if target is None:
                raise ValueError(
                    "Full-volume hard-negative mining requires training labels"
                )
            logits = sliding_window_logits(
                model,
                image,
                inference_patch_size,
                overlap,
                amp=amp_enabled,
            )
            probability = torch.sigmoid(logits)[0, 0].float().cpu()
            raw_mask = (probability >= probability_threshold).numpy()
            predicted = postprocess_prediction_masks(
                {"rnv": raw_mask}, inference_config
            )["rnv"].astype(bool)
            reference = target[0].numpy() > 0.5
            false_positive_mask = np.logical_and(predicted, ~reference)
            for index in by_case[case_index]:
                indexed_start = patch_requests[index][1]
                if indexed_start is None:
                    continue
                slices = tuple(
                    slice(int(begin), int(begin) + size)
                    for begin, size in zip(
                        indexed_start, patch_size, strict=True
                    )
                )
                evaluated_indices.append(index)
                score = _false_positive_patch_score(
                    probability[slices],
                    false_positive_mask[slices],
                    hard_config,
                )
                if score is not None:
                    scores[index] = score
    finally:
        dataset.training = was_training

    sampler.update_hard_negative_scores(
        scores,
        evaluated_indices,
        int(hard_config.get("maximum_pool_size", 512)),
        score_decay=float(hard_config.get("score_decay", 0.9)),
        minimum_retained_score=float(
            hard_config.get("minimum_retained_score", 0.35)
        ),
        maximum_per_case=int(hard_config.get("maximum_per_case", 12)),
    )
    return scores


def train(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict[str, Any],
    run_dir: str | Path,
    device: torch.device,
    resume_payload: dict[str, Any] | None = None,
    full_volume_evaluator: Callable[
        [torch.nn.Module, int], dict[str, float]
    ]
    | None = None,
) -> dict[str, Any]:
    training = config["training"]
    epochs = int(training["epochs"])
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    threshold = float(config.get("inference", {}).get("threshold", 0.45))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metric_name = training.get("checkpoint_metric", "dice")
    mode = training.get("checkpoint_mode", "max")
    best = -math.inf if mode == "max" else math.inf
    best_epoch = 0
    dice_at_best_loss = float("nan")
    full_volume_config = training.get("full_volume_checkpoint", {})
    full_volume_enabled = bool(full_volume_config.get("enabled", False))
    full_volume_metric_name = str(
        full_volume_config.get("preferred_metric", "validation_loss")
    )
    full_volume_mode = str(full_volume_config.get("preferred_mode", "min"))
    best_full_volume = (
        -math.inf if full_volume_mode == "max" else math.inf
    )
    best_full_volume_epoch = 0
    best_full_volume_dice = float("nan")
    checkpoint_min_delta = float(training.get("checkpoint_min_delta", 0.0))
    preferred_min_delta = float(
        full_volume_config.get("preferred_min_delta", 0.0)
    )
    patience = int(training.get("early_stopping_patience", epochs))
    without_improvement = 0
    run_dir = Path(run_dir)
    start_epoch = int(resume_payload.get("epoch", 0)) if resume_payload else 0
    history_path = run_dir / "logs" / "metrics.jsonl"
    scheduler_monitor_history: list[float] = []

    def read_history() -> list[dict[str, Any]]:
        if not history_path.is_file():
            return []
        return [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if resume_payload and "scaler_state" in resume_payload:
        scaler.load_state_dict(resume_payload["scaler_state"])
    if resume_payload:
        saved_state = resume_payload.get("training_state", {})
        if saved_state:
            best = float(saved_state.get("best", best))
            best_epoch = int(saved_state.get("best_epoch", 0))
            dice_at_best_loss = float(
                saved_state.get(
                    "dice_at_best_loss",
                    saved_state.get("best_dice", dice_at_best_loss),
                )
            )
            best_full_volume = float(
                saved_state.get(
                    "best_preferred_metric",
                    saved_state.get("best_full_volume", best_full_volume),
                )
            )
            best_full_volume_epoch = int(
                saved_state.get(
                    "best_dice_epoch",
                    saved_state.get("best_full_volume_epoch", 0),
                )
            )
            best_full_volume_dice = float(
                saved_state.get("best_full_volume_dice", best_full_volume_dice)
            )
            without_improvement = int(saved_state.get("without_improvement", 0))
        else:
            history = read_history()
            candidates = [
                row for row in history if metric_name in row and int(row["epoch"]) <= start_epoch
            ]
            if candidates:
                selected = (
                    max(candidates, key=lambda row: float(row[metric_name]))
                    if mode == "max"
                    else min(candidates, key=lambda row: float(row[metric_name]))
                )
                best = float(selected[metric_name])
                best_epoch = int(selected["epoch"])
                dice_at_best_loss = float(
                    selected.get("val_dice", dice_at_best_loss)
                )
                without_improvement = max(0, start_epoch - best_epoch)
        if best_epoch and not math.isfinite(dice_at_best_loss):
            matching_rows = [
                row
                for row in read_history()
                if int(row.get("epoch", -1)) == best_epoch
            ]
            if matching_rows:
                historical = matching_rows[-1]
                historical_dice = historical.get(
                    "val_dice",
                    historical.get("rnv_dice", historical.get("dice")),
                )
                if historical_dice is not None and math.isfinite(
                    float(historical_dice)
                ):
                    dice_at_best_loss = float(historical_dice)
            if (
                not math.isfinite(dice_at_best_loss)
                and metric_name in {"dice", "rnv_dice"}
                and math.isfinite(best)
            ):
                dice_at_best_loss = best
        sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_hard_negative_scores"):
            sampler.set_hard_negative_scores(
                saved_state.get("hard_negative_scores", {})
            )
        if hasattr(sampler, "set_hard_positive_scores"):
            sampler.set_hard_positive_scores(
                saved_state.get("hard_positive_scores", {})
            )
        monitor_name = str(
            config.get("scheduler", {}).get("monitor", "validation_loss")
        )
        log_name = "val_loss" if monitor_name == "validation_loss" else monitor_name
        scheduler_monitor_history = [
            float(row[log_name])
            for row in read_history()
            if int(row.get("epoch", 0)) <= start_epoch
            and row.get(log_name) is not None
            and math.isfinite(float(row[log_name]))
        ]
    if full_volume_enabled and full_volume_evaluator is None:
        raise ValueError(
            "training.full_volume_checkpoint.enabled requires a full-volume evaluator"
        )
    if full_volume_mode not in {"min", "max"}:
        raise ValueError(
            "training.full_volume_checkpoint.preferred_mode must be 'min' or 'max'"
        )
    if not full_volume_enabled and validation_loader is None:
        raise ValueError("Patch validation requires a validation loader")
    model.to(device=device, dtype=torch.float32)
    validation_description = (
        "full-volume validation"
        if full_volume_enabled
        else f"{len(validation_loader)} validation batches"
    )
    print(
        f"{'Resuming' if start_epoch else 'Training'} on {device}: "
        f"epochs {start_epoch + 1}-{epochs}, "
        f"{len(train_loader)} train batches and "
        f"{validation_description} per epoch",
        flush=True,
    )
    if start_epoch >= epochs:
        raise ValueError(
            f"Checkpoint epoch {start_epoch} has already reached configured epochs={epochs}"
        )
    epoch_progress = tqdm(
        range(start_epoch + 1, epochs + 1), desc="Training", unit="epoch"
    )
    for epoch in epoch_progress:
        model.train()
        train_batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        train_intersection = 0
        train_denominator = 0
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} train",
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(train_progress, start=1):
            image = batch["image"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            target = batch["target"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(image, target)
                loss = criterion(output.logits, target)
                loss = loss + sum(output.auxiliary_losses.values())
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                clip = float(training.get("gradient_clip_norm", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
            with torch.no_grad():
                prediction = torch.sigmoid(output.logits[:, 0]) >= threshold
                reference = target[:, 0] > 0.5
                train_intersection += int(torch.count_nonzero(prediction & reference))
                train_denominator += int(torch.count_nonzero(prediction)) + int(
                    torch.count_nonzero(reference)
                )
            running_train_dice = (2.0 * train_intersection + 1.0e-6) / (
                train_denominator + 1.0e-6
            )
            train_progress.set_postfix(
                loss=f"{total_loss / step:.4f}",
                dice=f"{running_train_dice:.4f}",
            )
        if full_volume_enabled:
            assert full_volume_evaluator is not None
            validation_metrics = full_volume_evaluator(model, epoch)
        else:
            validation_metrics = evaluate_loader(
                model,
                validation_loader,
                device,
                config,
                criterion=criterion,
                progress=True,
            )
        if scheduler is not None:
            scheduler_config = config.get("scheduler", {})
            scheduler_name = str(scheduler_config.get("name", "none")).lower()
            if scheduler_name == "reduce_on_plateau":
                monitor_name = str(scheduler_config.get("monitor", "validation_loss"))
                monitor_value = validation_metrics.get(monitor_name)
                if monitor_value is None or not math.isfinite(float(monitor_value)):
                    available = ", ".join(sorted(validation_metrics))
                    raise KeyError(
                        f"Scheduler monitor {monitor_name!r} is unavailable or non-finite; "
                        f"available validation metrics: {available}"
                    )
                scheduler_monitor_history.append(float(monitor_value))
                smoothing_window = max(
                    1, int(scheduler_config.get("smoothing_window", 1))
                )
                smoothed_monitor = float(
                    np.mean(scheduler_monitor_history[-smoothing_window:])
                )
                scheduler.step(smoothed_monitor)
            else:
                scheduler.step()
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        _refresh_hard_negative_scores(
            model,
            train_loader,
            device,
            config,
            epoch,
            amp_enabled,
        )
        _refresh_hard_positive_scores(
            model,
            train_loader,
            device,
            config,
            epoch,
            amp_enabled,
        )
        _refresh_full_volume_hard_negative_scores(
            model,
            train_loader,
            device,
            config,
            epoch,
            amp_enabled,
        )
        train_loss = total_loss / max(len(train_loader), 1)
        train_dice = (2.0 * train_intersection + 1.0e-6) / (
            train_denominator + 1.0e-6
        )
        score = validation_metrics.get(metric_name)
        if score is None:
            available = ", ".join(sorted(validation_metrics))
            raise KeyError(
                f"checkpoint_metric {metric_name!r} is unavailable; choose one of: {available}"
            )
        validation_dice = validation_metrics.get(
            "dice", validation_metrics.get("rnv_dice", float("nan"))
        )
        positive_dice = validation_metrics.get(
            "positive_dice",
            validation_metrics.get("rnv_positive_dice"),
        )
        negative_patch_clean_rate = validation_metrics.get(
            "negative_patch_clean_rate",
            validation_metrics.get("rnv_negative_patch_clean_rate"),
        )
        improved = score is not None and math.isfinite(float(score)) and (
            (mode == "max" and score > best + checkpoint_min_delta)
            or (mode == "min" and score < best - checkpoint_min_delta)
        )
        if improved:
            best = float(score)
            best_epoch = epoch
            dice_at_best_loss = float(validation_dice)
            without_improvement = 0
        else:
            without_improvement += 1
        will_stop = without_improvement >= patience
        full_volume_metrics: dict[str, float] | None = None
        full_volume_improved = False
        evaluate_full_volume = full_volume_enabled
        if evaluate_full_volume:
            full_volume_metrics = validation_metrics
            resolved_full_metric_name = full_volume_metric_name
            if (
                resolved_full_metric_name == "rnv_dice"
                and resolved_full_metric_name not in full_volume_metrics
                and "dice" in full_volume_metrics
            ):
                resolved_full_metric_name = "dice"
            full_volume_score = full_volume_metrics.get(resolved_full_metric_name)
            if full_volume_score is None or not math.isfinite(
                float(full_volume_score)
            ):
                available = ", ".join(sorted(full_volume_metrics))
                raise KeyError(
                    f"Full-volume checkpoint metric {full_volume_metric_name!r} "
                    f"is unavailable or non-finite; available metrics: {available}"
                )
            full_volume_dice = full_volume_metrics.get(
                "rnv_dice", full_volume_metrics.get("dice", float("nan"))
            )
            full_volume_improved = (
                full_volume_mode == "max"
                and full_volume_score > best_full_volume + preferred_min_delta
            ) or (
                full_volume_mode == "min"
                and full_volume_score < best_full_volume - preferred_min_delta
            )
            if full_volume_improved:
                best_full_volume = float(full_volume_score)
                best_full_volume_epoch = epoch
                best_full_volume_dice = float(full_volume_dice)
            append_jsonl(
                run_dir / "logs" / "full_volume_metrics.jsonl",
                {
                    "epoch": epoch,
                    "validation_loss": float(
                        full_volume_metrics["validation_loss"]
                    ),
                    "full_volume_dice": float(full_volume_dice),
                    "stratified_rnv_score": full_volume_metrics.get(
                        "stratified_rnv_score"
                    ),
                    "small_only_dice": full_volume_metrics.get(
                        "small_only_dice",
                        full_volume_metrics.get("rnv_small_only_dice"),
                    ),
                    "large_containing_dice": full_volume_metrics.get(
                        "large_containing_dice",
                        full_volume_metrics.get("rnv_large_containing_dice"),
                    ),
                    "is_best_loss": improved,
                    "is_best_dice": full_volume_improved,
                    "best_loss_epoch": best_epoch,
                    "best_dice_epoch": best_full_volume_epoch,
                    "best_dice": best_full_volume_dice,
                },
            )
        preferred_improved = full_volume_improved if full_volume_enabled else improved
        preferred_epoch = best_full_volume_epoch if full_volume_enabled else best_epoch
        preferred_dice = (
            best_full_volume_dice if full_volume_enabled else dice_at_best_loss
        )
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": validation_metrics.get("validation_loss", float("nan")),
            "val_dice": float(validation_dice),
            "positive_dice": None if positive_dice is None else float(positive_dice),
            "small_only_dice": validation_metrics.get(
                "small_only_dice", validation_metrics.get("rnv_small_only_dice")
            ),
            "large_containing_dice": validation_metrics.get(
                "large_containing_dice",
                validation_metrics.get("rnv_large_containing_dice"),
            ),
            "stratified_rnv_score": validation_metrics.get(
                "stratified_rnv_score"
            ),
            "negative_patch_clean_rate": (
                None
                if negative_patch_clean_rate is None
                else float(negative_patch_clean_rate)
            ),
            "lr": current_learning_rate,
            "is_best": preferred_improved,
            "is_best_loss": improved,
            "is_best_dice": preferred_improved,
            "best_epoch": preferred_epoch,
            "best_loss_epoch": best_epoch,
            "best_dice": preferred_dice if math.isfinite(preferred_dice) else None,
        }
        summary_parts = [
            f"train_loss={train_loss:.4f}",
            f"train_dice={train_dice:.4f}",
            f"val_loss={log_row['val_loss']:.4f}",
            f"val_dice={log_row['val_dice']:.4f}",
            f"lr={current_learning_rate:.3e}",
            f"is_best_loss={improved}",
            f"is_best_dice={preferred_improved}",
        ]
        append_jsonl(run_dir / "logs" / "metrics.jsonl", log_row)
        last_path = run_dir / "checkpoints" / "last.pt"
        best_path = run_dir / "checkpoints" / "best.pt"
        best_loss_path = run_dir / "checkpoints" / "best_loss.pt"
        best_dice_path = run_dir / "checkpoints" / "best_dice.pt"
        best_full_volume_path = (
            run_dir / "checkpoints" / "best_full_volume.pt"
        )
        training_state = {
            "best": best,
            "best_epoch": best_epoch,
            "dice_at_best_loss": dice_at_best_loss,
            "best_dice": preferred_dice,
            "best_dice_epoch": preferred_epoch,
            "best_preferred_metric": best_full_volume,
            "best_full_volume": best_full_volume,
            "best_full_volume_epoch": best_full_volume_epoch,
            "best_full_volume_dice": best_full_volume_dice,
            "without_improvement": without_improvement,
            "hard_negative_scores": getattr(
                getattr(train_loader, "batch_sampler", None),
                "hard_negative_scores",
                {},
            ),
            "hard_positive_scores": getattr(
                getattr(train_loader, "batch_sampler", None),
                "hard_positive_scores",
                {},
            ),
        }
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            validation_metrics,
            config,
            scheduler=scheduler,
            scaler=scaler,
            training_state=training_state,
        )
        if improved:
            loss_destination = best_loss_path if full_volume_enabled else best_path
            save_checkpoint(
                loss_destination,
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
                scheduler=scheduler,
                scaler=scaler,
                training_state=training_state,
            )
        if full_volume_improved:
            assert full_volume_metrics is not None
            save_checkpoint(
                best_dice_path,
                model,
                optimizer,
                epoch,
                full_volume_metrics,
                config,
                scheduler=scheduler,
                scaler=scaler,
                training_state=training_state,
            )
            _replace_checkpoint_alias(best_dice_path, best_full_volume_path)
            _replace_checkpoint_alias(best_dice_path, best_path)
            tqdm.write(
                "[BEST LOSS MODEL] "
                f"Epoch {epoch}/{epochs}: validation_loss={best_full_volume:.4f}, "
                f"dice={best_full_volume_dice:.4f}"
            )
        if improved:
            tqdm.write(
                "[BEST LOSS] "
                f"Epoch {epoch}/{epochs}: validation_loss={best:.4f}"
            )
        status_label = "[BEST MODEL]" if preferred_improved else "[NOT BEST]"
        tqdm.write(
            f"{status_label} Epoch {epoch}/{epochs}: " + ", ".join(summary_parts)
        )
        if will_stop:
            tqdm.write(
                f"Early stopping at epoch {epoch}: no {metric_name} improvement for "
                f"{without_improvement} epochs; best epoch={best_epoch}, best {metric_name}={best:.4f}"
            )
            break
    return {
        "checkpoint_metric": metric_name,
        "best_epoch_metric": best,
        "best_epoch": float(preferred_epoch),
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "best_loss_checkpoint": (
            str(run_dir / "checkpoints" / "best_loss.pt")
            if full_volume_enabled and best_epoch
            else None
        ),
        "best_loss_epoch": float(best_epoch),
        "best_validation_loss": best,
        "dice_at_best_loss": dice_at_best_loss,
        "best_dice_checkpoint": (
            str(run_dir / "checkpoints" / "best_dice.pt")
            if full_volume_enabled and best_full_volume_epoch
            else None
        ),
        "best_dice_epoch": (
            float(best_full_volume_epoch) if best_full_volume_epoch else None
        ),
        "best_dice": (
            best_full_volume_dice if best_full_volume_epoch else None
        ),
        "best_patch_checkpoint": (
            None
            if full_volume_enabled
            else str(run_dir / "checkpoints" / "best.pt")
        ),
        "best_full_volume_checkpoint": (
            str(run_dir / "checkpoints" / "best_full_volume.pt")
            if best_full_volume_epoch
            else None
        ),
        "best_full_volume_epoch": (
            float(best_full_volume_epoch) if best_full_volume_epoch else None
        ),
        "best_full_volume_dice": (
            best_full_volume_dice if best_full_volume_epoch else None
        ),
        "last_checkpoint": str(run_dir / "checkpoints" / "last.pt"),
        "epochs_completed": float(epoch),
    }
