from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from rnv_seg.data.dataset import target_names_from_config
from rnv_seg.metrics import lesion_metrics, surface_metrics, volume_metrics, voxel_metrics


def evaluate_binary_case(
    prediction: np.ndarray,
    target: np.ndarray,
    metric_config: dict[str, Any],
) -> dict[str, float]:
    metrics = voxel_metrics(prediction, target)
    metrics.update(
        volume_metrics(
            prediction,
            target,
            metric_config["volume"].get("voxel_spacing_mm", [1.0, 1.0, 1.0]),
        )
    )
    lesion_config = metric_config.get("lesion", {})
    if lesion_config.get("enabled", True):
        metrics.update(
            lesion_metrics(
                prediction,
                target,
                connectivity=int(lesion_config.get("connectivity", 26)),
                minimum_voxels=int(lesion_config.get("minimum_voxels", 3)),
                match_iou=float(lesion_config.get("match_iou", 0.1)),
            )
        )
    surface_config = metric_config.get("surface", {})
    if surface_config.get("enabled", False):
        metrics.update(
            surface_metrics(
                prediction,
                target,
                metric_config["volume"].get("voxel_spacing_mm", [1.0, 1.0, 1.0]),
                float(surface_config.get("tolerance_mm", 0.05)),
            )
        )
    return metrics


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict[str, Any],
    *,
    criterion: torch.nn.Module | None = None,
    progress: bool = False,
) -> dict[str, float]:
    model.eval()
    rows: defaultdict[str, list[float]] = defaultdict(list)
    validation_losses: list[float] = []
    threshold = float(config["inference"].get("threshold", 0.45))
    metric_config = config["metrics"]
    amp_enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    target_names = target_names_from_config(config.get("data", {}))
    progress_bar = tqdm(loader, desc="Validation", unit="batch", leave=False) if progress else None
    batches = progress_bar if progress_bar is not None else loader
    for batch in batches:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(image, target)
            if criterion is not None:
                loss = criterion(output.logits, target)
                loss = loss + sum(output.auxiliary_losses.values())
                validation_losses.append(float(loss))
        prediction = torch.sigmoid(output.logits) >= threshold
        probabilities = torch.sigmoid(output.logits)
        for index in range(image.shape[0]):
            for channel, task_name in enumerate(target_names):
                predicted_np = prediction[index, channel].cpu().numpy()
                probability_np = probabilities[index, channel].float().cpu().numpy()
                target_np = target[index, channel].cpu().numpy() > 0.5
                metrics = evaluate_binary_case(predicted_np, target_np, metric_config)
                for name, value in metrics.items():
                    key = name if len(target_names) == 1 else f"{task_name}_{name}"
                    if np.isfinite(value):
                        rows[key].append(float(value))
                    if task_name == "rnv" and target_np.any() and np.isfinite(value):
                        positive_key = (
                            f"positive_{name}"
                            if len(target_names) == 1
                            else f"rnv_positive_{name}"
                        )
                        rows[positive_key].append(float(value))
                if task_name == "rnv" and not target_np.any():
                    prefix = "" if len(target_names) == 1 else "rnv_"
                    false_positive_voxels = float(predicted_np.sum())
                    rows[f"{prefix}negative_false_positive_voxels"].append(
                        false_positive_voxels
                    )
                    rows[f"{prefix}negative_false_positive_fraction"].append(
                        float(predicted_np.mean())
                    )
                    rows[f"{prefix}negative_max_probability"].append(
                        float(probability_np.max())
                    )
                    rows[f"{prefix}negative_patch_clean_rate"].append(
                        float(false_positive_voxels == 0)
                    )
        if progress_bar is not None:
            dice_key = "dice" if len(target_names) == 1 else "rnv_dice"
            status = {}
            if rows[dice_key]:
                status["val_dice"] = f"{np.mean(rows[dice_key]):.4f}"
            if validation_losses:
                status["val_loss"] = f"{np.mean(validation_losses):.4f}"
            progress_bar.set_postfix(status)
    results = {name: float(np.mean(values)) for name, values in rows.items() if values}
    if len(target_names) > 1 and {"rnv_dice", "vasculature_dice"} <= results.keys():
        results["multitask_score"] = 0.7 * results["rnv_dice"] + 0.3 * results["vasculature_dice"]
    if validation_losses:
        results["validation_loss"] = float(np.mean(validation_losses))
    return results
