from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from rnv_seg.metrics import lesion_metrics, surface_metrics, volume_metrics, voxel_metrics


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    rows: defaultdict[str, list[float]] = defaultdict(list)
    threshold = float(config["inference"].get("threshold", 0.5))
    metric_config = config["metrics"]
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = torch.sigmoid(model(image).logits) >= threshold
        for index in range(image.shape[0]):
            predicted_np = prediction[index, 0].cpu().numpy()
            target_np = target[index, 0].cpu().numpy() > 0.5
            metrics = voxel_metrics(predicted_np, target_np)
            metrics.update(
                volume_metrics(
                    predicted_np,
                    target_np,
                    metric_config["volume"].get("voxel_spacing_mm", [1.0, 1.0, 1.0]),
                )
            )
            lesion_config = metric_config.get("lesion", {})
            if lesion_config.get("enabled", True):
                metrics.update(
                    lesion_metrics(
                        predicted_np,
                        target_np,
                        connectivity=int(lesion_config.get("connectivity", 26)),
                        minimum_voxels=int(lesion_config.get("minimum_voxels", 3)),
                        match_iou=float(lesion_config.get("match_iou", 0.1)),
                    )
                )
            surface_config = metric_config.get("surface", {})
            if surface_config.get("enabled", False):
                metrics.update(
                    surface_metrics(
                        predicted_np,
                        target_np,
                        metric_config["volume"].get("voxel_spacing_mm", [1.0, 1.0, 1.0]),
                        float(surface_config.get("tolerance_mm", 0.05)),
                    )
                )
            for name, value in metrics.items():
                if np.isfinite(value):
                    rows[name].append(float(value))
    return {name: float(np.mean(values)) for name, values in rows.items() if values}

