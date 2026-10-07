from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    dimensions = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * target).sum(dim=dimensions)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return (1.0 - score).mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1.0e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    dimensions = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * target).sum(dim=dimensions)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return (1.0 - score).pow(gamma).mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.components = dict(config.get("components", {"dice": 1.0, "bce": 1.0}))
        self.smooth = float(config.get("smooth", 1.0e-6))
        self.focal_alpha = float(config.get("focal_alpha", 0.25))
        self.focal_gamma = float(config.get("focal_gamma", 2.0))
        self.tversky_alpha = float(config.get("tversky_alpha", 0.3))
        self.tversky_beta = float(config.get("tversky_beta", 0.7))
        self.focal_tversky_alpha = float(config.get("focal_tversky_alpha", 0.3))
        self.focal_tversky_beta = float(config.get("focal_tversky_beta", 0.7))
        self.focal_tversky_gamma = float(config.get("focal_tversky_gamma", 0.75))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses: dict[str, torch.Tensor] = {
            "dice": dice_loss(logits, target, self.smooth),
            "bce": F.binary_cross_entropy_with_logits(
                logits.float(), target.float()
            ),
            "focal": focal_loss(logits, target, self.focal_alpha, self.focal_gamma),
            "tversky": tversky_loss(
                logits, target, self.tversky_alpha, self.tversky_beta, self.smooth
            ),
            "focal_tversky": focal_tversky_loss(
                logits,
                target,
                self.focal_tversky_alpha,
                self.focal_tversky_beta,
                self.focal_tversky_gamma,
                self.smooth,
            ),
        }
        unknown = set(self.components) - set(losses)
        if unknown:
            raise ValueError(f"Unknown loss components: {sorted(unknown)}")
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        for name, weight in self.components.items():
            total = total + float(weight) * losses[name]
        return total


class MultiTaskSegmentationLoss(nn.Module):
    """Weighted binary losses plus containment consistency for overlapping tasks."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        tasks = config.get("tasks", {})
        if len(tasks) < 2:
            raise ValueError("Multitask loss requires at least two configured tasks")
        shared = {
            key: value
            for key, value in config.items()
            if key not in {"tasks", "consistency_weight", "consistency_parent", "consistency_child"}
        }
        self.task_names = tuple(tasks)
        self.weights: dict[str, float] = {}
        self.losses = nn.ModuleDict()
        for name, task in tasks.items():
            task_config = dict(shared)
            task_config.update({key: value for key, value in task.items() if key != "weight"})
            self.weights[name] = float(task.get("weight", 1.0))
            self.losses[name] = CombinedSegmentationLoss(task_config)
        self.consistency_weight = float(config.get("consistency_weight", 0.0))
        self.parent = str(config.get("consistency_parent", "rnv"))
        self.child = str(config.get("consistency_child", "vasculature"))
        if self.consistency_weight and ({self.parent, self.child} - set(self.task_names)):
            raise ValueError("Consistency parent and child must be configured multitask targets")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape or logits.shape[1] != len(self.task_names):
            raise ValueError(
                f"Expected logits and target shaped [N,{len(self.task_names)},D,H,W], "
                f"got {tuple(logits.shape)} and {tuple(target.shape)}"
            )
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        for channel, name in enumerate(self.task_names):
            total = total + self.weights[name] * self.losses[name](
                logits[:, channel : channel + 1], target[:, channel : channel + 1]
            )
        if self.consistency_weight:
            parent = torch.sigmoid(
                logits[:, self.task_names.index(self.parent)].float()
            )
            child = torch.sigmoid(
                logits[:, self.task_names.index(self.child)].float()
            )
            total = total + self.consistency_weight * F.relu(child - parent).mean()
        return total


def build_loss(config: dict[str, Any]) -> nn.Module:
    if config.get("tasks"):
        return MultiTaskSegmentationLoss(config)
    return CombinedSegmentationLoss(config)
