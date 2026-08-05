from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
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
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * target).sum(dim=dimensions)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return (1.0 - score).mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.components = dict(config.get("components", {"dice": 1.0, "bce": 1.0}))
        self.smooth = float(config.get("smooth", 1.0e-6))
        self.focal_alpha = float(config.get("focal_alpha", 0.25))
        self.focal_gamma = float(config.get("focal_gamma", 2.0))
        self.tversky_alpha = float(config.get("tversky_alpha", 0.3))
        self.tversky_beta = float(config.get("tversky_beta", 0.7))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses: dict[str, torch.Tensor] = {
            "dice": dice_loss(logits, target, self.smooth),
            "bce": F.binary_cross_entropy_with_logits(logits, target),
            "focal": focal_loss(logits, target, self.focal_alpha, self.focal_gamma),
            "tversky": tversky_loss(
                logits, target, self.tversky_alpha, self.tversky_beta, self.smooth
            ),
        }
        unknown = set(self.components) - set(losses)
        if unknown:
            raise ValueError(f"Unknown loss components: {sorted(unknown)}")
        total = logits.new_tensor(0.0)
        for name, weight in self.components.items():
            total = total + float(weight) * losses[name]
        return total


def build_loss(config: dict[str, Any]) -> CombinedSegmentationLoss:
    return CombinedSegmentationLoss(config)

