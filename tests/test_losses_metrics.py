import numpy as np
import torch

from rnv_seg.losses import build_loss
from rnv_seg.metrics import lesion_metrics, volume_metrics, voxel_metrics


def test_combined_losses_are_finite() -> None:
    logits = torch.zeros(2, 1, 8, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 2:5, 2:5, 2:5] = 1
    for name in ("dice", "bce", "focal", "tversky"):
        loss = build_loss({"components": {name: 1.0}})(logits, target)
        assert torch.isfinite(loss)


def test_perfect_metrics() -> None:
    target = np.zeros((8, 8, 8), dtype=np.uint8)
    target[2:5, 2:5, 2:5] = 1
    assert voxel_metrics(target, target)["dice"] == 1.0
    assert lesion_metrics(target, target)["lesion_f1"] == 1.0
    assert volume_metrics(target, target)["absolute_volume_error_mm3"] == 0.0


def test_empty_metrics_have_explicit_behavior() -> None:
    empty = np.zeros((8, 8, 8), dtype=np.uint8)
    assert voxel_metrics(empty, empty)["dice"] == 1.0
    assert lesion_metrics(empty, empty)["lesion_sensitivity"] == 1.0

