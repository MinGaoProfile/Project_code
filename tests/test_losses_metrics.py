import numpy as np
import torch

from rnv_seg.losses import build_loss
from rnv_seg.metrics import lesion_metrics, volume_metrics, voxel_metrics


def test_combined_losses_are_finite() -> None:
    logits = torch.zeros(2, 1, 8, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 2:5, 2:5, 2:5] = 1
    for name in ("dice", "bce", "focal", "tversky", "focal_tversky"):
        loss = build_loss({"components": {name: 1.0}})(logits, target)
        assert torch.isfinite(loss)


def test_amp_sized_half_logits_use_float32_loss_reductions() -> None:
    # 131,072 voxels would overflow a direct float16 probability sum near 0.5.
    logits = torch.zeros(1, 1, 64, 64, 32, dtype=torch.float16, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 16:48, 16:48, 8:24] = 1
    criterion = build_loss(
        {
            "components": {
                "dice": 1.0,
                "bce": 1.0,
                "focal_tversky": 1.0,
            }
        }
    )

    loss = criterion(logits, target)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_multitask_loss_is_finite_and_enforces_containment() -> None:
    target = torch.zeros(1, 2, 4, 4, 4)
    target[:, 0, 1:3, 1:3, 1:3] = 1
    target[:, 1, 2, 2, 2] = 1
    criterion = build_loss(
        {
            "tasks": {
                "rnv": {"components": {"dice": 1.0}},
                "vasculature": {"weight": 0.5, "components": {"focal_tversky": 1.0}},
            },
            "consistency_weight": 0.1,
        }
    )
    logits = torch.zeros_like(target, requires_grad=True)
    loss = criterion(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


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
