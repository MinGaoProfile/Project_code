import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rnv_seg.config import load_config
from rnv_seg.data.sampling import BalancedPatchBatchSampler
from rnv_seg.engine import (
    postprocess_prediction_masks,
    postprocess_rnv_mask,
    postprocess_vasculature_mask,
    save_probability_heatmaps,
    sliding_window_logits,
    train,
)
from rnv_seg.engine.evaluator import evaluate_loader
from rnv_seg.engine.trainer import (
    _hard_positive_patch_score,
    _refresh_full_volume_hard_negative_scores,
    _refresh_hard_negative_scores,
    _refresh_hard_positive_scores,
)
from rnv_seg.losses import build_loss
from rnv_seg.models.unet3d import UNet3D
from rnv_seg.workflows import (
    _assert_resume_compatible,
    _scheduler,
    evaluate_full_volume_records,
)


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = torch.zeros(1, 8, 8, 8)
        target = torch.zeros(1, 8, 8, 8)
        target[:, 2:6, 2:6, 2:6] = 1
        image.copy_(target)
        return {"image": image, "target": target, "sample_id": f"synthetic-{index}"}


def test_resume_rejects_changed_lesion_size_grouping() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint_config = load_config(
        root / "configs" / "experiments" / "unet3d_oct_octa.yaml"
    )
    current_config = copy.deepcopy(checkpoint_config)
    current_config["data"]["lesion_size_groups"]["large_component_voxels"] += 1
    with pytest.raises(ValueError, match="data.lesion_size_groups"):
        _assert_resume_compatible(current_config, checkpoint_config)


def test_one_epoch_training_and_sliding_window(tmp_path: Path, capsys) -> None:
    loader = DataLoader(TinyDataset(), batch_size=1)
    model = UNet3D(1, base_channels=2, levels=2)
    criterion = build_loss({"components": {"dice": 1.0, "bce": 1.0}})
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    config = {
        "training": {
            "epochs": 1,
            "gradient_accumulation_steps": 1,
            "amp": False,
            "checkpoint_metric": "validation_loss",
            "checkpoint_mode": "min",
            "full_volume_checkpoint": {
                "enabled": True,
                "every_n_epochs": 1,
                "metric": "validation_loss",
                "mode": "min",
                "evaluate_at_training_end": True,
            },
            "early_stopping_patience": 2,
        },
        "inference": {"threshold": 0.5},
        "metrics": {
            "lesion": {"enabled": True, "connectivity": 26, "minimum_voxels": 1, "match_iou": 0.1},
            "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
            "surface": {"enabled": False},
        },
    }
    for child in ("checkpoints", "logs"):
        (tmp_path / child).mkdir()
    summary = train(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        None,
        config,
        tmp_path,
        torch.device("cpu"),
        full_volume_evaluator=lambda _model, _epoch: {
            "dice": 0.75,
            "validation_loss": 1.25,
        },
    )
    assert summary["epochs_completed"] == 1
    assert summary["best_epoch"] == 1
    assert summary["checkpoint_metric"] == "validation_loss"
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "checkpoints" / "best_loss.pt").is_file()
    assert (tmp_path / "checkpoints" / "best_dice.pt").is_file()
    assert summary["best_patch_checkpoint"] is None
    assert (tmp_path / "checkpoints" / "best_full_volume.pt").is_file()
    assert summary["best_full_volume_epoch"] == 1
    assert summary["best_full_volume_dice"] == 0.75
    metrics = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert set(metrics) == {
        "epoch",
        "train_loss",
        "train_dice",
        "val_loss",
        "val_dice",
        "positive_dice",
        "small_only_dice",
        "large_containing_dice",
        "stratified_rnv_score",
        "negative_patch_clean_rate",
        "lr",
        "is_best",
        "is_best_loss",
        "is_best_dice",
        "best_epoch",
        "best_loss_epoch",
        "best_dice",
    }
    assert metrics["is_best"] is True
    assert metrics["best_epoch"] == 1
    assert metrics["best_dice"] == metrics["val_dice"]
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["training_state"]["best_epoch"] == 1
    assert checkpoint["training_state"]["best_dice"] == metrics["best_dice"]
    assert checkpoint["training_state"]["best_dice_epoch"] == 1
    assert checkpoint["training_state"]["best_full_volume_epoch"] == 1
    full_volume_metrics = json.loads(
        (tmp_path / "logs" / "full_volume_metrics.jsonl").read_text().splitlines()[0]
    )
    assert full_volume_metrics == {
        "epoch": 1,
        "validation_loss": 1.25,
        "full_volume_dice": 0.75,
        "stratified_rnv_score": None,
        "small_only_dice": None,
        "large_containing_dice": None,
        "is_best_loss": True,
        "is_best_dice": True,
        "best_loss_epoch": 1,
        "best_dice_epoch": 1,
        "best_dice": 0.75,
    }
    preferred = torch.load(
        tmp_path / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert preferred["metrics"] == {"dice": 0.75, "validation_loss": 1.25}
    output = capsys.readouterr()
    progress_text = output.out + output.err
    assert "train_loss=" in progress_text
    assert "train_dice=" in progress_text
    assert "val_loss=" in progress_text
    assert "val_dice=" in progress_text
    assert "lr=" in progress_text
    assert "is_best_loss=True" in progress_text
    assert "is_best_dice=True" in progress_text
    assert "[BEST MODEL]" in progress_text
    assert "[BEST LOSS]" in progress_text
    assert "[BEST LOSS MODEL]" in progress_text
    assert "val_lesion_f1=" not in progress_text
    assert "checkpoint_metric=" not in progress_text
    assert "gpu_peak=" not in progress_text
    assert "last_saved=" not in progress_text
    logits = sliding_window_logits(model.eval(), torch.zeros(1, 8, 8, 8), (4, 4, 4), 0.5)
    assert logits.shape == (1, 1, 8, 8, 8)


def test_resume_legacy_checkpoint_recovers_best_dice_from_history(
    tmp_path: Path,
) -> None:
    loader = DataLoader(TinyDataset(), batch_size=1)
    model = UNet3D(1, base_channels=2, levels=2)
    criterion = build_loss({"components": {"dice": 1.0, "bce": 1.0}})
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    config = {
        "training": {
            "epochs": 2,
            "gradient_accumulation_steps": 1,
            "amp": False,
            "checkpoint_metric": "dice",
            "checkpoint_mode": "max",
            "early_stopping_patience": 5,
        },
        "inference": {"threshold": 0.5},
        "metrics": {
            "lesion": {"enabled": False},
            "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
            "surface": {"enabled": False},
        },
    }
    for child in ("checkpoints", "logs"):
        (tmp_path / child).mkdir()
    metrics_path = tmp_path / "logs" / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"epoch": 1, "dice": 0.75}) + "\n", encoding="utf-8"
    )
    train(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        None,
        config,
        tmp_path,
        torch.device("cpu"),
        resume_payload={
            "epoch": 1,
            "training_state": {
                "best": 2.0,
                "best_epoch": 1,
                "without_improvement": 0,
            },
        },
    )
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert rows[-1]["best_dice"] == 0.75


def test_full_volume_evaluation_uses_inference_postprocessing(monkeypatch) -> None:
    target = torch.zeros(1, 12, 12, 6)
    target[:, 2:4, 2:7, :5] = 1
    raw_prediction = target.clone()
    raw_prediction[:, 9, 9, 5] = 1

    monkeypatch.setattr(
        "rnv_seg.workflows.load_full_case",
        lambda _record, _config: (torch.zeros(1, 12, 12, 6), target),
    )
    monkeypatch.setattr(
        "rnv_seg.workflows.sliding_window_logits",
        lambda *_args, **_kwargs: torch.where(
            raw_prediction[None] > 0,
            torch.tensor(20.0),
            torch.tensor(-20.0),
        ),
    )
    config = {
        "data": {
            "modalities": ["oct"],
            "target_names": ["rnv"],
            "patch_size": [8, 8, 4],
        },
        "training": {"amp": False},
        "inference": {
            "threshold": 0.5,
            "patch_size": [12, 12, 6],
            "sliding_window_overlap": 0.5,
            "postprocessing": {
                "enabled": True,
                "minimum_2d_component_pixels": 10,
                "minimum_consecutive_bscans": 5,
                "connectivity_2d": 8,
                "connectivity_3d": 26,
            },
        },
        "metrics": {
            "lesion": {"enabled": False},
            "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
            "surface": {"enabled": False},
        },
    }
    metrics = evaluate_full_volume_records(
        UNet3D(1, base_channels=2, levels=2),
        [SimpleNamespace(sample_id="synthetic")],
        config,
        torch.device("cpu"),
        criterion=build_loss(
            {"components": {"dice": 1.0, "bce": 1.0}}
        ),
    )
    assert metrics["dice"] == 1.0
    assert 0.03 < metrics["validation_loss"] < 0.04


def test_full_volume_metrics_average_cases_not_voxels(monkeypatch) -> None:
    small_target = torch.zeros(1, 2, 2, 2)
    small_target.reshape(-1)[0] = 1
    large_target = torch.zeros(1, 4, 4, 4)
    large_target.reshape(-1)[:9] = 1
    targets = {"small": small_target, "large": large_target}

    monkeypatch.setattr(
        "rnv_seg.workflows.load_full_case",
        lambda record, _config: (
            torch.zeros_like(targets[record.sample_id]),
            targets[record.sample_id],
        ),
    )
    monkeypatch.setattr(
        "rnv_seg.workflows.sliding_window_logits",
        lambda _model, image, *_args, **_kwargs: torch.zeros(
            1, 1, *image.shape[-3:]
        ),
    )
    monkeypatch.setattr(
        "rnv_seg.workflows.evaluate_binary_case",
        lambda _prediction, reference, _config: {
            "dice": 0.2 if reference.size == 8 else 0.8
        },
    )

    class TargetSumLoss(torch.nn.Module):
        def forward(self, _logits, target):
            return target.sum()

    config = {
        "data": {
            "modalities": ["oct"],
            "target_names": ["rnv"],
            "patch_size": [2, 2, 2],
        },
        "training": {"amp": False},
        "inference": {
            "threshold": 0.5,
            "patch_size": [4, 4, 4],
            "sliding_window_overlap": 0.5,
            "postprocessing": {"enabled": False},
        },
        "metrics": {},
    }
    metrics = evaluate_full_volume_records(
        UNet3D(1, base_channels=2, levels=2),
        [
            SimpleNamespace(sample_id="small"),
            SimpleNamespace(sample_id="large"),
        ],
        config,
        torch.device("cpu"),
        criterion=TargetSumLoss(),
    )
    assert metrics["validation_loss"] == 5.0
    assert metrics["dice"] == pytest.approx(0.5)


def test_full_volume_reports_positive_and_negative_case_diagnostics(
    monkeypatch,
) -> None:
    positive = torch.zeros(1, 4, 4, 4)
    positive[:, :2, :2, :2] = 1
    negative = torch.zeros_like(positive)
    targets = {"positive": positive, "negative": negative}

    monkeypatch.setattr(
        "rnv_seg.workflows.load_full_case",
        lambda record, _config: (
            torch.zeros_like(targets[record.sample_id]),
            targets[record.sample_id],
        ),
    )

    queued_targets = iter((positive, negative))

    def logits_for_record(_model, _image, *_args, **_kwargs):
        # Calls have identical shapes, so use an explicit queue for deterministic cases.
        selected = next(queued_targets)
        return torch.where(
            selected[None] > 0,
            torch.tensor(20.0),
            torch.tensor(-20.0),
        )

    monkeypatch.setattr(
        "rnv_seg.workflows.sliding_window_logits", logits_for_record
    )
    config = {
        "data": {
            "modalities": ["oct"],
            "target_names": ["rnv"],
            "patch_size": [4, 4, 4],
        },
        "training": {"amp": False},
        "inference": {
            "threshold": 0.5,
            "patch_size": [4, 4, 4],
            "sliding_window_overlap": 0.5,
            "postprocessing": {"enabled": False},
        },
        "metrics": {
            "lesion": {"enabled": False},
            "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
            "surface": {"enabled": False},
        },
    }
    metrics = evaluate_full_volume_records(
        UNet3D(1, base_channels=2, levels=2),
        [
            SimpleNamespace(sample_id="positive"),
            SimpleNamespace(sample_id="negative"),
        ],
        config,
        torch.device("cpu"),
    )
    assert metrics["positive_dice"] == 1.0
    assert metrics["negative_false_positive_voxels"] == 0.0
    assert metrics["negative_false_positive_fraction"] == 0.0
    assert metrics["negative_patch_clean_rate"] == 1.0
    assert metrics["small_only_dice"] == 1.0
    assert metrics["small_only_case_count"] == 1.0
    assert metrics["rnv_negative_case_count"] == 1.0
    assert metrics["stratified_rnv_score"] == 1.0


def test_shared_postprocessing_constrains_vasculature_to_rnv() -> None:
    rnv = np.zeros((8, 8, 5), dtype=np.uint8)
    rnv[1:3, 1:6, :] = 1
    vasculature = np.zeros_like(rnv)
    vasculature[1, 2, 1:4] = 1
    vasculature[7, 7, 1:4] = 1
    masks = postprocess_prediction_masks(
        {"rnv": rnv, "vasculature": vasculature},
        {
            "postprocessing": {
                "enabled": True,
                "minimum_2d_component_pixels": 10,
                "minimum_consecutive_bscans": 5,
                "connectivity_2d": 8,
                "connectivity_3d": 26,
                "vasculature": {
                    "minimum_3d_component_voxels": 3,
                    "minimum_consecutive_bscans": 2,
                    "connectivity_3d": 26,
                },
            }
        },
    )
    assert masks["vasculature"][1, 2].sum() == 3
    assert masks["vasculature"][7, 7].sum() == 0


def test_validation_reports_positive_and_pure_negative_metrics() -> None:
    class MixedDataset(Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            image = torch.zeros(1, 8, 8, 8)
            target = torch.zeros(1, 8, 8, 8)
            if index == 0:
                target[:, 2:6, 2:6, 2:6] = 1
                image.copy_(target)
            return {"image": image, "target": target, "sample_id": str(index)}

    metrics = evaluate_loader(
        UNet3D(1, base_channels=2, levels=2),
        DataLoader(MixedDataset(), batch_size=2),
        torch.device("cpu"),
        {
            "data": {"target_names": ["rnv"]},
            "training": {"amp": False},
            "inference": {"threshold": 0.5},
            "metrics": {
                "lesion": {"enabled": False},
                "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
                "surface": {"enabled": False},
            },
        },
    )
    assert "positive_dice" in metrics
    assert "negative_false_positive_voxels" in metrics
    assert "negative_false_positive_fraction" in metrics
    assert "negative_max_probability" in metrics
    assert "negative_patch_clean_rate" in metrics


def test_hard_negative_mining_scores_only_negative_pool() -> None:
    class MiningDataset(Dataset):
        def __init__(self) -> None:
            self.training = True
            self.patch_requests = [
                (0, (0, 0, 0)),
                (1, (0, 0, 0)),
                (0, (0, 0, 8)),
                (1, (0, 0, 8)),
            ]
            self.positive_indices = [0, 1]
            self.negative_indices_by_category = {
                "adjacent": [3],
                "anatomical_coverage": [2],
            }
            self.patch_request_slots = [None, None, (0, 0, 8), (1, 0, 8)]

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            target = torch.zeros(1, 8, 8, 8)
            if index < 2:
                target[:, 2:6, 2:6, 2:6] = 1
            return {
                "image": target.clone(),
                "target": target,
                "sample_id": str(index),
            }

    dataset = MiningDataset()
    sampler = BalancedPatchBatchSampler(
        dataset,
        2,
        {"positive_fraction": 0.5, "hard_negative_fraction": 0.5},
        seed=42,
        training=True,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    model = UNet3D(1, base_channels=2, levels=2)
    mining_config = {
        "data": {
            "balanced_patch_sampling": {
                "hard_negative_mining": {
                    "enabled": True,
                    "warmup_epochs": 1,
                    "refresh_every": 1,
                    "candidate_limit": 2,
                    "probability_threshold": 0.0,
                    "minimum_false_positive_voxels": 1,
                    "use_inference_postprocessing": False,
                    "maximum_pool_size": 2,
                    "top_probability_voxels": 16,
                }
            }
        }
    }
    scores = _refresh_hard_negative_scores(
        model,
        loader,
        torch.device("cpu"),
        mining_config,
        epoch=1,
        amp_enabled=False,
    )
    assert set(scores) == {2, 3}
    assert sampler.hard_negative_scores == scores

    mining_config["data"]["balanced_patch_sampling"]["hard_negative_mining"][
        "probability_threshold"
    ] = 1.0
    clean_scores = _refresh_hard_negative_scores(
        model,
        loader,
        torch.device("cpu"),
        mining_config,
        epoch=2,
        amp_enabled=False,
    )
    assert clean_scores == {}
    assert sampler.hard_negative_scores == {}


def test_hard_positive_score_rewards_missed_low_contrast_rnv() -> None:
    probability = torch.full((8, 8, 8), 0.2)
    target = torch.zeros(8, 8, 8)
    target[2:6, 2:6, 2:6] = 1
    low_contrast_oct = torch.full((8, 8, 8), 0.5)
    high_contrast_oct = torch.zeros(8, 8, 8)
    high_contrast_oct[target.bool()] = 1.0
    config = {
        "probability_threshold": 0.45,
        "minimum_missed_voxels": 1,
        "minimum_missed_fraction": 0.01,
        "contrast_ring_radius": 1,
        "missed_fraction_weight": 0.4,
        "confidence_shortfall_weight": 0.2,
        "low_contrast_weight": 0.4,
    }

    low_score = _hard_positive_patch_score(
        probability, target, low_contrast_oct, config
    )
    high_score = _hard_positive_patch_score(
        probability, target, high_contrast_oct, config
    )

    assert low_score is not None
    assert high_score is not None
    assert low_score > high_score


def test_hard_positive_mining_scores_only_low_confidence_labelled_patches() -> None:
    class PositiveMiningDataset(Dataset):
        def __init__(self) -> None:
            self.training = True
            self.patch_requests = [
                (0, (0, 0, 0)),
                (1, (0, 0, 0)),
                (0, (0, 0, 8)),
                (1, (0, 0, 8)),
            ]
            self.positive_indices = [0, 1]
            self.positive_indices_by_category = {
                "complete": [1],
                "boundary": [0],
            }
            self.positive_indices_by_scan_group = {
                "small_only": [0],
                "large_containing": [1],
            }
            self.negative_indices_by_category = {
                "adjacent": [3],
                "anatomical_coverage": [2],
            }
            self.patch_request_slots = [None, None, (0, 0, 8), (1, 0, 8)]
            self.patch_request_lesions = [
                (0, "rnv", 1),
                (1, "rnv", 1),
                None,
                None,
            ]
            self.patch_request_types = [
                "positive_boundary",
                "positive_complete",
                "negative_anatomical_coverage",
                "negative_adjacent",
            ]
            self.patch_request_scan_groups = [
                "small_only",
                "large_containing",
                None,
                None,
            ]

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            target = torch.zeros(1, 8, 8, 8)
            if index < 2:
                target[:, 2:6, 2:6, 2:6] = 1
            return {
                "image": torch.full((1, 8, 8, 8), 0.5),
                "target": target,
                "sample_id": str(index),
            }

    class LowConfidenceModel(torch.nn.Module):
        def forward(self, image):
            return SimpleNamespace(
                logits=torch.full(
                    (image.shape[0], 1, *image.shape[2:]),
                    -2.0,
                    device=image.device,
                )
            )

    dataset = PositiveMiningDataset()
    sampler = BalancedPatchBatchSampler(
        dataset,
        2,
        {
            "positive_fraction": 0.5,
            "hard_positive_fraction": 0.5,
            "hard_positive_mining": {
                "maximum_views_per_lesion_per_epoch": 3,
            },
        },
        seed=42,
        training=True,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    config = {
        "data": {
            "modalities": ["oct"],
            "target_names": ["rnv"],
            "balanced_patch_sampling": {
                "hard_positive_mining": {
                    "enabled": True,
                    "warmup_epochs": 1,
                    "refresh_every": 1,
                    "candidate_limit": 2,
                    "scoring_batch_size": 2,
                    "recheck_fraction": 0.5,
                    "probability_threshold": 0.45,
                    "minimum_missed_voxels": 1,
                    "minimum_missed_fraction": 0.01,
                    "contrast_ring_radius": 1,
                    "maximum_pool_size": 2,
                    "maximum_per_case": 1,
                    "maximum_per_lesion": 1,
                    "small_only_bonus": 0.15,
                    "boundary_bonus": 0.10,
                }
            },
        }
    }

    scores = _refresh_hard_positive_scores(
        LowConfidenceModel(),
        loader,
        torch.device("cpu"),
        config,
        epoch=1,
        amp_enabled=False,
    )

    assert set(scores) == {0, 1}
    assert scores[0] > scores[1]
    assert sampler.hard_positive_scores == scores


def test_full_volume_hard_negative_mining_maps_training_false_positives(
    monkeypatch,
) -> None:
    class MiningDataset:
        def __init__(self) -> None:
            self.training = True
            self.records = [
                SimpleNamespace(sample_id="case-0"),
                SimpleNamespace(sample_id="case-1"),
            ]
            self.patch_size = (2, 2, 2)
            self.patch_requests = [
                (0, (0, 0, 0)),
                (1, (0, 0, 0)),
                (0, (0, 0, 0)),
                (0, (2, 2, 2)),
                (1, (0, 0, 0)),
                (1, (2, 2, 2)),
            ]
            self.positive_indices = [0, 1]
            self.negative_indices_by_category = {
                "adjacent": [2, 4],
                "anatomical_coverage": [3, 5],
            }
            self.patch_request_slots = [
                None,
                None,
                (0, 0, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 1, 1),
            ]

    dataset = MiningDataset()
    sampler = BalancedPatchBatchSampler(
        dataset,
        2,
        {"positive_fraction": 0.5, "hard_negative_fraction": 0.5},
        seed=42,
        training=True,
    )
    loader = SimpleNamespace(dataset=dataset, batch_sampler=sampler)
    target = torch.zeros(1, 4, 4, 4)
    monkeypatch.setattr(
        "rnv_seg.engine.trainer.load_full_case",
        lambda _record, _config: (torch.zeros_like(target), target),
    )

    def false_positive_logits(*_args, **_kwargs):
        logits = torch.full((1, 1, 4, 4, 4), -20.0)
        logits[:, :, :2, :2, :2] = 20.0
        return logits

    monkeypatch.setattr(
        "rnv_seg.engine.trainer.sliding_window_logits", false_positive_logits
    )
    config = {
        "seed": 42,
        "data": {
            "patch_size": [2, 2, 2],
            "balanced_patch_sampling": {
                "hard_negative_mining": {
                    "enabled": True,
                    "minimum_false_positive_voxels": 1,
                    "maximum_pool_size": 8,
                    "maximum_per_case": 4,
                    "top_probability_voxels": 8,
                    "voxel_count_weight": 0.25,
                    "score_decay": 0.9,
                    "minimum_retained_score": 0.0,
                    "full_volume_refresh": {
                        "enabled": True,
                        "warmup_epochs": 1,
                        "refresh_every": 1,
                        "cases_per_refresh": 2,
                        "probability_threshold": 0.5,
                    },
                }
            },
        },
        "inference": {
            "threshold": 0.5,
            "patch_size": [4, 4, 4],
            "sliding_window_overlap": 0.5,
            "postprocessing": {"enabled": False},
        },
    }
    scores = _refresh_full_volume_hard_negative_scores(
        UNet3D(1, base_channels=2, levels=2),
        loader,
        torch.device("cpu"),
        config,
        epoch=1,
        amp_enabled=False,
    )
    assert set(scores) == {2, 4}
    assert sampler.hard_negative_scores == scores


def test_reduce_on_plateau_scheduler_lowers_learning_rate_after_stagnation() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
    scheduler = _scheduler(
        optimizer,
        {
            "name": "reduce_on_plateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 0,
            "threshold": 1.0e-4,
            "minimum_learning_rate": 1.0e-6,
        },
        epochs=10,
    )
    scheduler.step(1.0)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == 5.0e-4


def test_full_volume_loss_controls_checkpoint_lr_and_early_stopping(
    tmp_path: Path,
) -> None:
    loader = DataLoader(TinyDataset(), batch_size=2)
    model = UNet3D(1, base_channels=2, levels=2)
    criterion = build_loss({"components": {"dice": 1.0, "bce": 1.0}})
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
    )
    config = {
        "data": {},
        "training": {
            "epochs": 3,
            "gradient_accumulation_steps": 1,
            "amp": False,
            "checkpoint_metric": "validation_loss",
            "checkpoint_mode": "min",
            "full_volume_checkpoint": {
                "enabled": True,
                "every_n_epochs": 1,
                "metric": "validation_loss",
                "mode": "min",
            },
            "early_stopping_patience": 1,
        },
        "scheduler": {
            "name": "reduce_on_plateau",
            "monitor": "validation_loss",
        },
        "inference": {"threshold": 0.5},
        "metrics": {
            "lesion": {"enabled": False},
            "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
            "surface": {"enabled": False},
        },
    }
    for child in ("checkpoints", "logs"):
        (tmp_path / child).mkdir()

    losses = {1: 1.0, 2: 1.1, 3: 0.9}
    summary = train(
        model,
        loader,
        None,
        criterion,
        optimizer,
        scheduler,
        config,
        tmp_path,
        torch.device("cpu"),
        full_volume_evaluator=lambda _model, epoch: {
            "dice": 0.8 + 0.01 * epoch,
            "validation_loss": losses[epoch],
        },
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()
    ]
    assert summary["epochs_completed"] == 2
    assert summary["best_epoch"] == 1
    assert summary["best_loss_epoch"] == 1
    assert summary["best_dice_epoch"] == 1
    assert [row["is_best"] for row in rows] == [True, False]
    assert [row["is_best_loss"] for row in rows] == [True, False]
    assert [row["is_best_dice"] for row in rows] == [True, False]
    assert rows[-1]["lr"] == 5.0e-4
    best_loss = torch.load(
        tmp_path / "checkpoints" / "best_loss.pt",
        map_location="cpu",
        weights_only=False,
    )
    preferred = torch.load(
        tmp_path / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert best_loss["epoch"] == 1
    assert best_loss["metrics"]["validation_loss"] == 1.0
    assert preferred["epoch"] == 1
    assert preferred["metrics"]["validation_loss"] == 1.0


def test_prediction_postprocessing_requires_area_and_five_consecutive_bscans() -> None:
    mask = torch.zeros(24, 24, 12, dtype=torch.uint8).numpy()
    mask[1:6, 1:6, 0:5] = 1  # 25 pixels per frame: remove even across five frames.
    mask[8:13, 8:14, 1:6] = 1  # 30 pixels per frame across five frames: keep.
    mask[16:21, 16:22, 7:11] = 1  # 30 pixels but only four frames: remove.
    cleaned = postprocess_rnv_mask(
        mask,
        minimum_2d_pixels=30,
        minimum_consecutive_bscans=5,
    )
    assert cleaned[1:6, 1:6].sum() == 0
    assert cleaned[8:13, 8:14, 1:6].sum() == 150
    assert cleaned[16:21, 16:22].sum() == 0
    assert cleaned.dtype.name == "uint8"


def test_prediction_postprocessing_rejects_nonconsecutive_frame_runs() -> None:
    mask = torch.zeros(16, 16, 10, dtype=torch.uint8).numpy()
    mask[4:6, 4:9, [0, 1, 2, 4, 5, 6]] = 1
    cleaned = postprocess_rnv_mask(
        mask,
        minimum_2d_pixels=10,
        minimum_consecutive_bscans=5,
    )
    assert cleaned.sum() == 0


def test_vasculature_postprocessing_keeps_thin_consecutive_3d_component() -> None:
    mask = torch.zeros(8, 8, 5, dtype=torch.uint8).numpy()
    mask[3, 3, 1:4] = 1
    mask[6, 6, 0] = 1
    cleaned = postprocess_vasculature_mask(
        mask, minimum_3d_voxels=3, minimum_consecutive_bscans=2
    )
    assert cleaned[3, 3].sum() == 3
    assert cleaned[6, 6].sum() == 0


def test_probability_heatmaps_save_raw_diagnostic_views(tmp_path: Path) -> None:
    probability = np.zeros((8, 10, 5), dtype=np.float32)
    probability[2:6, 3:8, 2] = 0.8
    probability[3:5, 4:7, 4] = 0.6
    image = np.zeros((2, 8, 10, 5), dtype=np.float32)
    image[0] = np.linspace(0.0, 1.0, image[0].size).reshape(image[0].shape)
    image[1, 2:6, 3:8, 2] = 1.0

    metadata = save_probability_heatmaps(
        probability,
        image,
        ["oct", "octa"],
        tmp_path,
        "rnv",
        0.45,
        bscan_count=2,
        overlay_alpha=0.7,
    )

    expected = {
        "rnv_probability_enface.png",
        "rnv_probability_enface_overlay.png",
        "rnv_probability_bscans.png",
        "rnv_probability_bscan_overlay.png",
        "rnv_probability_heatmap.json",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert metadata["selected_bscan_indices_zero_based"] == [2, 4]
    assert metadata["selected_bscan_indices_matlab_one_based"] == [3, 5]
    assert metadata["binary_threshold"] == 0.45
    assert metadata["voxels_at_or_above_threshold"] == 26
    assert metadata["enface_background_modality"] == "octa"
    assert Image.open(tmp_path / "rnv_probability_enface.png").size == (10, 5)
