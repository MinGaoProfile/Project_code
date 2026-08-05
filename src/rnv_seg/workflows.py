from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rnv_seg.config import load_config, modality_count
from rnv_seg.data import (
    assert_no_group_leakage,
    discover_scans,
    split_train_validation,
    vessel_map_dir_for_split,
)
from rnv_seg.data.dataset import build_dataset, load_full_case
from rnv_seg.data.splits import save_split_manifest
from rnv_seg.engine import evaluate_loader, sliding_window_logits, train
from rnv_seg.losses import build_loss
from rnv_seg.models import build_model
from rnv_seg.utils import create_run_directory, load_checkpoint, seed_everything


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _records(config: dict[str, Any], manifest_path: Path | None = None):
    data = config["data"]
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
        train_records, validation_records = split_train_validation(
            train_records,
            float(data.get("validation_fraction", 0.2)),
            int(config["seed"]),
            data.get("split_group", "subject"),
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
        )
    return train_records, validation_records


def _loader(records, data_config: dict[str, Any], training: bool, batch_size: int):
    dataset = build_dataset(records, data_config, training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        persistent_workers=bool(data_config.get("num_workers", 0) > 0),
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
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def train_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    repository = _repository_root()
    run_dir = create_run_directory(config, repository)
    train_records, validation_records = _records(config, run_dir / "splits" / "manifest.json")
    batch_size = int(config["training"].get("batch_size", 1))
    train_loader = _loader(train_records, config["data"], True, batch_size)
    validation_loader = _loader(validation_records, config["data"], False, batch_size)
    model = build_model(config["model"], modality_count(config), config["data"]["modalities"])
    criterion = build_loss(config["loss"])
    optimizer = _optimizer(model, config["optimizer"])
    scheduler = _scheduler(optimizer, config["scheduler"], int(config["training"]["epochs"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = train(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        run_dir,
        device,
    )
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def evaluate_from_config(config_path: str | Path, checkpoint: str | Path) -> dict[str, float]:
    config = load_config(config_path)
    _, validation_records = _records(config)
    loader = _loader(validation_records, config["data"], False, int(config["training"].get("batch_size", 1)))
    model_config = dict(config["model"])
    if model_config.get("name") == "foundation_unet3d":
        model_config["foundation_checkpoint"] = ""
        model_config["require_foundation_checkpoint"] = False
    if model_config.get("name") == "octcube_unet3d":
        model_config["octcube"] = dict(model_config["octcube"])
        model_config["octcube"]["checkpoint"] = ""
        model_config["octcube"]["require_checkpoint"] = False
    model = build_model(model_config, modality_count(config), config["data"]["modalities"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(checkpoint, model, device=device)
    model.to(device)
    return evaluate_loader(model, loader, device, config)


def infer_from_config(config_path: str | Path, checkpoint: str | Path, output_dir: str | Path) -> None:
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
    model_config = dict(config["model"])
    if model_config.get("name") == "foundation_unet3d":
        model_config["foundation_checkpoint"] = ""
        model_config["require_foundation_checkpoint"] = False
    if model_config.get("name") == "octcube_unet3d":
        model_config["octcube"] = dict(model_config["octcube"])
        model_config["octcube"]["checkpoint"] = ""
        model_config["octcube"]["require_checkpoint"] = False
    model = build_model(model_config, modality_count(config), config["data"]["modalities"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(checkpoint, model, device=device)
    model.to(device).eval()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    threshold = float(config["inference"].get("threshold", 0.5))
    for record in records:
        image, _ = load_full_case(record, data)
        logits = sliding_window_logits(
            model,
            image,
            data["patch_size"],
            float(config["inference"].get("sliding_window_overlap", 0.5)),
        )
        mask = (torch.sigmoid(logits)[0, 0] >= threshold).numpy().astype(np.uint8)
        np.savez_compressed(destination / f"{record.sample_id}_prediction.npz", mask=mask)
