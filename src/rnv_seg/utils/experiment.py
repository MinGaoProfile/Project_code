from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from rnv_seg.config import save_config


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run_directory(config: dict[str, Any], repository: str | Path) -> Path:
    root = Path(repository)
    output_root = root / config["project"].get("output_root", "outputs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = config["project"]["experiment_name"]
    seed = int(config["seed"])
    run_dir = output_root / f"{stamp}_{name}_seed{seed}"
    for child in ("checkpoints", "logs", "predictions", "visualizations", "splits"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "resolved_config.yaml")
    metadata = {
        "experiment_name": name,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_environment_snapshot(run_dir / "environment.json")
    return run_dir


def write_environment_snapshot(path: str | Path) -> None:
    packages = {}
    for name in ("torch", "numpy", "scipy", "PyYAML", "Pillow", "tqdm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")

