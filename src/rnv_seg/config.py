from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Environment variable {name!r} is required by the configuration")

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, recursively merge defaults, and expand environment variables."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    defaults = current.pop("defaults", [])
    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = (config_path.parent / default).resolve()
        merged = _deep_merge(merged, load_config(default_path))
    merged = _deep_merge(merged, current)
    merged = _expand_env(merged)
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    required = ("project", "data", "model", "loss", "training")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    modalities = config["data"].get("modalities", [])
    supported = {"oct", "octa", "artery", "vein"}
    unknown = set(modalities) - supported
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if not modalities:
        raise ValueError("At least one input modality is required")
    patch_size = config["data"].get("patch_size", [])
    if len(patch_size) != 3 or any(int(size) <= 0 for size in patch_size):
        raise ValueError("data.patch_size must contain three positive integers")
    levels = int(config["model"].get("levels", 4))
    divisor = 2 ** (levels - 1)
    if any(int(size) % divisor for size in patch_size):
        raise ValueError(f"Patch dimensions must be divisible by {divisor} for {levels} levels")
    if config["model"].get("name") == "octcube_unet3d":
        if "oct" not in modalities:
            raise ValueError("octcube_unet3d requires data.modalities to include oct")
        octcube = config["model"].get("octcube", {})
        axis_order = octcube.get("axis_order", [2, 0, 1])
        if sorted(axis_order) != [0, 1, 2]:
            raise ValueError("model.octcube.axis_order must be a permutation of [0, 1, 2]")
        input_size = int(octcube.get("input_size", 256))
        spatial_patch = int(octcube.get("patch_size", 16))
        num_frames = int(octcube.get("num_frames", 64))
        temporal_patch = int(octcube.get("temporal_patch_size", 2))
        if input_size % spatial_patch or num_frames % temporal_patch:
            raise ValueError("OCTCube input dimensions must be divisible by its patch dimensions")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def modality_count(config: dict[str, Any]) -> int:
    return len(config["data"]["modalities"])
