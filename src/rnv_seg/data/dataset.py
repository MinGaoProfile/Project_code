from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from rnv_seg.registry import DATASETS
from rnv_seg.transforms import apply_random_flips, choose_patch_slices, pad_to_shape

from .discovery import ScanRecord


def _normalize(array: np.ndarray, method: str, percentiles: Sequence[float]) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if method == "none":
        return array
    if method == "minmax":
        low, high = float(array.min()), float(array.max())
    elif method == "percentile":
        low, high = np.percentile(array, percentiles)
    elif method == "zscore":
        std = float(array.std())
        return (array - float(array.mean())) / max(std, 1e-6)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _center_pad_or_crop_2d(array: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    result = array
    for axis, desired in enumerate(target):
        current = result.shape[axis]
        if current > desired:
            start = (current - desired) // 2
            slices = [slice(None), slice(None)]
            slices[axis] = slice(start, start + desired)
            result = result[tuple(slices)]
    pads = []
    for current, desired in zip(result.shape, target, strict=True):
        total = max(0, desired - current)
        pads.append((total // 2, total - total // 2))
    return np.pad(result, pads, mode="constant")


def _align_vessel_plane(
    array: np.ndarray, target: tuple[int, int], alignment: str
) -> np.ndarray:
    if alignment == "exact":
        if array.shape != target:
            raise ValueError(
                f"Vessel prior shape {array.shape} does not exactly match lateral volume shape {target}"
            )
        return array
    if alignment == "center_pad_or_crop":
        return _center_pad_or_crop_2d(array, target)
    raise ValueError(f"Unknown vessel-prior alignment method: {alignment}")


def _vessel_volume_patch(
    path: Path,
    channel: int,
    volume_shape: tuple[int, int, int],
    patch_slices: tuple[slice, slice, slice],
    depth_axis: int,
    alignment: str,
) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    lateral_axes = [axis for axis in range(3) if axis != depth_axis]
    target_plane = (volume_shape[lateral_axes[0]], volume_shape[lateral_axes[1]])
    plane = _align_vessel_plane(rgb[..., channel], target_plane, alignment)
    plane_slices = (patch_slices[lateral_axes[0]], patch_slices[lateral_axes[1]])
    plane_patch = plane[plane_slices]
    expanded = np.expand_dims(plane_patch, axis=depth_axis)
    return np.repeat(expanded, patch_slices[depth_axis].stop - patch_slices[depth_axis].start, axis=depth_axis)


@DATASETS.register("rnv_mat")
class RNVMatDataset(Dataset[dict[str, Any]]):
    """Lazy MATLAB reader that crops before float conversion to control memory use."""

    def __init__(
        self,
        records: Sequence[ScanRecord],
        config: dict[str, Any],
        training: bool,
    ) -> None:
        self.records = list(records)
        self.config = config
        self.training = training
        self.modalities = list(config["modalities"])
        self.variables = config["mat_variables"]
        self.axis_order = tuple(config.get("axis_order", [0, 1, 2]))
        self.patch_size = tuple(int(value) for value in config["patch_size"])
        self.samples_per_volume = int(config.get("samples_per_volume", 1)) if training else 1
        self.foreground_probability = float(config.get("foreground_patch_probability", 0.0))
        self.normalize = config.get("normalize", "percentile")
        self.percentiles = config.get("percentile_range", [0.5, 99.5])
        self.vessel = config.get("vessel_prior", {})
        self.random_flip_axes = tuple(int(axis) for axis in config.get("random_flip_axes", [0, 1, 2]))
        self.random_flip_probability = float(config.get("random_flip_probability", 0.5))

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_volume

    def _load_target(self, record: ScanRecord) -> np.ndarray | None:
        if record.target_path is None:
            return None
        data = loadmat(record.target_path, variable_names=[self.variables["target"]])
        target = np.asarray(data[self.variables["target"]])
        return np.transpose(target, self.axis_order)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index // self.samples_per_volume]
        draw = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(draw + index)
        target = self._load_target(record)

        variable_names = [self.variables[name] for name in self.modalities if name in {"oct", "octa"}]
        image_data = loadmat(record.image_path, variable_names=variable_names)
        reference = target
        if reference is None:
            reference = np.transpose(np.asarray(image_data[variable_names[0]]), self.axis_order)
        reference = pad_to_shape(reference, self.patch_size)
        shape = tuple(int(value) for value in reference.shape)
        slices = choose_patch_slices(
            shape,
            self.patch_size,
            rng,
            foreground=reference if target is not None else None,
            foreground_probability=self.foreground_probability if self.training else 0.0,
            random_crop=self.training,
        )

        channels: list[np.ndarray] = []
        for modality in self.modalities:
            if modality in {"oct", "octa"}:
                volume = np.transpose(np.asarray(image_data[self.variables[modality]]), self.axis_order)
                volume = pad_to_shape(volume, self.patch_size)[slices]
                channels.append(_normalize(volume, self.normalize, self.percentiles))
            elif modality in {"artery", "vein"}:
                if record.vessel_map_path is None:
                    raise FileNotFoundError(f"Vessel map is required for sample {record.sample_id}")
                channel_key = f"{modality}_channel"
                channels.append(
                    _vessel_volume_patch(
                        record.vessel_map_path,
                        int(self.vessel[channel_key]),
                        shape,
                        slices,
                        int(self.vessel.get("depth_axis", 0)),
                        self.vessel.get("alignment", "center_pad_or_crop"),
                    ).astype(np.float32)
                )

        image = np.stack(channels, axis=0)
        target_patch = None
        if target is not None:
            target_patch = pad_to_shape(target, self.patch_size)[slices]
            target_patch = (target_patch > 0).astype(np.float32)
        if self.training:
            image, target_patch = apply_random_flips(
                image,
                target_patch,
                rng,
                probability=self.random_flip_probability,
                axes=self.random_flip_axes,
            )
        sample: dict[str, Any] = {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "sample_id": record.sample_id,
        }
        if target_patch is not None:
            sample["target"] = torch.from_numpy(target_patch[None])
        return sample


def build_dataset(
    records: Sequence[ScanRecord], config: dict[str, Any], training: bool
) -> RNVMatDataset:
    name = config.get("dataset_name", "rnv_mat")
    return DATASETS.build(name, records=records, config=config, training=training)


def load_full_case(record: ScanRecord, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load one complete case for sliding-window inference.

    Full 768x600x600 volumes are memory intensive; callers should process one case
    at a time and keep sliding-window outputs on CPU.
    """
    modalities = list(config["modalities"])
    variables = config["mat_variables"]
    axis_order = tuple(config.get("axis_order", [0, 1, 2]))
    names = [variables[name] for name in modalities if name in {"oct", "octa"}]
    payload = loadmat(record.image_path, variable_names=names)
    volumes: list[np.ndarray] = []
    shape: tuple[int, int, int] | None = None
    for modality in modalities:
        if modality in {"oct", "octa"}:
            array = np.transpose(np.asarray(payload[variables[modality]]), axis_order)
            shape = tuple(int(value) for value in array.shape)
            volumes.append(
                _normalize(array, config.get("normalize", "percentile"), config.get("percentile_range", [0.5, 99.5]))
            )
        else:
            if shape is None:
                raise ValueError("OCT or OCTA must precede 2D vessel priors in data.modalities")
            if record.vessel_map_path is None:
                raise FileNotFoundError(f"Vessel map is required for sample {record.sample_id}")
            with Image.open(record.vessel_map_path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
            vessel = config.get("vessel_prior", {})
            depth_axis = int(vessel.get("depth_axis", 0))
            lateral_axes = [axis for axis in range(3) if axis != depth_axis]
            plane = _align_vessel_plane(
                rgb[..., int(vessel[f"{modality}_channel"])],
                (shape[lateral_axes[0]], shape[lateral_axes[1]]),
                vessel.get("alignment", "center_pad_or_crop"),
            )
            volumes.append(np.repeat(np.expand_dims(plane, depth_axis), shape[depth_axis], axis=depth_axis))
    target = None
    if record.target_path is not None:
        target_payload = loadmat(record.target_path, variable_names=[variables["target"]])
        target = np.transpose(np.asarray(target_payload[variables["target"]]), axis_order)
        target = torch.from_numpy((target > 0).astype(np.float32, copy=False)[None])
    return torch.from_numpy(np.stack(volumes).astype(np.float32, copy=False)), target
