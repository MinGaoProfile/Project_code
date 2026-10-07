from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _uint8_plane(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = (float(value) for value in np.percentile(finite, [1.0, 99.0]))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _selected_bscans(target: np.ndarray, count: int) -> list[int]:
    available = int(target.shape[-1])
    count = min(max(1, int(count)), available)
    scores = target.sum(axis=(0, 1))
    positive = np.flatnonzero(scores > 0)
    selected = list(positive[np.argsort(scores[positive])[-count:]]) if positive.size else []
    if len(selected) < count:
        evenly_spaced = np.linspace(0, available - 1, count, dtype=int)
        selected.extend(int(index) for index in evenly_spaced if int(index) not in selected)
    return sorted(int(index) for index in selected[:count])


def _join_planes(planes: list[np.ndarray], separator: int = 4) -> np.ndarray:
    if len(planes) == 1:
        return planes[0]
    gap_shape = list(planes[0].shape)
    gap_shape[1] = separator
    gap = np.full(gap_shape, 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for index, plane in enumerate(planes):
        if index:
            parts.append(gap)
        parts.append(plane)
    return np.concatenate(parts, axis=1)


def _record_index(dataset, index: int) -> int:
    requests = getattr(dataset, "patch_requests", None)
    if requests is not None:
        return int(requests[index][0])
    samples_per_volume = max(1, int(getattr(dataset, "samples_per_volume", 1)))
    return int(index) // samples_per_volume


def _round_robin_cases(
    dataset,
    indices: list[int],
    rng: np.random.Generator,
    *,
    excluded_cases: set[int] | None = None,
) -> list[int]:
    by_case: dict[int, list[int]] = {}
    for index in indices:
        by_case.setdefault(_record_index(dataset, index), []).append(int(index))
    cases = list(by_case)
    rng.shuffle(cases)
    for values in by_case.values():
        rng.shuffle(values)
    excluded_cases = excluded_cases or set()
    cases.sort(key=lambda case: case in excluded_cases)
    ordered: list[int] = []
    while any(by_case.values()):
        for case in cases:
            if by_case[case]:
                ordered.append(by_case[case].pop())
    return ordered


def _diverse_positive_indices(
    dataset,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    candidates = [int(index) for index in getattr(dataset, "positive_indices", [])]
    if not candidates or count <= 0:
        return []
    lesion_metadata = getattr(dataset, "patch_request_lesions", None)
    request_types = getattr(dataset, "patch_request_types", None)
    grouped: dict[object, list[int]] = {}
    for index in candidates:
        lesion = None if lesion_metadata is None else lesion_metadata[index]
        key: object = (
            ("lesion", *lesion)
            if lesion is not None
            else ("case", _record_index(dataset, index))
        )
        grouped.setdefault(key, []).append(index)

    representatives: list[int] = []
    for values in grouped.values():
        rng.shuffle(values)
        if request_types is not None:
            values.sort(
                key=lambda index: request_types[index] != "positive_complete"
            )
        representatives.append(values[0])
    ordered = _round_robin_cases(dataset, representatives, rng)
    if len(ordered) < count:
        remaining = [index for index in candidates if index not in set(ordered)]
        ordered.extend(_round_robin_cases(dataset, remaining, rng))
    return ordered[:count]


def _diverse_negative_indices(
    dataset,
    count: int,
    rng: np.random.Generator,
    excluded_cases: set[int],
) -> list[int]:
    categories = getattr(dataset, "negative_indices_by_category", {})
    if count <= 0 or not categories:
        return []
    category_order = ["anatomical_coverage", "adjacent"]
    category_order.extend(
        category for category in categories if category not in category_order
    )
    ordered_by_category = {
        category: _round_robin_cases(
            dataset,
            [int(index) for index in categories.get(category, [])],
            rng,
            excluded_cases=excluded_cases,
        )
        for category in category_order
    }
    selected: list[int] = []
    while len(selected) < count and any(ordered_by_category.values()):
        for category in category_order:
            if ordered_by_category[category]:
                selected.append(ordered_by_category[category].pop(0))
                if len(selected) == count:
                    break
    return selected


def _preview_indices(
    dataset,
    count: int,
    seed: int,
    positive_count: int | None,
    negative_count: int,
) -> list[int]:
    count = min(max(0, int(count)), len(dataset))
    if count == 0:
        return []
    rng = np.random.default_rng(int(seed))
    has_indexed_pools = bool(getattr(dataset, "positive_indices", []))
    if not has_indexed_pools:
        candidates = np.arange(len(dataset), dtype=np.int64)
        rng.shuffle(candidates)
        return [int(index) for index in candidates[:count]]

    requested_positive = count if positive_count is None else int(positive_count)
    requested_positive = min(max(0, requested_positive), count)
    requested_negative = min(max(0, int(negative_count)), count - requested_positive)
    positives = _diverse_positive_indices(dataset, requested_positive, rng)
    positive_cases = {_record_index(dataset, index) for index in positives}
    negatives = _diverse_negative_indices(
        dataset, requested_negative, rng, positive_cases
    )
    selected = positives + negatives
    if len(selected) < count:
        remaining = [index for index in range(len(dataset)) if index not in set(selected)]
        selected.extend(_round_robin_cases(dataset, remaining, rng)[: count - len(selected)])
    return selected[:count]


def _attach_preview_source_metadata(dataset, sample: dict[str, Any], index: int) -> None:
    sample["preview_dataset_index"] = int(index)
    records = getattr(dataset, "records", None)
    if records:
        record = records[_record_index(dataset, index)]
        sample["case_id"] = str(record.case_id)
    requests = getattr(dataset, "patch_requests", None)
    if requests is not None and requests[index][1] is not None:
        sample["patch_start"] = [int(value) for value in requests[index][1]]
    lesion_metadata = getattr(dataset, "patch_request_lesions", None)
    if lesion_metadata is not None and lesion_metadata[index] is not None:
        _, guidance_name, component_id = lesion_metadata[index]
        sample["lesion"] = {
            "target": str(guidance_name),
            "component_id": int(component_id),
        }


def save_sample_preview(
    sample: dict[str, Any],
    destination: str | Path,
    index: int,
    bscan_slices: int = 4,
    modalities: list[str] | None = None,
    target_names: list[str] | None = None,
) -> Path:
    output = Path(destination) / f"sample_{index:03d}_{sample['sample_id']}"
    output.mkdir(parents=True, exist_ok=True)
    image = sample["image"].detach().cpu().numpy().astype(np.float32, copy=False)
    target = sample["target"].detach().cpu().numpy().astype(np.float32, copy=False)
    np.savez_compressed(output / "sample.npz", image=image, target=target)

    modality_names = modalities or [f"channel_{channel}" for channel in range(image.shape[0])]
    label_names = target_names or (["rnv"] if target.shape[0] == 1 else [f"target_{i}" for i in range(target.shape[0])])
    if len(label_names) != target.shape[0]:
        raise ValueError("The number of target names must match target channels")
    reference_channel = modality_names.index("oct") if "oct" in modality_names else 0
    indices = _selected_bscans(target.max(axis=0), bscan_slices)
    colors = ([255, 0, 0], [0, 255, 255])
    for target_channel, target_name in enumerate(label_names):
        label_planes: list[np.ndarray] = []
        overlay_planes: list[np.ndarray] = []
        for bscan in indices:
            input_plane = _uint8_plane(image[reference_channel, :, :, bscan])
            label_plane = (target[target_channel, :, :, bscan] > 0.5).astype(np.uint8) * 255
            overlay = np.repeat(input_plane[..., None], 3, axis=2)
            foreground = label_plane > 0
            color = np.asarray(colors[min(target_channel, len(colors) - 1)])
            overlay[foreground] = np.rint(
                0.35 * overlay[foreground] + 0.65 * color
            ).astype(np.uint8)
            label_planes.append(label_plane)
            overlay_planes.append(overlay)
        Image.fromarray(_join_planes(label_planes)).save(output / f"label_{target_name}_bscans.png")
        Image.fromarray(_join_planes(overlay_planes)).save(output / f"overlay_{target_name}_bscans.png")
        if target_channel == 0:
            Image.fromarray(_join_planes(label_planes)).save(output / "label_bscans.png")
            Image.fromarray(_join_planes(overlay_planes)).save(output / "overlay_bscans.png")
    channel_stats: dict[str, dict[str, Any]] = {}
    for channel, modality in enumerate(modality_names):
        values = np.unique(image[channel])
        channel_stats[modality] = {
            "min": float(image[channel].min()),
            "max": float(image[channel].max()),
            "unique_values": values.tolist() if values.size <= 10 else None,
        }
        if modality in {"artery", "vein", "large_vessel"}:
            enface = (image[channel].max(axis=0).T > 0.5).astype(np.uint8) * 255
            Image.fromarray(enface).save(output / f"input_{modality}_enface.png")
        else:
            planes = [_uint8_plane(image[channel, :, :, bscan]) for bscan in indices]
            Image.fromarray(_join_planes(planes)).save(
                output / f"input_{modality}_bscans.png"
            )
    metadata = {
        "sample_id": str(sample["sample_id"]),
        "case_id": sample.get("case_id"),
        "preview_dataset_index": sample.get("preview_dataset_index"),
        "sampling_type": sample.get("sampling_type"),
        "patch_start": sample.get("patch_start"),
        "lesion": sample.get("lesion"),
        "image_shape": list(image.shape),
        "target_shape": list(target.shape),
        "image_dtype": str(image.dtype),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "target_voxels": {
            name: int((target[channel] > 0.5).sum())
            for channel, name in enumerate(label_names)
        },
        "target_names": label_names,
        "selected_bscans": indices,
        "modalities": modality_names,
        "channel_stats": channel_stats,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def save_dataset_previews(
    dataset,
    destination: str | Path,
    count: int,
    bscan_slices: int,
    seed: int,
    positive_count: int | None = None,
    negative_count: int = 0,
) -> list[Path]:
    if not len(dataset):
        return []
    cpu_rng_state = torch.random.get_rng_state()
    outputs: list[Path] = []
    try:
        source_indices = _preview_indices(
            dataset,
            count,
            seed,
            positive_count,
            negative_count,
        )
        for output_index, source_index in enumerate(source_indices):
            torch.manual_seed(int(seed) + output_index)
            sample = dict(dataset[source_index])
            _attach_preview_source_metadata(dataset, sample, source_index)
            outputs.append(
                save_sample_preview(sample, destination, output_index, bscan_slices)
                if not hasattr(dataset, "modalities")
                else save_sample_preview(
                    sample,
                    destination,
                    output_index,
                    bscan_slices,
                    list(dataset.modalities),
                    list(getattr(dataset, "target_names", ["rnv"])),
                )
            )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
    return outputs
