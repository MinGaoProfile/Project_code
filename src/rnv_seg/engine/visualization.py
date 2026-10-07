from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


def _normalize_grayscale(plane: np.ndarray) -> np.ndarray:
    values = np.asarray(plane, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _colorize_probability(probability: np.ndarray) -> np.ndarray:
    """Map probabilities to a dark-blue, cyan, yellow, red diagnostic palette."""
    values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            [0, 0, 64],
            [0, 64, 255],
            [0, 255, 255],
            [255, 255, 0],
            [255, 0, 0],
        ],
        dtype=np.float32,
    )
    channels = [np.interp(values, stops, colors[:, index]) for index in range(3)]
    return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)


def _overlay_probability(
    background: np.ndarray,
    probability: np.ndarray,
    overlay_alpha: float,
) -> np.ndarray:
    gray = _normalize_grayscale(background)
    gray_rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    heatmap = _colorize_probability(probability).astype(np.float32)
    alpha = (overlay_alpha * np.clip(probability, 0.0, 1.0))[..., None]
    return np.rint(gray_rgb * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)


def _labeled_montage(planes: Sequence[np.ndarray], labels: Sequence[str]) -> Image.Image:
    if not planes:
        raise ValueError("At least one plane is required for a probability montage")
    images = [Image.fromarray(np.asarray(plane, dtype=np.uint8)) for plane in planes]
    label_height = 22
    gap = 4
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images) + label_height
    montage = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(montage)
    left = 0
    for image, label in zip(images, labels, strict=True):
        montage.paste(image, (left, label_height))
        draw.text((left + 4, 4), label, fill=(0, 0, 0))
        left += image.width + gap
    return montage


def _select_bscans(probability: np.ndarray, count: int) -> list[int]:
    frame_count = probability.shape[2]
    count = min(max(1, int(count)), frame_count)
    scores = probability.sum(axis=(0, 1), dtype=np.float64)
    if np.any(scores > 0):
        selected = np.argsort(scores, kind="stable")[-count:]
    else:
        selected = np.linspace(0, frame_count - 1, count, dtype=int)
    return sorted(int(index) for index in selected)


def save_probability_heatmaps(
    probability: np.ndarray,
    image: np.ndarray,
    modalities: Sequence[str],
    output_dir: str | Path,
    name: str,
    threshold: float,
    *,
    bscan_count: int = 6,
    overlay_alpha: float = 0.70,
) -> dict[str, object]:
    """Save en-face and high-probability B-scan probability visualizations.

    Volumes use the project axis convention ``[depth, width, bscan]`` and images
    use ``[channel, depth, width, bscan]``.
    """
    probability = np.asarray(probability, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    if probability.ndim != 3:
        raise ValueError("Probability heatmaps require a [depth, width, bscan] volume")
    if image.ndim != 4 or image.shape[1:] != probability.shape:
        raise ValueError("Image must have shape [channel, depth, width, bscan]")
    if image.shape[0] != len(modalities):
        raise ValueError("Image channel count must match modalities")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Probability threshold must be in [0, 1]")
    if not 0.0 <= float(overlay_alpha) <= 1.0:
        raise ValueError("Probability overlay_alpha must be in [0, 1]")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0)
    selected = _select_bscans(clipped, bscan_count)

    oct_index = modalities.index("oct") if "oct" in modalities else 0
    enface_index = modalities.index("octa") if "octa" in modalities else oct_index
    if modalities[enface_index] == "octa":
        enface_background = image[enface_index].max(axis=0).T
    else:
        enface_background = image[enface_index].mean(axis=0).T
    enface_probability = clipped.max(axis=0).T

    enface_heatmap = _colorize_probability(enface_probability)
    enface_overlay = _overlay_probability(
        enface_background,
        enface_probability,
        float(overlay_alpha),
    )
    Image.fromarray(enface_heatmap).save(
        destination / f"{name}_probability_enface.png"
    )
    Image.fromarray(enface_overlay).save(
        destination / f"{name}_probability_enface_overlay.png"
    )

    heatmap_planes = [_colorize_probability(clipped[:, :, index]) for index in selected]
    overlay_planes = [
        _overlay_probability(
            image[oct_index, :, :, index],
            clipped[:, :, index],
            float(overlay_alpha),
        )
        for index in selected
    ]
    labels = [f"B-scan {index} (MATLAB {index + 1})" for index in selected]
    _labeled_montage(heatmap_planes, labels).save(
        destination / f"{name}_probability_bscans.png"
    )
    _labeled_montage(overlay_planes, labels).save(
        destination / f"{name}_probability_bscan_overlay.png"
    )

    frame_scores = clipped.sum(axis=(0, 1), dtype=np.float64)
    metadata: dict[str, object] = {
        "target": name,
        "volume_shape_depth_width_bscan": list(clipped.shape),
        "probability_min": float(clipped.min()),
        "probability_max": float(clipped.max()),
        "probability_mean": float(clipped.mean()),
        "binary_threshold": float(threshold),
        "voxels_at_or_above_threshold": int((clipped >= threshold).sum()),
        "selected_bscan_indices_zero_based": selected,
        "selected_bscan_indices_matlab_one_based": [index + 1 for index in selected],
        "selected_bscan_probability_sums": [float(frame_scores[index]) for index in selected],
        "enface_background_modality": modalities[enface_index],
        "bscan_background_modality": modalities[oct_index],
        "overlay_alpha": float(overlay_alpha),
    }
    (destination / f"{name}_probability_heatmap.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return metadata
