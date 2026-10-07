from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat, whosmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnv_seg.config import load_config
from rnv_seg.data import discover_scans, vessel_map_dir_for_split
from rnv_seg.data.vascular_labels import generate_vasculature_mask


def _uint8(array: np.ndarray) -> np.ndarray:
    low, high = (float(value) for value in np.percentile(array, [1.0, 99.0]))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint(np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _save_preview(
    destination: Path,
    octa: np.ndarray,
    structural: np.ndarray,
    vasculature: np.ndarray,
    count: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    scores = vasculature.sum(axis=(0, 1))
    positive = np.flatnonzero(scores > 0)
    indices = list(positive[np.argsort(scores[positive])[-count:]]) if positive.size else []
    if len(indices) < count:
        indices.extend(
            int(index)
            for index in np.linspace(0, octa.shape[2] - 1, count, dtype=int)
            if int(index) not in indices
        )
    for bscan in sorted(indices[:count]):
        plane = _uint8(octa[:, :, bscan])
        overlay = np.repeat(plane[..., None], 3, axis=2)
        structural_plane = structural[:, :, bscan] > 0
        vessel_plane = vasculature[:, :, bscan] > 0
        overlay[structural_plane] = np.rint(
            0.6 * overlay[structural_plane] + 0.4 * np.array([255, 0, 0])
        ).astype(np.uint8)
        overlay[vessel_plane] = np.rint(
            0.25 * overlay[vessel_plane] + 0.75 * np.array([0, 255, 255])
        ).astype(np.uint8)
        Image.fromarray(overlay).save(destination / f"bscan_{bscan:03d}_overlay.png")
    enface = np.zeros((structural.shape[1], structural.shape[2], 3), dtype=np.uint8)
    enface[..., 0] = structural.max(axis=0).T.astype(np.uint8) * 255
    vessel_enface = vasculature.max(axis=0).T > 0
    enface[vessel_enface] = np.array([0, 255, 255], dtype=np.uint8)
    Image.fromarray(enface).save(destination / "enface_structural_vasculature.png")


def _write_target(
    target_path: Path,
    source_payload: dict[str, Any],
    variable: str,
    vasculature: np.ndarray,
    backup_root: Path,
    overwrite: bool,
) -> None:
    payload = {
        key: value
        for key, value in source_payload.items()
        if not key.startswith("__")
    }
    if variable in payload and not overwrite:
        raise FileExistsError(
            f"{variable} already exists in {target_path}; pass --overwrite to replace it"
        )
    backup = backup_root / target_path.parent.name / target_path.name
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(target_path, backup)
    payload[variable] = vasculature.astype(np.uint8, copy=False)
    with tempfile.NamedTemporaryFile(
        dir=target_path.parent, prefix=target_path.stem + "_", suffix=".mat", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        savemat(temporary, payload, do_compression=True)
        temporary.replace(target_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate OCTA-guided RNV vascular pseudo-labels inside structural masks"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--preview-scans", type=int, default=3)
    parser.add_argument("--preview-bscans", type=int, default=4)
    parser.add_argument("--write-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    records = discover_scans(
        data["dataset_root"],
        data["train_split"],
        vessel_map_dir_for_split(data, data["train_split"]),
        data.get("vessel_prior", {}).get("filename_suffix", "_inner_octa_mask.png"),
        require_target=True,
        strict=bool(data.get("strict_discovery", True)),
    )
    records = records[: args.max_scans or None]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backup_root = output / "original_target_backups"
    variables = data["mat_variables"]
    vascular_variable = str(variables.get("vasculature_target", "vasculature_mask"))
    generation = data.get("vascular_label_generation", {})
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        existing_variables = {name for name, _shape, _dtype in whosmat(record.target_path)}
        case_output = output / record.sample_id
        statistics_path = case_output / "statistics.json"
        if args.write_labels and vascular_variable in existing_variables and not args.overwrite:
            if statistics_path.is_file():
                rows.append(json.loads(statistics_path.read_text(encoding="utf-8")))
            print(
                f"[{index}/{len(records)}] {record.sample_id}: {vascular_variable} already exists; skipped",
                flush=True,
            )
            continue
        image = loadmat(record.image_path, variable_names=[variables["octa"]])
        target = loadmat(record.target_path) if args.write_labels else loadmat(
            record.target_path, variable_names=[variables["target"]]
        )
        octa = np.asarray(image[variables["octa"]])
        structural = np.asarray(target[variables["target"]]) > 0
        vasculature, statistics = generate_vasculature_mask(octa, structural, generation)
        if index <= max(0, int(args.preview_scans)):
            _save_preview(
                case_output,
                octa,
                structural,
                vasculature,
                max(1, int(args.preview_bscans)),
            )
        else:
            case_output.mkdir(parents=True, exist_ok=True)
        row = {"sample_id": record.sample_id, **statistics}
        rows.append(row)
        statistics_path.write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        if args.write_labels:
            _write_target(
                record.target_path,
                target,
                vascular_variable,
                vasculature,
                backup_root,
                bool(args.overwrite),
            )
        print(
            f"[{index}/{len(records)}] {record.sample_id}: "
            f"{statistics['vasculature_voxels']}/{statistics['structural_voxels']} voxels "
            f"({100 * float(statistics.get('vascular_fraction', 0.0)):.1f}%)",
            flush=True,
        )
    (output / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_status = (
        f"Labels were written; originals are backed up under {backup_root}."
        if args.write_labels
        else "No labels were written (preview mode)."
    )
    print(
        f"Saved previews and statistics for {len(rows)} scans to {output}. "
        + write_status,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
