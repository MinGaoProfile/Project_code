from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from scipy.io import whosmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnv_seg.config import load_config
from rnv_seg.data import assert_no_group_leakage, discover_scans, vessel_map_dir_for_split


def variables(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    return {name: (tuple(shape), dtype) for name, shape, dtype in whosmat(path)}


def vessel_summary(records) -> dict[str, object]:
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    available = 0
    for record in records:
        if record.vessel_map_path is None:
            continue
        available += 1
        with Image.open(record.vessel_map_path) as image:
            dimensions[f"{image.width}x{image.height}"] += 1
            modes[image.mode] += 1
    return {
        "available": available,
        "dimensions": dict(sorted(dimensions.items())),
        "modes": dict(sorted(modes.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only RNV dataset structure audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-records", type=int, default=0, help="0 checks every record")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    root = Path(data["dataset_root"])
    suffix = data.get("vessel_prior", {}).get("filename_suffix", "_inner_octa_mask.png")
    train_folders = [path for path in (root / data["train_split"]).iterdir() if path.is_dir()]
    test_folders = [path for path in (root / data["test_split"]).iterdir() if path.is_dir()]
    invalid_train_folders = sum(
        len(list(path.glob("*_oct_octa.mat"))) != 1 or len(list(path.glob("*_mask.mat"))) != 1
        for path in train_folders
    )
    invalid_test_folders = sum(len(list(path.glob("*_oct_octa.mat"))) != 1 for path in test_folders)
    train = discover_scans(
        data["dataset_root"],
        data["train_split"],
        vessel_map_dir_for_split(data, data["train_split"]),
        suffix,
        True,
        False,
    )
    test = discover_scans(
        data["dataset_root"],
        data["test_split"],
        vessel_map_dir_for_split(data, data["test_split"]),
        suffix,
        False,
        False,
    )
    assert_no_group_leakage({"train_source": train, "test": test}, data.get("split_group", "subject"))
    selected = train[: args.max_records or None]
    errors: list[str] = []
    if invalid_train_folders:
        errors.append(f"{invalid_train_folders} training folders have missing/duplicate image or target files")
    if invalid_test_folders:
        errors.append(f"{invalid_test_folders} test folders have missing/duplicate image files")
    vessel_modalities = {"artery", "vein", "large_vessel"} & set(data["modalities"])
    missing_test_vessels = 0
    if vessel_modalities:
        missing_test_vessels = sum(record.vessel_map_path is None for record in test)
        if missing_test_vessels:
            errors.append(
                f"{missing_test_vessels} test scans lack vessel priors required by {sorted(vessel_modalities)}"
            )
    image_shapes: set[tuple[int, ...]] = set()
    target_shapes: set[tuple[int, ...]] = set()
    variables_config = data["mat_variables"]
    required_images = [variables_config[name] for name in data["modalities"] if name in {"oct", "octa"}]
    for record in selected:
        image_vars = variables(record.image_path)
        target_vars = variables(record.target_path) if record.target_path else {}
        for name in required_images:
            if name not in image_vars:
                errors.append(f"Missing image variable {name} in sample {record.sample_id}")
            else:
                image_shapes.add(image_vars[name][0])
        for target in data.get("target_names", ["rnv"]):
            key = "target" if target == "rnv" else f"{target}_target"
            target_name = variables_config[key]
            if target_name not in target_vars:
                errors.append(f"Missing target variable {target_name} in sample {record.sample_id}")
            else:
                target_shapes.add(target_vars[target_name][0])
        if vessel_modalities:
            if record.vessel_map_path is None:
                errors.append(f"Missing vessel prior for sample {record.sample_id}")
            else:
                with Image.open(record.vessel_map_path) as image:
                    if image.mode not in {"RGB", "RGBA"}:
                        errors.append(f"Vessel prior is not RGB for sample {record.sample_id}")
    report = {
        "train_scans": len(train),
        "train_subjects": len({record.subject_id for record in train}),
        "test_scans": len(test),
        "test_subjects": len({record.subject_id for record in test}),
        "records_checked": len(selected),
        "invalid_train_folders": invalid_train_folders,
        "invalid_test_folders": invalid_test_folders,
        "missing_test_vessel_priors": missing_test_vessels,
        "train_vessel_priors": vessel_summary(train),
        "test_vessel_priors": vessel_summary(test),
        "image_shapes": sorted(image_shapes),
        "target_shapes": sorted(target_shapes),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
