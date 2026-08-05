from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnv_seg.config import load_config
from rnv_seg.data import discover_scans, split_train_validation, vessel_map_dir_for_split
from rnv_seg.data.splits import save_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible subject-grouped split manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    records = discover_scans(
        data["dataset_root"],
        data["train_split"],
        vessel_map_dir_for_split(data, data["train_split"]),
        data.get("vessel_prior", {}).get("filename_suffix", "_inner_octa_mask.png"),
        True,
    )
    train, validation = split_train_validation(
        records,
        float(data.get("validation_fraction", 0.2)),
        int(config["seed"]),
        data.get("split_group", "subject"),
    )
    save_split_manifest(
        args.output,
        train,
        validation,
        int(config["seed"]),
        data.get("split_version", "v1"),
        data.get("split_group", "subject"),
        data["dataset_root"],
    )
    print(f"Saved {len(train)} training and {len(validation)} validation scans to {args.output}")


if __name__ == "__main__":
    main()
