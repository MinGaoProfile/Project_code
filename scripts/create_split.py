from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnv_seg.config import load_config
from rnv_seg.data import (
    build_validation_strata,
    discover_scans,
    split_train_validation,
    vessel_map_dir_for_split,
)
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
    stratification = data.get("validation_stratification", {})
    strata_by_sample = None
    record_metadata = None
    if stratification.get("enabled", False):
        strata_by_sample, record_metadata = build_validation_strata(
            records,
            str(data["mat_variables"]["target"]),
            field_code_regex=str(
                stratification.get(
                    "field_code_regex", r"(?:^|[-_])(F\d{3})(?:[_-]|$)"
                )
            ),
            large_component_voxels=int(
                stratification.get("large_component_voxels", 36_586)
            ),
            connectivity=int(stratification.get("connectivity", 26)),
            minimum_voxels=int(stratification.get("minimum_voxels", 30)),
            progress=True,
        )
    train, validation = split_train_validation(
        records,
        float(data.get("validation_fraction", 0.2)),
        int(config["seed"]),
        data.get("split_group", "subject"),
        strata_by_sample=strata_by_sample,
        search_trials=int(stratification.get("search_trials", 4096)),
        required_validation_strata=stratification.get(
            "required_validation_strata", []
        ),
    )
    save_split_manifest(
        args.output,
        train,
        validation,
        int(config["seed"]),
        data.get("split_version", "v1"),
        data.get("split_group", "subject"),
        data["dataset_root"],
        record_metadata=record_metadata,
    )
    print(f"Saved {len(train)} training and {len(validation)} validation scans to {args.output}")


if __name__ == "__main__":
    main()
