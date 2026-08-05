from __future__ import annotations

import json
import random
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .discovery import ScanRecord


def _group_value(record: ScanRecord, group: str) -> str:
    if group == "subject":
        return record.subject_id
    if group == "eye":
        return record.eye_id
    raise ValueError("Split group must be 'subject' or 'eye'")


def split_train_validation(
    records: Sequence[ScanRecord],
    validation_fraction: float,
    seed: int,
    group: str = "subject",
) -> tuple[list[ScanRecord], list[ScanRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = sorted({_group_value(record, group) for record in records})
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required for train/validation split")
    random.Random(seed).shuffle(groups)
    count = min(max(1, round(len(groups) * validation_fraction)), len(groups) - 1)
    validation_groups = set(groups[:count])
    train = [record for record in records if _group_value(record, group) not in validation_groups]
    validation = [record for record in records if _group_value(record, group) in validation_groups]
    assert_no_group_leakage({"train": train, "validation": validation}, group)
    return train, validation


def assert_no_group_leakage(splits: dict[str, Sequence[ScanRecord]], group: str = "subject") -> None:
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for split_name, records in splits.items():
        for record in records:
            value = _group_value(record, group)
            if value in seen and seen[value] != split_name:
                collisions.append(value)
            seen[value] = split_name
    if collisions:
        raise ValueError(
            f"Subject-level data leakage detected across splits ({len(set(collisions))} groups)"
        )


def save_split_manifest(
    path: str | Path,
    train: Sequence[ScanRecord],
    validation: Sequence[ScanRecord],
    seed: int,
    version: str,
    group: str,
    dataset_root: str | Path,
) -> None:
    """Save a local manifest. The destination must remain Git-ignored."""
    root = Path(dataset_root).resolve()

    def rows(records: Sequence[ScanRecord]) -> list[dict[str, str]]:
        return [
            {
                "sample_id": record.sample_id,
                "subject_id": record.subject_id,
                "eye_id": record.eye_id,
                "relative_scan_dir": str(record.scan_dir.resolve().relative_to(root)),
            }
            for record in records
        ]

    payload = {
        "split_version": version,
        "seed": seed,
        "group": group,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train": rows(train),
        "validation": rows(validation),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

