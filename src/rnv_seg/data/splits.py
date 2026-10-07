from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.io import loadmat
from tqdm.auto import tqdm

from .discovery import ScanRecord
from .lesion_groups import summarize_rnv_lesions


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
    *,
    strata_by_sample: Mapping[str, Sequence[str]] | None = None,
    search_trials: int = 4096,
    required_validation_strata: Sequence[str] = (),
) -> tuple[list[ScanRecord], list[ScanRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = sorted({_group_value(record, group) for record in records})
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required for train/validation split")
    count = min(max(1, round(len(groups) * validation_fraction)), len(groups) - 1)
    if strata_by_sample:
        validation_groups = _stratified_validation_groups(
            records,
            groups,
            count,
            validation_fraction,
            seed,
            group,
            strata_by_sample,
            search_trials,
            required_validation_strata,
        )
    else:
        random.Random(seed).shuffle(groups)
        validation_groups = set(groups[:count])
    train = [record for record in records if _group_value(record, group) not in validation_groups]
    validation = [record for record in records if _group_value(record, group) in validation_groups]
    assert_no_group_leakage({"train": train, "validation": validation}, group)
    return train, validation


def _stratified_validation_groups(
    records: Sequence[ScanRecord],
    groups: Sequence[str],
    validation_count: int,
    validation_fraction: float,
    seed: int,
    group: str,
    strata_by_sample: Mapping[str, Sequence[str]],
    search_trials: int,
    required_validation_strata: Sequence[str],
) -> set[str]:
    """Select whole groups while matching multi-label scan distributions."""
    features_by_group: defaultdict[str, Counter[str]] = defaultdict(Counter)
    total_features: Counter[str] = Counter()
    supporting_groups: defaultdict[str, set[str]] = defaultdict(set)
    scans_by_group: Counter[str] = Counter()
    for record in records:
        group_name = _group_value(record, group)
        scans_by_group[group_name] += 1
        for feature in strata_by_sample.get(record.sample_id, ("field:unknown",)):
            feature = str(feature)
            features_by_group[group_name][feature] += 1
            total_features[feature] += 1
            supporting_groups[feature].add(group_name)

    required = {
        feature for feature in required_validation_strata if total_features[feature]
    }

    def score(candidate: Sequence[str]) -> tuple[float, float, float, tuple[str, ...]]:
        selected = set(candidate)
        validation_features: Counter[str] = Counter()
        validation_scans = 0
        for group_name in selected:
            validation_features.update(features_by_group[group_name])
            validation_scans += scans_by_group[group_name]
        coverage_penalty = 0.0
        for feature, total in total_features.items():
            validation_value = validation_features[feature]
            train_value = total - validation_value
            if len(supporting_groups[feature]) >= 2 and (
                validation_value == 0 or train_value == 0
            ):
                coverage_penalty += 1.0
            if feature in required and validation_value == 0:
                coverage_penalty += 2.0
        distribution_error = sum(
            abs(validation_features[feature] - total * validation_fraction)
            / max(float(total), 1.0)
            for feature, total in total_features.items()
        )
        scan_error = abs(
            validation_scans - len(records) * validation_fraction
        ) / len(records)
        return (
            coverage_penalty,
            distribution_error,
            scan_error,
            tuple(sorted(selected)),
        )

    rng = random.Random(seed)
    ordered_groups = sorted(groups)
    candidates: set[tuple[str, ...]] = set()
    trials = max(int(search_trials), len(groups) * 32)
    for _ in range(trials):
        candidates.add(tuple(sorted(rng.sample(ordered_groups, validation_count))))
    # Include a deterministic baseline even when search_trials is very small.
    shuffled = ordered_groups.copy()
    random.Random(seed).shuffle(shuffled)
    candidates.add(tuple(sorted(shuffled[:validation_count])))
    return set(min(candidates, key=score))


def build_validation_strata(
    records: Sequence[ScanRecord],
    target_variable: str,
    *,
    field_code_regex: str = r"(?:^|[-_])(F\d{3})(?:[_-]|$)",
    large_component_voxels: int = 36_586,
    connectivity: int = 26,
    minimum_voxels: int = 30,
    progress: bool = False,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, Any]]]:
    """Read field codes and largest-component RNV groups for stratification."""
    pattern = re.compile(field_code_regex, re.IGNORECASE)
    lesion_summaries = {}
    field_codes: dict[str, str] = {}
    iterator = tqdm(
        records,
        desc="Indexing validation strata",
        unit="scan",
        disable=not progress,
        leave=False,
    )
    for record in iterator:
        if record.target_path is None:
            raise ValueError(f"Training target is missing for {record.sample_id}")
        payload = loadmat(record.target_path, variable_names=[target_variable])
        if target_variable not in payload:
            raise KeyError(
                f"Target variable {target_variable!r} is missing from {record.target_path}"
            )
        lesion_summaries[record.sample_id] = summarize_rnv_lesions(
            payload[target_variable],
            large_component_voxels=int(large_component_voxels),
            connectivity=int(connectivity),
            minimum_voxels=int(minimum_voxels),
        )
        match = pattern.search(record.scan_dir.name)
        field_codes[record.sample_id] = match.group(1).upper() if match else "UNKNOWN"

    strata: dict[str, tuple[str, ...]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        summary = lesion_summaries[record.sample_id]
        presence = "negative" if summary.group == "rnv_negative" else "positive"
        field = field_codes[record.sample_id]
        strata[record.sample_id] = (
            f"field:{field}",
            f"presence:{presence}",
            f"rnv_group:{summary.group}",
        )
        metadata[record.sample_id] = {
            "field_code": field,
            "rnv_voxels": summary.total_voxels,
            "largest_rnv_component_voxels": summary.largest_component_voxels,
            "rnv_component_count": summary.component_count,
            "rnv_group": summary.group,
            # Retained for old analysis scripts; values now use the two-group rule.
            "rnv_burden": summary.group,
        }
    return strata, metadata


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
    record_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Save a local manifest. The destination must remain Git-ignored."""
    root = Path(dataset_root).resolve()

    def rows(records: Sequence[ScanRecord]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for record in records:
            row: dict[str, Any] = {
                "sample_id": record.sample_id,
                "subject_id": record.subject_id,
                "eye_id": record.eye_id,
                "relative_scan_dir": str(record.scan_dir.resolve().relative_to(root)),
            }
            if record_metadata:
                row.update(record_metadata.get(record.sample_id, {}))
            output.append(row)
        return output

    def summary(records: Sequence[ScanRecord]) -> dict[str, Any]:
        fields: Counter[str] = Counter()
        groups: Counter[str] = Counter()
        for record in records:
            metadata = (record_metadata or {}).get(record.sample_id, {})
            fields[str(metadata.get("field_code", "UNKNOWN"))] += 1
            groups[str(metadata.get("rnv_group", "unknown"))] += 1
        return {
            "scans": len(records),
            "subjects": len({_group_value(record, group) for record in records}),
            "field_codes": dict(sorted(fields.items())),
            "rnv_group": dict(sorted(groups.items())),
        }

    payload = {
        "split_version": version,
        "seed": seed,
        "group": group,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train": rows(train),
        "validation": rows(validation),
        "summary": {
            "train": summary(train),
            "validation": summary(validation),
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
