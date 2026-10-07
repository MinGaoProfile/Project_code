import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from rnv_seg.data.discovery import ScanRecord, vessel_map_dir_for_split
from rnv_seg.data.lesion_groups import summarize_rnv_lesions
from rnv_seg.data.splits import (
    assert_no_group_leakage,
    build_validation_strata,
    save_split_manifest,
    split_train_validation,
)


def record(subject: str, scan: int) -> ScanRecord:
    path = Path(f"/synthetic/{subject}/{scan}")
    return ScanRecord(str(scan), subject, f"{subject}_OD", path, path, path, None, None)


def test_subjects_never_cross_split() -> None:
    records = [record(subject, scan) for subject in ("A", "B", "C", "D") for scan in range(2)]
    train, validation = split_train_validation(records, 0.25, seed=7, group="subject")
    assert_no_group_leakage({"train": train, "validation": validation}, "subject")
    assert {item.subject_id for item in train}.isdisjoint({item.subject_id for item in validation})


def test_split_is_reproducible() -> None:
    records = [record(subject, 0) for subject in ("A", "B", "C", "D", "E")]
    first = split_train_validation(records, 0.4, seed=13)
    second = split_train_validation(records, 0.4, seed=13)
    assert [[item.subject_id for item in split] for split in first] == [
        [item.subject_id for item in split] for split in second
    ]


def test_split_specific_vessel_directories() -> None:
    config = {
        "vessel_map_dirs": {
            "train_data": "train_artery_vein_mask",
            "test_data": "test_artery_vein_mask",
        }
    }
    assert vessel_map_dir_for_split(config, "train_data") == "train_artery_vein_mask"
    assert vessel_map_dir_for_split(config, "test_data") == "test_artery_vein_mask"


def test_scan_group_uses_largest_component_not_total_foreground() -> None:
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask[7:9, 7:9, 7:9] = 1
    summary = summarize_rnv_lesions(
        mask,
        large_component_voxels=12,
        minimum_voxels=1,
    )
    assert summary.total_voxels == 16
    assert summary.largest_component_voxels == 8
    assert summary.component_count == 2
    assert summary.group == "small_only"


def test_subject_stratification_requires_negative_and_field_coverage() -> None:
    records = [record(subject, scan) for subject in "ABCDE" for scan in range(4)]
    fields = ("F120", "F122", "F130", "F132")
    # sample_id is intentionally duplicated by the helper, so give this test unique IDs.
    records = [
        ScanRecord(
            f"{item.subject_id}-{item.sample_id}",
            item.subject_id,
            item.eye_id,
            item.scan_dir,
            item.image_path,
            item.target_path,
            item.segmentation_path,
            item.vessel_map_path,
        )
        for item in records
    ]
    strata = {
        item.sample_id: [
            f"field:{fields[int(item.sample_id.split('-')[1])]}",
            (
                "presence:negative"
                if item.subject_id == "E" and item.sample_id.endswith("-0")
                else "presence:positive"
            ),
            f"burden:{('small', 'medium', 'large')[int(item.sample_id.split('-')[1]) % 3]}",
        ]
        for item in records
    }
    train, validation = split_train_validation(
        records,
        0.4,
        seed=11,
        group="subject",
        strata_by_sample=strata,
        search_trials=2000,
        required_validation_strata=["presence:negative"],
    )
    assert_no_group_leakage({"train": train, "validation": validation})
    validation_subjects = {item.subject_id for item in validation}
    assert "E" in validation_subjects
    validation_strata = {
        label for item in validation for label in strata[item.sample_id]
    }
    assert {f"field:{field}" for field in fields} <= validation_strata


def test_build_validation_strata_and_manifest_summary(tmp_path: Path) -> None:
    records = []
    counts = (0, 2, 8, 16)
    fields = ("F120", "F122", "F130", "F132")
    for index, (count, field) in enumerate(zip(counts, fields, strict=True)):
        scan_dir = tmp_path / f"S{index}_OD-{field}_scan"
        scan_dir.mkdir()
        target_path = scan_dir / "target_mask.mat"
        mask = np.zeros((2, 2, 4), dtype=np.uint8)
        mask.reshape(-1)[:count] = 1
        savemat(target_path, {"mask": mask})
        records.append(
            ScanRecord(
                str(index),
                f"S{index}",
                f"S{index}_OD",
                scan_dir,
                scan_dir / "image.mat",
                target_path,
                None,
                None,
            )
        )
    strata, metadata = build_validation_strata(
        records, "mask", large_component_voxels=8, minimum_voxels=1
    )
    assert metadata["0"] == {
        "field_code": "F120",
        "rnv_voxels": 0,
        "largest_rnv_component_voxels": 0,
        "rnv_component_count": 0,
        "rnv_group": "rnv_negative",
        "rnv_burden": "rnv_negative",
    }
    assert {metadata[str(index)]["rnv_group"] for index in range(1, 4)} == {
        "small_only",
        "large_containing",
    }
    assert "presence:negative" in strata["0"]

    manifest = tmp_path / "manifest.json"
    save_split_manifest(
        manifest,
        records[2:],
        records[:2],
        seed=42,
        version="test",
        group="subject",
        dataset_root=tmp_path,
        record_metadata=metadata,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["summary"]["validation"]["rnv_group"] == {
        "rnv_negative": 1,
        "small_only": 1,
    }
