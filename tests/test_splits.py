from pathlib import Path

from rnv_seg.data.discovery import ScanRecord, vessel_map_dir_for_split
from rnv_seg.data.splits import assert_no_group_leakage, split_train_validation


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
