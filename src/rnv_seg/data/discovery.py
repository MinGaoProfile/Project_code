from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_SUBJECT_EYE = re.compile(r"^(?P<subject>[^_]+)_(?P<eye>OD|OS)(?:-|_)", re.IGNORECASE)


@dataclass(frozen=True)
class ScanRecord:
    sample_id: str
    subject_id: str
    eye_id: str
    scan_dir: Path
    image_path: Path
    target_path: Path | None
    segmentation_path: Path | None
    vessel_map_path: Path | None

    @property
    def case_id(self) -> str:
        """Return the normalized human-readable case identifier used in local logs."""
        return self.subject_id.lower()

    def safe_dict(self) -> dict[str, str | None]:
        """Return fields suitable for local, Git-ignored manifests."""
        raw = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}


def parse_subject_eye(folder_name: str) -> tuple[str, str]:
    match = _SUBJECT_EYE.match(folder_name)
    if not match:
        raise ValueError(f"Cannot parse subject and eye from folder {folder_name!r}")
    return match.group("subject").upper(), match.group("eye").upper()


def _private_sample_id(folder_name: str) -> str:
    return hashlib.sha256(folder_name.encode("utf-8")).hexdigest()[:16]


def vessel_map_dir_for_split(data_config: dict[str, object], split: str) -> str | None:
    """Resolve a split-specific vessel-prior directory with legacy fallback."""
    directories = data_config.get("vessel_map_dirs", {})
    if isinstance(directories, dict):
        selected = directories.get(split)
        if selected:
            return str(selected)
    legacy = data_config.get("vessel_map_dir")
    return str(legacy) if legacy else None


def discover_scans(
    dataset_root: str | Path,
    split: str,
    vessel_map_dir: str | None = None,
    vessel_suffix: str = "_inner_octa_mask.png",
    require_target: bool = True,
    strict: bool = True,
) -> list[ScanRecord]:
    root = Path(dataset_root).expanduser()
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {split_root}")
    vessel_root = root / vessel_map_dir if vessel_map_dir else None
    records: list[ScanRecord] = []
    for scan_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        image_files = sorted(scan_dir.glob("*_oct_octa.mat"))
        if len(image_files) != 1:
            if strict:
                raise ValueError(f"Expected one *_oct_octa.mat in {scan_dir}, found {len(image_files)}")
            continue
        targets = sorted(scan_dir.glob("*_mask.mat"))
        if require_target and len(targets) != 1:
            if strict:
                raise ValueError(f"Expected one *_mask.mat in {scan_dir}, found {len(targets)}")
            continue
        segmentations = sorted(scan_dir.glob("*_seg.mat"))
        subject, eye = parse_subject_eye(scan_dir.name)
        vessel_path = vessel_root / f"{scan_dir.name}{vessel_suffix}" if vessel_root else None
        if vessel_path is not None and not vessel_path.exists():
            vessel_path = None
        records.append(
            ScanRecord(
                sample_id=_private_sample_id(scan_dir.name),
                subject_id=subject,
                eye_id=f"{subject}_{eye}",
                scan_dir=scan_dir,
                image_path=image_files[0],
                target_path=targets[0] if targets else None,
                segmentation_path=segmentations[0] if segmentations else None,
                vessel_map_path=vessel_path,
            )
        )
    if not records:
        raise ValueError(f"No scan folders were found in {split_root}")
    return records
