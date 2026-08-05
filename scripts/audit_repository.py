from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

BANNED_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".mat",
    ".dcm",
    ".dicom",
    ".h5",
    ".hdf5",
    ".tif",
    ".tiff",
    ".ckpt",
    ".pth",
    ".pt",
)
BANNED_PARTS = {
    "dataset",
    "patient_data",
    "raw_data",
    "medical_images",
    "labels",
    "masks",
    "checkpoints",
    "predictions",
    "logs",
}
MAX_TRACKED_BYTES = 10 * 1024 * 1024


def git_paths(repository: Path, staged: bool) -> list[Path]:
    if not (repository / ".git").exists():
        if staged:
            raise RuntimeError("Initialize the Project_code Git repository before using --staged")
        return [path for path in repository.rglob("*") if path.is_file()]
    git_prefix = ["git", "-c", f"safe.directory={repository.as_posix()}"]
    command = (
        git_prefix + ["diff", "--cached", "--name-only", "--diff-filter=ACMR"]
        if staged
        else git_prefix + ["ls-files", "--cached", "--others", "--exclude-standard"]
    )
    result = subprocess.run(command, cwd=repository, text=True, capture_output=True)
    if result.returncode != 0:
        if staged:
            raise RuntimeError(result.stderr.strip() or "Unable to inspect staged files")
        return [path for path in repository.rglob("*") if path.is_file() and ".git" not in path.parts]
    return [repository / line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail if private data or large artifacts could enter Git")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in git_paths(repository, args.staged):
        relative = path.relative_to(repository)
        lower = str(relative).replace("\\", "/").lower()
        parts = {part.lower() for part in relative.parts[:-1]}
        if lower.endswith(BANNED_SUFFIXES):
            violations.append(f"banned file type: {relative}")
        if parts & BANNED_PARTS and str(relative) not in {"outputs/.gitkeep"}:
            violations.append(f"private/artifact directory: {relative}")
        if path.exists() and path.stat().st_size > MAX_TRACKED_BYTES:
            violations.append(f"file exceeds 10 MiB: {relative}")
    if violations:
        print("Repository privacy audit failed:")
        for violation in sorted(set(violations)):
            print(f"  - {violation}")
        return 1
    print("Repository privacy audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
