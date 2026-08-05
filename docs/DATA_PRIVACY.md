# Data privacy and release safety

This repository is code-only. Real patient data, derived medical images, labels,
subject identifiers, private paths, and experiment outputs are never eligible for Git.

## Controls

- The dataset root comes from `RNV_DATASET_ROOT` or a Git-ignored local YAML.
- Medical and artifact extensions/directories are denied in `.gitignore`.
- `scripts/audit_repository.py` checks tracked or staged paths and large files.
- `.pre-commit-config.yaml` runs the staged-file guard.
- Predictions use hashes rather than scan folder names.
- Synthetic tests generate data in temporary directories at runtime.

## Before every push

```bash
python scripts/audit_repository.py
git status --short
git ls-files
git diff --cached --stat
```

Manually confirm that no text file, notebook output, plot title, screenshot, or commit
message contains subject identifiers or private infrastructure. If data was ever
committed, stop: removing the working-tree file is insufficient because Git history
still contains it. Revoke access as needed and use an approved history-rewrite and
incident-response process before any push.

