# 3D RNV Segmentation

A configuration-driven PyTorch research framework for comparing input modalities and
3D segmentation architectures for retinal neovascularization (RNV). The repository
contains code and public-safe configuration only. Patient data and experiment outputs
must remain outside Git.

## Scientific objective

Measure how volumetric OCT, OCTA, artery/vein priors, network architecture, loss,
preprocessing, augmentation, patch size, initialization, and random seed affect 3D RNV
segmentation. All comparisons use the same discovery, split, training, inference, and
evaluation interfaces.

## Supported experiment axes

Inputs are selected with `data.modalities`:

- `oct` -> MATLAB variable `imgMat`
- `octa` -> MATLAB variable `imgMatAg`
- `artery` and `vein` -> red/green channels from an RGB en-face prior, aligned and
  broadcast along the configured depth axis

Implemented models:

- `unet3d`: standard 3D U-Net baseline
- `diffusion_unet3d`: conditional diffusion-style mask refinement around a 3D U-Net
- `foundation_unet3d`: 3D U-Net whose encoder is initialized from a compatible local
  foundation checkpoint, with coverage validation and optional freezing
- `octcube_unet3d`: the official pretrained OCTCube ViT for OCT, independent 3D
  encoders for OCTA/artery/vein, scale-wise multimodal fusion, and a shared U-Net decoder

The registry pattern allows attention, residual, nnU-Net, transformer, and other models
to be added without duplicating the training loop.

## Repository layout

```text
configs/                  shared and experiment YAML files
docs/                     architecture, experiment, privacy, and Git workflow guides
notebooks/                local exploration only
outputs/                  ignored local runs
scripts/                  training, inference, environment, data, and privacy commands
src/rnv_seg/
  data/                   discovery, subject-grouped splits, MATLAB dataset
  transforms/             patch sampling and augmentation
  models/                 model registry and three model families
  losses/                 Dice, BCE, focal, and Tversky losses
  metrics/                voxel, lesion, volume, scan, and optional surface metrics
  engine/                 training, evaluation, and sliding-window inference
  utils/                  checkpoints, seeds, environment, and experiment metadata
tests/                    synthetic-only tests
.github/                  issue and pull-request templates
```

## Environment setup

Use WSL2 or Linux with Python 3.10+ and an NVIDIA driver visible to WSL. Select the
PyTorch wheel matching the installed driver/CUDA environment from the official
[PyTorch installation selector](https://docs.pytorch.org/get-started/locally/), then:

```bash
cd /path/to/Project_code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-specific torch wheel selected on pytorch.org first.
python -m pip install -e ".[dev,visualization]"
python scripts/check_environment.py
```

`environment.yml` is also provided. Record the exact PyTorch, CUDA, driver, Python, and
package versions for every run; the training workflow writes them to
`outputs/<run>/environment.json` automatically.

## Dataset configuration

Never put the dataset inside this repository. Set a private external path:

```bash
export RNV_DATASET_ROOT=/path/to/RNV_dataset
```

Expected layout:

```text
/path/to/RNV_dataset/
  train_data/<scan_folder>/*_oct_octa.mat
  train_data/<scan_folder>/*_mask.mat
  train_data/<scan_folder>/*_seg.mat
  test_data/<scan_folder>/*_oct_octa.mat
  test_data/<scan_folder>/*_seg.mat
  train_artery_vein_mask/<scan_folder>_inner_octa_mask.png
  test_artery_vein_mask/<scan_folder>_inner_octa_mask.png
```

The default mapping is `imgMat` = OCT, `imgMatAg` = OCTA, and `mask` = RNV target.
Before training, run the read-only audit:

```bash
python scripts/check_data.py --config configs/experiments/unet3d_oct.yaml
```

If no validation folder exists, the framework creates a reproducible subject-grouped
split. It keeps every scan from one subject in one split and writes the local manifest
under the ignored run directory. To create it explicitly:

```bash
python scripts/create_split.py \
  --config configs/experiments/unet3d_oct.yaml \
  --output outputs/splits/rnv_v1_seed42.json
```

## Training and validation

```bash
python scripts/train.py --config configs/experiments/unet3d_oct.yaml
python scripts/train.py --config configs/experiments/unet3d_oct_octa.yaml
python scripts/train.py --config configs/experiments/unet3d_multimodal.yaml
python scripts/train.py --config configs/experiments/diffusion_unet_oct_octa.yaml
```

Foundation initialization requires a compatible, local encoder checkpoint:

```bash
export RNV_FOUNDATION_CHECKPOINT=/path/to/foundation_encoder.pt
python scripts/train.py --config configs/experiments/foundation_unet_multimodal.yaml
```

OCTCube experiments use the official external repository and authorized checkpoint;
the large third-party weights are never copied into this repository:

```bash
export OCTCUBE_SOURCE_ROOT=/path/to/OCTCubeM
export OCTCUBE_CHECKPOINT=/path/to/OCTCube.pth
python scripts/train.py --config configs/experiments/octcube_unet_oct_frozen.yaml
python scripts/train.py --config configs/experiments/octcube_unet_oct_finetune.yaml
python scripts/train.py --config configs/experiments/octcube_unet_oct_octa.yaml
python scripts/train.py --config configs/experiments/octcube_unet_multimodal.yaml
python scripts/train.py --config configs/experiments/octcube_unet_multimodal_attention.yaml
```

Use `octcube_unet_oct_partial_finetune.yaml` for last-block partial tuning. See
`docs/OCTCUBE_INTEGRATION.md` for checkpoint access, geometry validation, fusion
methods, and the required preflight checks.

Evaluate the deterministic validation split:

```bash
python scripts/evaluate.py \
  --config configs/experiments/unet3d_oct.yaml \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

## Testing and inference

The supplied `test_data` pattern is treated as unlabeled inference data. Full-volume
prediction uses overlap-averaged 3D sliding windows and writes only pseudonymous
sample hashes:

```bash
python scripts/infer.py \
  --config configs/experiments/unet3d_oct.yaml \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --output-dir outputs/<run>/predictions
```

Full volumes are large; inference processes one scan at a time and keeps accumulated
logits on CPU. Adjust patch size and overlap to fit GPU and host memory.
Configurations containing `artery` or `vein` select the matching split-specific
vessel-prior directory. The current train and test priors are both aligned 600×600 RGB
maps; `alignment: exact` makes any future size mismatch fail before inference.

## Metrics

- Voxel: Dice, IoU, precision, recall/sensitivity, specificity
- Lesion: component sensitivity, false positives, precision, F1, detected/missed counts
- Volume/scan: predicted volume, absolute and relative error, presence correctness
- Optional surface: surface Dice, 95th-percentile Hausdorff distance, ASSD

Surface metrics are disabled by default. Tiny irregular RNV components make these
metrics sensitive to voxel spacing, single-voxel boundary changes, and empty masks.
Enable them only after confirming physical voxel spacing and report the empty-mask
policy. Lesion matching thresholds and minimum component size are configuration fields
and must be held fixed across comparisons.

## Experiment tracking and naming

Run directories use `<UTC>_<model-and-input-name>_seed<seed>`. Each run records the
resolved configuration, seed, Git commit, Python/package/CUDA environment, JSONL loss
and metric history, `last.pt`, `best.pt`, predictions, and visualizations. Examples:

- `unet3d_oct`
- `unet3d_oct_octa`
- `unet3d_oct_octa_artery_vein`
- `diffusion_unet3d_oct_octa`
- `foundation_unet3d_oct_octa_artery_vein`

Change one major experiment axis at a time, repeat key comparisons across multiple
seeds, and never compare runs produced by different split manifests without declaring
that difference.

## Reproducibility and privacy

- Seeds cover Python, NumPy, CPU/CUDA PyTorch, and deterministic kernels.
- Split manifests record seed, grouping level, version, and membership locally.
- `.gitignore`, a pre-commit guard, and `scripts/audit_repository.py` block medical
  formats, checkpoints, predictions, logs, caches, local paths, and large files.
- Do not place subject IDs, scan folder names, screenshots, internal server paths, or
  patient-derived examples in issues, pull requests, notebooks, or documentation.

Before every commit or push:

```bash
python scripts/audit_repository.py
git status --short
git ls-files
git check-ignore /path/to/RNV_dataset/example.mat outputs/example/checkpoints/model.pt
```

See `docs/DATA_PRIVACY.md` for the release checklist.

## Development workflow

Use a single modular monorepo so every experiment shares one data contract and one
evaluation implementation. See `docs/GITHUB_WORKFLOW.md` for branch, commit, tag,
release, issue, pull-request, and first-push commands. Run before a pull request:

```bash
python -m pytest
python scripts/audit_repository.py
```

To add a modality, update the dataset modality mapping and add a public-safe config.
To add a model, implement the shared `SegmentationOutput` interface and register a
factory in `src/rnv_seg/models/__init__.py`. To add a loss, implement it in
`src/rnv_seg/losses/segmentation.py`, expose a configuration key, and add synthetic
tests. Detailed contracts are in `docs/ARCHITECTURE.md`.
