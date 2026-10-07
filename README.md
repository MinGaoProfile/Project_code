# 3D RNV Segmentation

A configuration-driven PyTorch research framework for comparing input modalities and
3D segmentation architectures for retinal neovascularization (RNV). The repository
contains code and public-safe configuration only. Patient data and experiment outputs
must remain outside Git.

## Scientific objective

Measure how volumetric OCT, OCTA, large-vessel priors, network architecture, loss,
preprocessing, augmentation, patch size, initialization, and random seed affect 3D RNV
segmentation. All comparisons use the same discovery, split, training, inference, and
evaluation interfaces.

## Supported experiment axes

Inputs are selected with `data.modalities`:

- `oct` -> MATLAB variable `imgMat`
- `octa` -> MATLAB variable `imgMatAg`
- `large_vessel` -> union of the cleaned artery (red + blue) and vein (green + blue)
  binary masks from an RGB en-face prior, aligned and broadcast along the configured
  depth axis. Multimodal experiments use this single vessel channel rather than
  separate artery and vein channels.

Implemented models:

- `unet3d`: five-level residual 3D U-Net with 24 base channels and dropout 0.1
- `diffusion_unet3d`: conditional diffusion-style refinement using residual 3D U-Nets
- `foundation_unet3d`: 3D U-Net whose encoder is initialized from a compatible local
  foundation checkpoint, with coverage validation and optional freezing
- `octcube_unet3d`: the official pretrained OCTCube ViT for OCT, independent 3D
  encoders for OCTA/large-vessel inputs, scale-wise multimodal fusion, and a shared
  U-Net decoder

The detailed layer-by-layer description and the exact training/inference paths for
the baseline 3D U-Net and U-Net + diffusion experiment are documented together in
[`docs/UNET_AND_DIFFUSION_ARCHITECTURE.md`](docs/UNET_AND_DIFFUSION_ARCHITECTURE.md).

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
  losses/                 Dice, BCE, focal, Tversky, and Focal Tversky losses
  metrics/                voxel, lesion, volume, scan, and optional surface metrics
  engine/                 training, evaluation, and sliding-window inference
  utils/                  checkpoints, seeds, environment, and experiment metadata
tests/                    synthetic-only tests
.github/                  issue and pull-request templates
```

## Data-processing documentation

The complete data-processing specification is maintained in
[`docs/DATA_PROCESSING.md`](docs/DATA_PROCESSING.md). It documents data discovery,
subject-grouped splits, OCT/OCTA normalization, large-vessel construction, ILM/BM
curved ROI extraction, RNV-guided patches, augmentation, preview outputs, validation,
and full-volume inference.

Any change to data-processing code or configuration must update both this README and
`docs/DATA_PROCESSING.md` in the same change, together with relevant tests. This is a
required repository maintenance rule, not optional follow-up documentation.

Current shared data defaults are: training patch `[480,224,24]`, CUDA AMP physical
batch size 2 with two-step gradient accumulation (effective batch size 4), full-B-scan
inference/validation patch `[768,600,24]`, multimodal order
`[oct,octa,large_vessel]`, and voxel spacing `[0.003,0.015,0.015]` mm. The curved ROI uses ILM layer 1 shifted to 20 px
above the highest RNV point and BM layer 8 shifted 100 px downward. OCT/OCTA training
augmentation uses contrast range `[0.8,1.2]` and Gaussian-noise standard-deviation
range `[0.0,0.05]`; large-vessel and label channels remain binary. Curved ROI is used
only to choose crop coordinates—the complete rectangular patch is then cut from the
unmasked original image, so augmentation remains spatially uniform.
The shared residual U-Net defaults use 24 base channels, five levels,
`Dropout3d=0.1`, and AdamW weight decay `1e-5`. A 16-channel/strong-regularization
ablation reduced full-volume validation Dice, so new runs restore the better-capacity
baseline. This changes tensor shapes from 16-channel runs; start a new training run
and do not resume their `last.pt`. To evaluate,
calibrate, or infer an older checkpoint, pass that run's saved `resolved_config.yaml`
so its original architecture is reconstructed.
OCT/OCTA percentile bounds are always computed from the complete raw case volume.
Training patches reuse cached case-level bounds, so training and full-volume inference
apply exactly the same intensity mapping.
Training indexing normally generates up to six views per eligible 3D RNV component: required
base views plus width/B-scan shifts of 25% of the patch size. Shifted views must retain
at least 20% of the component; a base view covering at least 80% is classified as
complete. Duplicate clipped coordinates are removed, ordinary per-component/per-case
limits are 6/20, and required base views are never dropped by these limits. Each epoch
samples complete/boundary positive views at 55%/45% using case-and-lesion round-robin.
Scans are classified offline from the largest complete 3D GT component using the
frozen 36,586-voxel threshold; ordinary positive sampling allocates 70% to `small_only`
scans and 30% to `large_containing` scans. This metadata never enters the network input.
At most two views from one 3D lesion are used in an epoch; other shifted views rotate
across epochs instead of repeatedly reinforcing the same lesion appearance.
Starting at epoch 3, hard-positive mining rescans up to 256 labelled positive patches
after every epoch with augmentation disabled. A patch enters the pool when at least 10
true-RNV voxels and 5% of its RNV label remain below probability `0.45`. Its score combines
missed-label fraction (0.5), normalized confidence shortfall (0.3), and low OCT contrast
against an expanded local bounding-box context (0.2), with bonuses for `small_only`
and boundary views. Half the mining budget rechecks the existing pool and half discovers
new case/lesion-balanced candidates. The pool is limited to 256 patches, 12 per case,
and three per lesion; stale scores decay by 0.9. Hard positives replace at most 30% of
positive slots and may use a third augmented view for a difficult lesion, while every
physical batch remains exactly 50% positive and 50% negative.
Pure-negative candidates must contain zero structural-RNV voxels. They are sampled as
lesion-adjacent (30%) and scan-wide anatomical coverage (70%). Retinal thickness is not
used to identify macula because temporal retina can also be thin; neither the scan centre
nor an acquisition field code is treated as an anatomical label. Alternative
top/centre/bottom and left/centre/right positions are created within each non-overlapping
B-scan slot. Selection is round-robin across scan records and slots, so scans containing
macula, disc, and temporal retina all contribute without inventing their locations.
At most one negative from each slot is taken per epoch. Every batch remains exactly
50% positive and 50% negative. Starting at epoch 8, up to
512 pure-negative candidates are rescored every two epochs in batches of two. One
quarter of that budget rechecks accumulated hard negatives; the rest performs
case/category/slot-balanced discovery. A patch enters the pool only when its prediction at
probability `>=0.4` survives the configured test-time area and consecutive-B-scan
postprocessing and contains at least 20 voxels. Its rank combines top-4096 confidence
and false-positive extent. The pool removes duplicate anatomical slots, keeps at most
16 entries per case and 768 total, and decays stale scores by `0.9` per refresh until
they fall below `0.35`. Case-round-robin hard selection prevents one scan from
dominating; hard examples replace at most 50% of ordinary negatives.
The `patch_plus_full_volume_v2` strategy starts at epoch 8; every four epochs the same
model also performs test-time sliding-window
inference on four rotating complete labelled training scans. Persistent false-positive
components at probability `>=0.5` are mapped back to the already verified pure-negative
patch coordinates. Validation scans are never eligible for hard-negative mining.
Hard-positive and hard-negative pools are both stored in checkpoint training state and
restored on resume; neither adds fields to the compact epoch metrics log.
Focal Tversky uses `alpha=0.6` for false positives and `beta=0.4` for false negatives.
Local patch-indexing progress uses the normalized lowercase case ID (for example,
`wfdr036`) instead of the 16-character hashed sample ID or the full scan-folder name.
The RNV depth anchor is placed at fraction `0.40` of the patch rather than at its
center. For a 480-pixel depth this places the anchor about 192 pixels below the patch
top, moving the crop 48 pixels upward relative to the previous `0.30` setting while
retaining approximately 288 pixels below the lesion anchor.

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
The optional dual-output experiment uses a second 3D variable named
`vasculature_mask` in the same `*_mask.mat`; both masks have the same shape, and
vasculature voxels are contained by `mask`. These labels can be generated as
OCTA-guided pseudo-labels from the structural RNV masks.
Before training, run the read-only audit:

```bash
python scripts/check_data.py --config configs/experiments/unet3d_oct.yaml
```

If no validation folder exists, the framework creates a reproducible subject-grouped,
multi-factor stratified split. Every scan from one subject remains in one split. Among
whole-subject candidates, it selects the validation set that best preserves acquisition
field codes, completely negative scans, and the `small_only`/`large_containing`
groups determined by the largest complete 3D RNV component.
Folder names expose `F120/F122/...` rather than reliable anatomical quadrant names, so
these field codes are balanced directly instead of being guessed as
superior/inferior/temporal/nasal. The local manifest records per-scan field code,
RNV voxel count (`rnv_voxels`), largest component size, component count, RNV group, and
train/validation summaries. To create and inspect it
explicitly:

```bash
python scripts/create_split.py \
  --config configs/experiments/unet3d_oct.yaml \
  --output outputs/splits/rnv_v3_seed42.json
```

The split search also requires a negative scan in validation when one exists. If a
field code occurs in at least two subjects, the optimizer penalizes omitting it from
either training or validation. Anatomical quadrant names should only be added after a
verified mapping from acquisition field code to retinal location is available. The
current manifest schema/split policy is versioned as
`v3_subject_lesion_size_stratified`.

## Training and validation

```bash
python scripts/train.py --config configs/experiments/unet3d_oct.yaml
python scripts/train.py --config configs/experiments/unet3d_oct_octa.yaml
python scripts/train.py --config configs/experiments/unet3d_multimodal.yaml
python scripts/train.py --config configs/experiments/unet3d_multimodal_vasculature.yaml
python scripts/train.py --config configs/experiments/diffusion_unet_oct_octa.yaml
```

Generate review previews without changing labels:

```bash
python scripts/generate_vasculature_labels.py \
  --config configs/experiments/unet3d_multimodal_vasculature.yaml \
  --output-dir outputs/vasculature_label_preview \
  --max-scans 3
```

After reviewing the red structural / cyan vascular overlays, write all labels:

```bash
python scripts/generate_vasculature_labels.py \
  --config configs/experiments/unet3d_multimodal_vasculature.yaml \
  --output-dir outputs/vasculature_label_generation \
  --write-labels
```

The generator denoises OCTA inside each structural RNV component, subtracts a local
background, combines intensity and local contrast, and applies Otsu-hysteresis
connected growth. It always intersects the result with `mask`. Before replacing a
target MAT, it stores the untouched file under
`<output-dir>/original_target_backups/<scan_folder>/`. Existing labels are skipped
unless `--overwrite` is supplied, so interrupted generation can be resumed safely.

`unet3d_multimodal_vasculature.yaml` has a shared decoder and independent sigmoid
heads for `rnv` and `vasculature`. Its loss is RNV loss + `0.5` vasculature loss +
`0.1` containment consistency. Like the single-output experiments, LR and early
stopping and the preferred inference checkpoint are all controlled by the same
smoothed full-volume validation loss. Run the data audit before training:

```bash
python scripts/check_data.py \
  --config configs/experiments/unet3d_multimodal_vasculature.yaml
```

Training keeps float32 model parameters and inputs but enables CUDA autocast and gradient
scaling, so most convolution activations use lower precision. The B-scan patch extent is
24 rather than 32 frames, reducing activation voxels by 25% while retaining full
480-pixel depth and 224-pixel width context. The physical batch size remains 2, and every
balanced microbatch therefore contains one positive and one negative patch. Gradients are
accumulated across two microbatches, giving each optimizer update an effective batch of
four patches (two positive and two negative). Dice, BCE, Focal, and Tversky loss
reductions remain float32 under autocast to prevent large-volume FP16 sum overflow.
Hard-positive and patch hard-negative scoring use batch size 1. The
terminal remains concise: batch progress shows only running
loss and RNV Dice; full-volume validation shows only validation loss and RNV Dice.
The final epoch line contains `train_loss`, `train_dice`, `val_loss`, `val_dice`, `lr`,
`is_best_loss`, and `is_best_dice`. New minima/maxima are also marked with
`[BEST LOSS]` and `[BEST LOSS MODEL]`; `[BEST MODEL]` means validation loss improved.
`last.pt` is saved every epoch. `logs/metrics.jsonl` contains only `epoch`,
`train_loss`, `train_dice`, `val_loss`, `val_dice`, `positive_dice`,
`small_only_dice`, `large_containing_dice`, `stratified_rnv_score`,
`negative_patch_clean_rate`, `lr`, `is_best`, `is_best_loss`, `is_best_dice`,
`best_epoch`, `best_loss_epoch`, and `best_dice`.
`positive_dice` averages only full-volume scans containing RNV;
`negative_patch_clean_rate` is the fraction of completely negative validation scans
with no postprocessed predicted voxels. A diagnostic is `null` only when that case
category is absent. `best_epoch` and `best_loss_epoch` both identify the lowest-loss
epoch; `best_dice` records the overall RNV Dice at that same epoch. Stratified metrics
remain diagnostics and never select a checkpoint or change LR/early stopping.
`train_dice` is the epoch-level hard Dice accumulated from the RNV output channel at
the configured inference threshold; `val_dice` is the mean validation RNV Dice.
Learning rate is controlled by a three-epoch moving mean of validation loss rather
than a fixed cosine schedule. If the smoothed loss fails to improve by at least 0.002
for five epochs, LR is multiplied by `0.5` down to `1e-6`. The updated learning rate appears in the concise epoch line and
metrics log; scheduler state remains in the checkpoint for exact resume.
Early stopping uses the same validation loss with patience 18, leaving time for multiple LR
reduction and several lower-LR epochs while limiting prolonged overfitting.

Every epoch is validated on every complete validation scan; ROI curves and patch
validation are not used. Validation uses the same percentile normalization and
`[768,600,24]` overlapping sliding windows as test inference. Full-volume loss is
computed from the overlap-averaged raw logits and complete labels; Dice is computed
after a `0.45` probability threshold and the configured test-time postprocessing.
The lower threshold recovers uncertain lesion boundaries while the 30-pixel and
five-consecutive-B-scan filters continue suppressing isolated detections. Loss and Dice are calculated separately
for each scan and then averaged across scans, so larger volumes do not receive extra
weight. Validation reports `small_only` and `large_containing` metrics using the same
36,586-voxel largest-component threshold as training. `validation_loss` controls
`best.pt`, `best_loss.pt`, the smoothed `ReduceLROnPlateau` input, and early stopping.
`best_full_volume.pt` and the legacy `best_dice.pt` are compatibility aliases of the
same lowest-loss model. The size-stratified score still reports small/large-lesion and
negative-scan behavior, but it cannot override the loss-selected model.

After training, calibrate `best.pt` without changing its weights. The command performs
one full-volume inference pass per validation scan, then evaluates all 60 combinations
of threshold `[0.45,0.5,0.55,0.6,0.65]`, per-B-scan minimum area
`[5,10,20,30]`, and minimum consecutive B-scans `[3,5,7]`:

```bash
python scripts/calibrate.py \
  --config outputs/<run>/resolved_config.yaml \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --output-dir outputs/<run>/calibration
```

Only combinations with pooled lesion sensitivity `>=0.82` are eligible. Eligible rows
are ranked by `0.5 × positive Dice + 0.5 × lesion F1`, with fewer false-positive lesions
as the tie-breaker. `calibration_results.csv` contains every combination,
`calibration_summary.json` records the decision, and `calibrated_config.yaml` contains
the selected inference settings. If no row meets the sensitivity floor, no calibrated
configuration is written. Calibration uses validation labels only; final test scans must
remain untouched to avoid optimistic test results.

Resume an interrupted run from its latest checkpoint. This continues in the original
run directory, appends to the existing metrics logs, preserves the checkpoint tracks, and starts at
the next epoch:

```bash
python scripts/train.py \
  --config configs/experiments/unet3d_multimodal.yaml \
  --resume outputs/<run>/checkpoints/last.pt
```

Resume restores model, optimizer, scheduler, best-loss epoch, and early-stopping
state. Runs created with the previous patch-Dice checkpoint objective are intentionally
incompatible with the new full-volume-loss objective and must be started as a new run.
Likewise, `patch_plus_full_volume_v2` changes the training sample stream and is not
silently injected into an older checkpoint. Resume an older run with that run's saved
`resolved_config.yaml`; start a new run to evaluate the optimized mining strategy.

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
prediction uses overlap-averaged 3D sliding windows:

```bash
python scripts/infer.py \
  --config configs/experiments/unet3d_oct.yaml \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --output-dir outputs/<run>/predicts
```

Full volumes are large; inference processes one scan at a time and keeps accumulated
logits on CPU. Adjust patch size and overlap to fit GPU and host memory.
Configurations containing `large_vessel` select the matching split-specific
vessel-prior directory. The current train and test priors are both aligned 600×600 RGB
maps; `alignment: exact` makes any future size mismatch fail before inference.

Prediction masks are postprocessed before saving. First, 8-connected 2D regions
smaller than 30 pixels are removed independently from each B-scan. Each remaining
26-connected 3D RNV component must then occur in at least five consecutive B-scans;
five nonconsecutive appearances do not pass. Results are stored in one folder per
original case ID:

```text
outputs/<run>/predicts/<case_id>/<source_file_id>_mask.mat
```

Single-output experiments save `mask`. The vasculature experiment saves two binary
`uint8` variables in the same MAT file: `mask` and `vasculature_mask`. RNV uses the
existing 2D-area/5-frame postprocessing. Vasculature uses a less aggressive 3D filter
(at least 10 voxels and two consecutive B-scans) and is finally intersected with the
postprocessed RNV mask.

Each case folder contains diagnostic probability outputs for every head:

- `<head>_probability_enface.png`: maximum-probability en-face projection;
- `<head>_probability_enface_overlay.png`: projection over OCTA when available,
  otherwise over OCT;
- `<head>_probability_bscans.png`: the six B-scans with the largest total predicted
  probability;
- `<head>_probability_bscan_overlay.png`: those probabilities over their OCT B-scans;
- `<head>_probability_heatmap.json`: zero-based/Python and one-based/MATLAB frame
  indices, threshold, probability summary, and threshold-crossing voxel count.

The PNG palette always represents the continuous range 0–1 and is not changed by
postprocessing. It therefore distinguishes predictions below the `0.45` threshold
from regions removed later by the 30-pixel/five-frame cleanup. Full float32 probability
volumes are disabled by default to reduce host-memory and output-file pressure. Set
`inference.save_probabilities: true` only when exact threshold recalibration is needed;
this adds `mask_probability` (and `vasculature_mask_probability` for the auxiliary head)
to the MAT file.

## Metrics

- Voxel: Dice, IoU, precision, recall/sensitivity, specificity
- Balanced validation: positive-patch metrics plus negative false-positive voxels,
  false-positive fraction, maximum RNV probability, and clean-patch rate
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
resolved configuration, seed, Git commit, Python/package/CUDA environment, compact
training/full-volume histories, `last.pt`, lowest-loss `best.pt`/`best_loss.pt`, legacy
compatibility aliases `best_dice.pt`/`best_full_volume.pt`,
predictions, and visualizations. Examples:

- `unet3d_oct`
- `unet3d_oct_octa`
- `unet3d_oct_octa_large_vessel`
- `diffusion_unet3d_oct_octa`
- `foundation_unet3d_oct_octa_large_vessel`

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


## preview data

Automatic training previews no longer take dataset indices `0,1,2`, which can be shifted
views of one lesion. The default writes four samples: three positive patches selected
from distinct cases/3D RNV components whenever available, plus one scan-wide
anatomical-coverage pure-negative patch. `metadata.json` records `case_id`, `preview_dataset_index`,
`patch_start`, lesion component ID, and sampling type so the selection can be audited.

```bash
python scripts/preview_data.py \
  --config configs/experiments/unet3d_oct.yaml \
  --output-dir outputs/data_preview \
  --samples 4

python scripts/preview_data.py \
  --config configs/experiments/unet3d_oct_octa.yaml \
  --output-dir outputs/data_preview_oct_octa \
  --samples 4
```
