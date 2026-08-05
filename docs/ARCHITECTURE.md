# Architecture and extension contracts

## Overall flow

`YAML -> discovery -> subject-grouped split -> lazy MAT reader -> patch transforms ->
model registry -> shared trainer -> shared metrics -> ignored run directory`

The dataset remains external. Discovery produces `ScanRecord` objects in memory and
uses SHA-256-derived sample IDs in prediction filenames. Local split manifests can
contain private membership data and therefore live only under ignored outputs.

## Data contract

Every dataset item is a dictionary:

- `image`: float tensor `[C, D, H, W]`
- `target`: binary float tensor `[1, D, H, W]` when available
- `sample_id`: a non-reversible short hash for logs and output filenames

MATLAB variables and axis order are configuration values. The loader reads only the
selected variables, crops the uint8 volume before float conversion during training,
and never modifies source files. En-face artery/vein planes are selected from
split-specific directories and repeated along the configured depth axis. The current
aligned 600×600 maps use strict `exact` shape validation; the legacy center-pad/crop
method remains available only through an explicit configuration. Spatial alignment
must still be visually validated against anatomy before publication.

## Model contract

Models implement `forward(image, target=None) -> SegmentationOutput` with:

- `logits`: `[B, 1, D, H, W]`
- `auxiliary_losses`: optional named scalar losses

The trainer applies the configured segmentation loss and adds auxiliary losses. New
models register one factory name in `rnv_seg.models`. Keep model-specific logic out of
the trainer.

`octcube_unet3d` follows the same contract. The OCT channel is reordered and resized
inside an OCTCube adapter, while OCTA, artery, and vein channels use independent 3D
feature pyramids. Concatenation, addition, learned modality attention, or cross-modal
context fusion is applied at every U-Net scale. The fused pyramid is decoded by the
same 3D U-Net-style decoder and therefore needs no special trainer or evaluator.

## Adding a modality

1. Add the public modality name and source-variable/path mapping to the data config.
2. Implement loading and spatial alignment in `RNVMatDataset` and `load_full_case`.
3. Add a synthetic shape/alignment test.
4. Add an experiment YAML; do not add patient-derived sample data.
5. Run the data and repository audits.

## Adding a model

1. Implement the shared output contract under `src/rnv_seg/models/`.
2. Register a unique lowercase name in `models/__init__.py`.
3. Keep `in_channels` dynamic so all modality combinations work.
4. Add forward/backward and odd-size tests.
5. Add a configuration and document checkpoint provenance if pretrained.

## Adding a loss or metric

1. Implement a pure function or module in the corresponding package.
2. Define behavior for empty prediction/target volumes.
3. Add synthetic edge cases: both empty, target-only, prediction-only, perfect match,
   disconnected components, and anisotropic spacing where applicable.
4. Hold threshold definitions constant across experiment comparisons.
