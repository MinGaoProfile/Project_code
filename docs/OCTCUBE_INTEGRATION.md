# OCTCube integration and experiment controls

## Provenance and external dependency

This project integrates the official OCTCube implementation from
<https://github.com/ZucksLiu/OCTCubeM> through an adapter. It does not vendor the
third-party source or checkpoint. The official `OCTCube.pth` file is distributed from
<https://huggingface.co/zucksliu/OCTCubeM> under gated access; obtain authorization,
keep the file outside Git, and set:

```bash
export OCTCUBE_SOURCE_ROOT=/path/to/OCTCubeM
export OCTCUBE_CHECKPOINT=/path/to/OCTCube.pth
```

The adapter validates compatible parameter coverage before training, so a checkpoint
for the wrong architecture or token geometry cannot load silently. Evaluation and
inference restore OCTCube parameters from the RNV training checkpoint, but still need
the official source import to construct the architecture.

## Input compatibility contract

The checked-in production configuration records the official grayscale, patch-16,
256-pixel spatial input contract and makes temporal frame count, temporal patch size,
position embedding type, and class-token use explicit. Each OCT patch is processed as:

1. Select the `oct` channel by modality name, not by an assumed fixed channel index.
2. Permute dataset `[D,H,W]` to OCTCube `[frames,height,width]` using
   `model.octcube.axis_order`.
3. Resample to `[num_frames,input_size,input_size]`.
4. Apply the configured OCTCube intensity transform.
5. Convert transformer tokens back to a 3D grid, build a multi-scale feature pyramid,
   and invert the axis permutation before segmentation decoding.

The default `[2,0,1]` axis mapping assumes MATLAB axes are `[axial,fast-scan,slow-scan]`.
It is a declared acquisition assumption, not a fact inferred from array shape. Before
production training, verify it from scanner export metadata and visually compare at
least three landmarked volumes before and after the round trip.

The current staged normalization first uses the dataset percentile transform to map
OCT to `[0,1]`, then applies `(x-0.25)/0.25` in the OCTCube adapter. This matches the
explicit fixed transform exposed by the official dataset utility, including preserving
zero-valued background voxels. The configured dataset flip axes map to OCTCube's
temporal and width flips after reorientation. Record and freeze these choices across
all OCTCube comparisons; if the authorized pretrained run used a different transform,
change the YAML and rerun every OCTCube arm.

## Multimodal architecture

- OCT: official OCTCube encoder plus projected spatial feature pyramid.
- OCTA: independent 3D convolutional pyramid.
- Artery and vein priors: separate 3D convolutional pyramids, preserving the option to
  ablate either prior without changing another branch.
- Fusion: `concatenation`, `addition`, learned `attention`, or `cross_modal` context.
- Output: scale-wise fused skips and a 3D U-Net decoder producing one RNV logit volume.

All variants use the same dataset, registry, optimizer builder, trainer, losses,
metrics, validation, sliding-window inference, checkpointing, and run metadata.

## Training strategies and configs

| Strategy | Configuration | OCTCube trainable parameters |
|---|---|---|
| Frozen | `octcube_unet_oct_frozen.yaml` | None |
| Partial fine-tuning | `octcube_unet_oct_partial_finetune.yaml` | Last 4 blocks and final norm |
| Full fine-tuning | `octcube_unet_oct_finetune.yaml` | All |
| OCT + OCTA | `octcube_unet_oct_octa.yaml` | Frozen by default |
| Four modalities | `octcube_unet_multimodal.yaml` | Frozen by default |
| Four modalities + attention | `octcube_unet_multimodal_attention.yaml` | Frozen by default |
| Scratch control | `unet3d_oct.yaml` and other `unet3d_*` configs | Entire 3D U-Net |

For a fair frozen/partial/full comparison, hold the split manifest, patch sampler,
decoder width, loss, validation schedule, and seed fixed. Full fine-tuning should use a
separate, lower encoder learning rate in a future parameter-group experiment if GPU
memory permits; the current common optimizer intentionally preserves the shared-engine
baseline.

## Preflight checklist

- Confirm OCTCube source commit and checkpoint hash in the run notes.
- Confirm source factory is `vit_large_patch16` and checkpoint coverage is at least 90%.
- Confirm acquisition axis semantics and vessel-prior alignment visually.
- Confirm all training and test vessel priors pass strict 600×600 exact alignment.
- Confirm input statistics after both normalization stages.
- Run a one-batch forward/backward smoke test at the production patch size.
- Check GPU memory separately for frozen, partial, and full tuning.
- Use the same subject-grouped split for every architecture comparison.
