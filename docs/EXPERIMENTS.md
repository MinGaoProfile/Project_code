# Experiment protocol

## Minimum comparison matrix

Run each model/input combination with the same split version, preprocessing, patch
size, optimizer budget, stopping rule, threshold policy, and evaluation code. The five
starter configurations are the minimum matrix; use at least three seeds for conclusions
that depend on small differences.

## Controlled axes

- Input: OCT; OCT+OCTA; OCT+OCTA+artery+vein
- Model: U-Net; conditional diffusion U-Net; foundation-initialized U-Net
- Loss: Dice+BCE baseline, then focal/Tversky ablations
- Preprocessing: percentile normalization and documented alternatives
- Patch: size and foreground-sampling probability
- Optimization: learning rate, batch/accumulation, initialization, seed

Do not change multiple axes under one experiment name. Copy a starter YAML and encode
the changed axis in the name.

## Acceptance checks

1. Repository privacy audit passes.
2. Dataset variables/shapes and vessel priors pass the read-only data audit.
3. Split manifest has zero subject or eye overlap.
4. Training resolves the configuration and environment snapshot before epoch 1.
5. Best/last checkpoints and JSONL curves are present.
6. Predictions use pseudonymous filenames and remain ignored.
7. All enabled metrics include an explicit threshold and component definition.
8. Key results are repeated across seeds and reported with distribution/uncertainty.

