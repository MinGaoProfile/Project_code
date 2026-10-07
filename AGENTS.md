# Repository maintenance instructions

When changing any data-processing behavior or configuration—including discovery, splits, variables, axes, normalization, modalities, vessel masks, ROI curves, patch extraction, sampling, augmentation, previews, validation data handling, or inference—update both `README.md` and `docs/DATA_PROCESSING.md` in the same change. Add or update tests that verify the new behavior and key documented defaults.

Do not treat documentation synchronization as optional cleanup; it is part of completing the implementation.

