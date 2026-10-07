from __future__ import annotations

import argparse
import json

from rnv_seg.workflows import (
    calibrate_from_config,
    evaluate_from_config,
    infer_from_config,
    preview_from_config,
    train_from_config,
)


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train an RNV segmentation model")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        help="Resume in the existing run directory from a last.pt checkpoint",
    )
    args = parser.parse_args()
    print(train_from_config(args.config, args.resume))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RNV segmentation checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_from_config(args.config, args.checkpoint), indent=2))


def calibrate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate RNV threshold and postprocessing on validation scans"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+")
    parser.add_argument("--minimum-areas", type=int, nargs="+")
    parser.add_argument("--consecutive-bscans", type=int, nargs="+")
    parser.add_argument("--minimum-lesion-sensitivity", type=float)
    args = parser.parse_args()
    summary = calibrate_from_config(
        args.config,
        args.checkpoint,
        args.output_dir,
        thresholds=args.thresholds,
        minimum_areas=args.minimum_areas,
        consecutive_bscans=args.consecutive_bscans,
        minimum_lesion_sensitivity=args.minimum_lesion_sensitivity,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


def infer_main() -> None:
    parser = argparse.ArgumentParser(description="Run full-volume sliding-window inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-scans", type=int)
    args = parser.parse_args()
    infer_from_config(args.config, args.checkpoint, args.output_dir, args.max_scans)


def preview_main() -> None:
    parser = argparse.ArgumentParser(description="Save cropped training inputs and labels for inspection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/data_preview")
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()
    preview_from_config(args.config, args.output_dir, args.samples)
