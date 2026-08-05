from __future__ import annotations

import argparse
import json

from rnv_seg.workflows import evaluate_from_config, infer_from_config, train_from_config


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train an RNV segmentation model")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train_from_config(args.config))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RNV segmentation checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_from_config(args.config, args.checkpoint), indent=2))


def infer_main() -> None:
    parser = argparse.ArgumentParser(description="Run full-volume sliding-window inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    infer_from_config(args.config, args.checkpoint, args.output_dir)

