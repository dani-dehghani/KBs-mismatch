from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_drift.codebook import run_aligned_codebook
from semantic_drift.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate identical initial transmitter/receiver codebooks."
    )
    parser.add_argument("--config", type=Path, required=True, help="Baseline YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trained baseline checkpoint; defaults to its standard output path.",
    )
    parser.add_argument("--device", help="Override the configured device (auto/cpu/cuda/mps).")
    parser.add_argument("--output-dir", type=Path, help="Exact codebook artifact directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.training.device = args.device
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(config.output_dir) / config.experiment_name / "best.pt"

    summary = run_aligned_codebook(
        config,
        checkpoint,
        output_directory=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
