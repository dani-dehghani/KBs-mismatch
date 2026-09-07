from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_drift.config import load_config
from semantic_drift.training import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a semantic-drift experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--device", help="Override the configured device (auto/cpu/cuda/mps).")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs.")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.training.device = args.device
    if args.epochs is not None:
        if args.epochs <= 0:
            raise SystemExit("--epochs must be positive.")
        config.training.epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
