from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from semantic_drift.config import ExperimentConfig
from semantic_drift.data import build_dataloaders
from semantic_drift.models import build_model
from semantic_drift.reproducibility import seed_everything


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    limit_batches: int | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (inputs, targets) in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No batches were processed; check the dataset and batch limits.")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    seed_everything(config.seed, config.training.deterministic)
    device = resolve_device(config.training.device)
    bundle = build_dataloaders(config.data, config.seed)
    model = build_model(config.model, bundle.num_classes).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("The model has no trainable parameters.")

    optimizer = AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

    run_directory = Path(config.output_dir) / config.experiment_name
    run_directory.mkdir(parents=True, exist_ok=True)
    _write_json(run_directory / "config.json", config.to_dict())

    metrics_path = run_directory / "metrics.csv"
    best_accuracy = float("-inf")
    best_epoch = 0
    started_at = time.perf_counter()

    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, config.training.epochs + 1):
            train_metrics = _run_epoch(
                model,
                bundle.train,
                criterion,
                device,
                optimizer=optimizer,
                limit_batches=config.training.limit_train_batches,
            )
            validation_metrics = _run_epoch(
                model,
                bundle.validation,
                criterion,
                device,
                limit_batches=config.training.limit_eval_batches,
            )
            current_learning_rate = optimizer.param_groups[0]["lr"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "learning_rate": current_learning_rate,
                    "train_loss": train_metrics.loss,
                    "train_accuracy": train_metrics.accuracy,
                    "validation_loss": validation_metrics.loss,
                    "validation_accuracy": validation_metrics.accuracy,
                }
            )
            stream.flush()
            print(
                f"epoch={epoch:02d}/{config.training.epochs:02d} "
                f"train_loss={train_metrics.loss:.4f} "
                f"train_accuracy={train_metrics.accuracy:.4f} "
                f"validation_loss={validation_metrics.loss:.4f} "
                f"validation_accuracy={validation_metrics.accuracy:.4f}",
                flush=True,
            )

            if validation_metrics.accuracy > best_accuracy:
                best_accuracy = validation_metrics.accuracy
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "validation_accuracy": validation_metrics.accuracy,
                        "config": config.to_dict(),
                    },
                    run_directory / "best.pt",
                )
            scheduler.step()

    checkpoint = torch.load(run_directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = _run_epoch(
        model,
        bundle.test,
        criterion,
        device,
        limit_batches=config.training.limit_eval_batches,
    )

    summary: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "duration_seconds": time.perf_counter() - started_at,
        "run_directory": str(run_directory.resolve()),
    }
    _write_json(run_directory / "summary.json", summary)
    return summary
