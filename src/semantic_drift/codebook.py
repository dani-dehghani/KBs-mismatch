from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from semantic_drift.config import ExperimentConfig
from semantic_drift.data import build_dataloaders
from semantic_drift.models import build_model
from semantic_drift.reproducibility import seed_everything
from semantic_drift.training import resolve_device


@dataclass(frozen=True)
class CodebookDriftMetrics:
    mean_cosine_distance: float
    mean_euclidean_distance: float
    max_absolute_difference: float


@dataclass(frozen=True)
class CodebookEvaluation:
    samples: int
    classifier_accuracy: float
    transmitter_accuracy: float
    receiver_accuracy: float
    transmitter_receiver_agreement: float


def _encode(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    encode = getattr(model, "encode", None)
    if not callable(encode):
        raise TypeError("The model must expose an encode(inputs) method.")
    embeddings = encode(inputs)
    if embeddings.ndim != 2:
        raise ValueError("Model embeddings must have shape [batch, embedding_dim].")
    return embeddings


@torch.inference_mode()
def build_class_prototypes(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    *,
    limit_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average deterministic training embeddings per class and L2-normalize them."""
    model.eval()
    sums: torch.Tensor | None = None
    counts = torch.zeros(num_classes, dtype=torch.long)

    for batch_index, (inputs, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        # Split the MPS-to-CPU transfer from float64 conversion. Combining both in one
        # ``to`` call can produce non-finite values with some PyTorch/MPS versions.
        embeddings = _encode(model, inputs.to(device)).detach().to(device="cpu")
        embeddings = embeddings.to(dtype=torch.float64)
        if not torch.isfinite(embeddings).all():
            invalid = (~torch.isfinite(embeddings)).sum().item()
            raise RuntimeError(
                f"Embedding batch {batch_index} contains {invalid} non-finite values."
            )
        targets = targets.detach().to(device="cpu", dtype=torch.long)
        if sums is None:
            sums = torch.zeros((num_classes, embeddings.shape[1]), dtype=torch.float64)
        sums.index_add_(0, targets, embeddings)
        counts += torch.bincount(targets, minlength=num_classes)

    if sums is None:
        raise RuntimeError("No batches were processed while building the codebook.")
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise RuntimeError(f"Cannot build prototypes for classes with no samples: {missing}")

    prototypes = (sums / counts.to(torch.float64).unsqueeze(1)).to(torch.float32)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    if not torch.isfinite(prototypes).all():
        raise RuntimeError("The generated class-prototype codebook contains non-finite values.")
    return prototypes.contiguous(), counts


def clone_aligned_codebooks(codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create equal transmitter/receiver values backed by independent storage."""
    transmitter = codebook.detach().clone().contiguous()
    receiver = codebook.detach().clone().contiguous()
    return transmitter, receiver


def measure_codebook_drift(
    transmitter_codebook: torch.Tensor,
    receiver_codebook: torch.Tensor,
) -> CodebookDriftMetrics:
    if transmitter_codebook.shape != receiver_codebook.shape:
        raise ValueError("Transmitter and receiver codebooks must have the same shape.")
    if torch.equal(transmitter_codebook, receiver_codebook):
        return CodebookDriftMetrics(
            mean_cosine_distance=0.0,
            mean_euclidean_distance=0.0,
            max_absolute_difference=0.0,
        )
    transmitter = transmitter_codebook.detach().to(device="cpu", dtype=torch.float64)
    receiver = receiver_codebook.detach().to(device="cpu", dtype=torch.float64)
    cosine_distance = (1.0 - F.cosine_similarity(transmitter, receiver, dim=1)).clamp_min(0)
    euclidean_distance = torch.linalg.vector_norm(transmitter - receiver, dim=1)
    return CodebookDriftMetrics(
        mean_cosine_distance=cosine_distance.mean().item(),
        mean_euclidean_distance=euclidean_distance.mean().item(),
        max_absolute_difference=(transmitter - receiver).abs().max().item(),
    )


@torch.inference_mode()
def evaluate_aligned_codebooks(
    model: nn.Module,
    loader: DataLoader,
    transmitter_codebook: torch.Tensor,
    receiver_codebook: torch.Tensor,
    device: torch.device,
    *,
    limit_batches: int | None = None,
) -> CodebookEvaluation:
    """Evaluate nearest-prototype transmission and receiver-side decoding."""
    model.eval()
    transmitter = F.normalize(transmitter_codebook.to(device), p=2, dim=1)
    receiver = F.normalize(receiver_codebook.to(device), p=2, dim=1)
    classifier = getattr(model, "classifier", None)
    if not callable(classifier):
        raise TypeError("The model must expose a classifier(embeddings) module.")

    total_samples = 0
    classifier_correct = 0
    transmitter_correct = 0
    receiver_correct = 0
    agreement = 0

    for batch_index, (inputs, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        raw_embeddings = _encode(model, inputs)
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        transmitter_indices = (embeddings @ transmitter.T).argmax(dim=1)
        semantic_messages = transmitter[transmitter_indices]
        receiver_indices = (semantic_messages @ receiver.T).argmax(dim=1)
        classifier_predictions = classifier(raw_embeddings).argmax(dim=1)

        batch_size = targets.shape[0]
        total_samples += batch_size
        classifier_correct += (classifier_predictions == targets).sum().item()
        transmitter_correct += (transmitter_indices == targets).sum().item()
        receiver_correct += (receiver_indices == targets).sum().item()
        agreement += (transmitter_indices == receiver_indices).sum().item()

    if total_samples == 0:
        raise RuntimeError("No batches were processed while evaluating the codebook.")
    return CodebookEvaluation(
        samples=total_samples,
        classifier_accuracy=classifier_correct / total_samples,
        transmitter_accuracy=transmitter_correct / total_samples,
        receiver_accuracy=receiver_correct / total_samples,
        transmitter_receiver_agreement=agreement / total_samples,
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_class_summary(
    path: Path,
    class_names: tuple[str, ...],
    counts: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["class_index", "class_name", "training_samples", "prototype_norm"],
        )
        writer.writeheader()
        norms = torch.linalg.vector_norm(codebook, dim=1)
        for index, class_name in enumerate(class_names):
            writer.writerow(
                {
                    "class_index": index,
                    "class_name": class_name,
                    "training_samples": counts[index].item(),
                    "prototype_norm": norms[index].item(),
                }
            )


def run_aligned_codebook(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Build, clone, verify, evaluate, and save the phase-1 aligned codebooks."""
    seed_everything(config.seed, config.training.deterministic)
    device = resolve_device(config.training.device)
    # A single deterministic loader avoids MPS/fork instability during feature extraction.
    codebook_data_config = replace(config.data, num_workers=0)
    bundle = build_dataloaders(codebook_data_config, config.seed)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # The checkpoint already contains every backbone weight, so avoid fetching pretrained weights.
    model_config = replace(config.model, pretrained=False)
    model = build_model(model_config, bundle.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    canonical_codebook, class_counts = build_class_prototypes(
        model,
        bundle.codebook,
        bundle.num_classes,
        device,
    )
    transmitter_codebook, receiver_codebook = clone_aligned_codebooks(canonical_codebook)
    drift = measure_codebook_drift(transmitter_codebook, receiver_codebook)
    if not torch.equal(transmitter_codebook, receiver_codebook):
        transmitter_finite = torch.isfinite(transmitter_codebook).all().item()
        receiver_finite = torch.isfinite(receiver_codebook).all().item()
        raise RuntimeError(
            "Initial transmitter and receiver codebooks are not exactly equal "
            f"(transmitter_finite={transmitter_finite}, receiver_finite={receiver_finite}, "
            f"max_difference={drift.max_absolute_difference})."
        )
    independent_storage = (
        transmitter_codebook.untyped_storage().data_ptr()
        != receiver_codebook.untyped_storage().data_ptr()
    )
    if not independent_storage:
        raise RuntimeError("Initial codebooks must use independent tensor storage.")

    validation = evaluate_aligned_codebooks(
        model,
        bundle.validation,
        transmitter_codebook,
        receiver_codebook,
        device,
        limit_batches=config.training.limit_eval_batches,
    )
    test = evaluate_aligned_codebooks(
        model,
        bundle.test,
        transmitter_codebook,
        receiver_codebook,
        device,
        limit_batches=config.training.limit_eval_batches,
    )

    if output_directory is None:
        output_directory = Path(config.output_dir) / config.experiment_name / "aligned_codebook"
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": 1,
        "kind": "normalized_class_prototype_codebook",
        "num_classes": bundle.num_classes,
        "embedding_dim": canonical_codebook.shape[1],
        "class_names": list(bundle.class_names),
        "class_counts": class_counts.tolist(),
        "normalization": "l2",
        "similarity": "cosine",
        "num_workers": codebook_data_config.num_workers,
        "seed": config.seed,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_epoch": checkpoint.get("epoch"),
    }
    torch.save(
        {"codebook": canonical_codebook, "metadata": metadata},
        output_directory / "initial_codebook.pt",
    )
    torch.save(
        {"codebook": transmitter_codebook, "metadata": {**metadata, "endpoint": "transmitter"}},
        output_directory / "transmitter_codebook.pt",
    )
    torch.save(
        {"codebook": receiver_codebook, "metadata": {**metadata, "endpoint": "receiver"}},
        output_directory / "receiver_codebook.pt",
    )
    _write_class_summary(
        output_directory / "class_prototypes.csv",
        bundle.class_names,
        class_counts,
        canonical_codebook,
    )

    summary: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "device": str(device),
        "seed": config.seed,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_epoch": checkpoint.get("epoch"),
        "codebook_shape": list(canonical_codebook.shape),
        "class_names": list(bundle.class_names),
        "class_counts": class_counts.tolist(),
        "codebook_sha256": _tensor_sha256(canonical_codebook),
        "transmitter_sha256": _tensor_sha256(transmitter_codebook),
        "receiver_sha256": _tensor_sha256(receiver_codebook),
        "exactly_equal": torch.equal(transmitter_codebook, receiver_codebook),
        "independent_storage": independent_storage,
        "drift": asdict(drift),
        "validation": asdict(validation),
        "test": asdict(test),
        "output_directory": str(output_directory.resolve()),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary
