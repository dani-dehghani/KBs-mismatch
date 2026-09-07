from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    name: str = "cifar10"
    root: str = "data"
    batch_size: int = 128
    num_workers: int = 4
    val_fraction: float = 0.1
    download: bool = True
    train_size: int = 128
    val_size: int = 64
    test_size: int = 64
    num_classes: int = 10
    image_size: int = 32
    normalization: str = "cifar10"


@dataclass
class ModelConfig:
    name: str = "resnet18"
    embedding_dim: int = 128
    pretrained: bool = False
    freeze_backbone: bool = False
    small_input: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    device: str = "auto"
    deterministic: bool = True
    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None


@dataclass
class ExperimentConfig:
    experiment_name: str = "baseline"
    seed: int = 42
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ConfigSection = TypeVar("ConfigSection")


def _build_section(section_type: type[ConfigSection], values: Any, name: str) -> ConfigSection:
    if values is None:
        return section_type()
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")

    valid_keys = {item.name for item in fields(section_type)}
    unknown_keys = set(values) - valid_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown key(s) in configuration section '{name}': {unknown}")
    return section_type(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if not isinstance(raw, dict):
        raise ValueError("The experiment configuration must be a mapping.")

    valid_top_level = {"experiment_name", "seed", "output_dir", "data", "model", "training"}
    unknown_keys = set(raw) - valid_top_level
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown top-level configuration key(s): {unknown}")

    config = ExperimentConfig(
        experiment_name=raw.get("experiment_name", "baseline"),
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "outputs"),
        data=_build_section(DataConfig, raw.get("data"), "data"),
        model=_build_section(ModelConfig, raw.get("model"), "model"),
        training=_build_section(TrainingConfig, raw.get("training"), "training"),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative.")
    if not 0 <= config.data.val_fraction < 1:
        raise ValueError("data.val_fraction must be in [0, 1).")
    if config.data.image_size <= 0:
        raise ValueError("data.image_size must be positive.")
    if config.data.normalization not in {"cifar10", "imagenet"}:
        raise ValueError("data.normalization must be either 'cifar10' or 'imagenet'.")
    if config.model.embedding_dim <= 0:
        raise ValueError("model.embedding_dim must be positive.")
    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    if config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive.")
