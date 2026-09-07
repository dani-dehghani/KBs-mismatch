from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from semantic_drift.config import DataConfig


@dataclass(frozen=True)
class DataBundle:
    train: DataLoader
    codebook: DataLoader
    validation: DataLoader
    test: DataLoader
    num_classes: int
    class_names: tuple[str, ...]


def _loader(
    dataset: Dataset,
    config: DataConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def _synthetic_dataset(
    size: int,
    prototypes: torch.Tensor,
    *,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(len(prototypes), (size,), generator=generator)
    noise = torch.randn((size, *prototypes.shape[1:]), generator=generator) * 0.25
    images = prototypes[labels] + noise
    return TensorDataset(images, labels)


def build_synthetic_loaders(config: DataConfig, seed: int) -> DataBundle:
    if min(config.train_size, config.val_size, config.test_size) <= 0:
        raise ValueError("Synthetic dataset sizes must be positive.")
    if config.num_classes <= 1:
        raise ValueError("Synthetic data requires at least two classes.")

    generator = torch.Generator().manual_seed(seed)
    shape = (config.num_classes, 3, config.image_size, config.image_size)
    prototypes = torch.randn(shape, generator=generator)

    train = _synthetic_dataset(config.train_size, prototypes, seed=seed + 1)
    validation = _synthetic_dataset(config.val_size, prototypes, seed=seed + 2)
    test = _synthetic_dataset(config.test_size, prototypes, seed=seed + 3)
    return DataBundle(
        train=_loader(train, config, shuffle=True, seed=seed + 4),
        codebook=_loader(train, config, shuffle=False, seed=seed + 5),
        validation=_loader(validation, config, shuffle=False, seed=seed + 6),
        test=_loader(test, config, shuffle=False, seed=seed + 7),
        num_classes=config.num_classes,
        class_names=tuple(f"class_{index}" for index in range(config.num_classes)),
    )


def build_cifar10_loaders(config: DataConfig, seed: int) -> DataBundle:
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError(
            "CIFAR-10 requires torchvision. Install the project dependencies with "
            "`uv sync --extra dev`."
        ) from error

    normalization = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }
    mean, std = normalization[config.normalization]

    if config.image_size == 32:
        train_geometry = [transforms.RandomCrop(32, padding=4)]
        evaluation_geometry = []
    else:
        train_geometry = [
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        ]
        evaluation_geometry = [
            transforms.Resize(round(config.image_size * 256 / 224)),
            transforms.CenterCrop(config.image_size),
        ]

    train_transform = transforms.Compose(
        [
            *train_geometry,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [*evaluation_geometry, transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    augmented_train = datasets.CIFAR10(
        config.root,
        train=True,
        transform=train_transform,
        download=config.download,
    )
    evaluation_train = datasets.CIFAR10(
        config.root,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test = datasets.CIFAR10(
        config.root,
        train=False,
        transform=evaluation_transform,
        download=config.download,
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(augmented_train), generator=generator).tolist()
    validation_size = int(len(indices) * config.val_fraction)
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    if not validation_indices:
        raise ValueError("data.val_fraction creates an empty CIFAR-10 validation split.")

    train = Subset(augmented_train, train_indices)
    codebook = Subset(evaluation_train, train_indices)
    validation = Subset(evaluation_train, validation_indices)
    return DataBundle(
        train=_loader(train, config, shuffle=True, seed=seed + 1),
        codebook=_loader(codebook, config, shuffle=False, seed=seed + 2),
        validation=_loader(validation, config, shuffle=False, seed=seed + 3),
        test=_loader(test, config, shuffle=False, seed=seed + 4),
        num_classes=10,
        class_names=tuple(augmented_train.classes),
    )


def build_dataloaders(config: DataConfig, seed: int) -> DataBundle:
    name = config.name.lower()
    if name == "synthetic":
        return build_synthetic_loaders(config, seed)
    if name == "cifar10":
        return build_cifar10_loaders(config, seed)
    raise ValueError(f"Unsupported dataset: {config.name}")
