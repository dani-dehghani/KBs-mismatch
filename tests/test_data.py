import torch

from semantic_drift.config import DataConfig
from semantic_drift.data import build_synthetic_loaders


def test_synthetic_data_is_reproducible() -> None:
    config = DataConfig(
        name="synthetic",
        batch_size=8,
        num_workers=0,
        train_size=16,
        val_size=8,
        test_size=8,
        num_classes=3,
        image_size=8,
    )
    first = build_synthetic_loaders(config, seed=9)
    second = build_synthetic_loaders(config, seed=9)

    first_inputs, first_targets = next(iter(first.validation))
    second_inputs, second_targets = next(iter(second.validation))
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)
