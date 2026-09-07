import torch

from semantic_drift.config import ModelConfig
from semantic_drift.models import build_model


def test_tiny_classifier_exposes_embeddings() -> None:
    model = build_model(ModelConfig(name="tiny_cnn", embedding_dim=12), num_classes=4)
    inputs = torch.randn(5, 3, 16, 16)

    embeddings = model.encode(inputs)
    logits = model(inputs)

    assert embeddings.shape == (5, 12)
    assert logits.shape == (5, 4)
