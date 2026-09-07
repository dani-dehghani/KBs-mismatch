from __future__ import annotations

import torch
from torch import nn

from semantic_drift.config import ModelConfig


class TinyClassifier(nn.Module):
    """Small network used only for fast pipeline verification."""

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(32, embedding_dim), nn.ReLU())
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(inputs))


class ResNet18Classifier(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise RuntimeError(
                "ResNet-18 requires torchvision. Install dependencies with `uv sync --extra dev`."
            ) from error

        weights = ResNet18_Weights.DEFAULT if config.pretrained else None
        base = resnet18(weights=weights)
        if config.small_input:
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

        feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.freeze_backbone = config.freeze_backbone
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, config.embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(config.embedding_dim, num_classes)

    def train(self, mode: bool = True) -> ResNet18Classifier:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(inputs))


def build_model(config: ModelConfig, num_classes: int) -> nn.Module:
    name = config.name.lower()
    if name == "tiny_cnn":
        return TinyClassifier(num_classes, config.embedding_dim)
    if name == "resnet18":
        return ResNet18Classifier(config, num_classes)
    raise ValueError(f"Unsupported model: {config.name}")
