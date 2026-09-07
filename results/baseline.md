# Clean CIFAR-10 baseline

## Result

| Field | Value |
| --- | ---: |
| Test accuracy | **87.07%** |
| Test cross-entropy loss | 0.831703 |
| Best validation accuracy | 88.04% |
| Best epoch | 9 of 10 |
| Seed | 42 |
| Device | Apple MPS |
| Training duration | 1,066.14 seconds |

## Run definition

- Dataset: CIFAR-10, with 45,000 train, 5,000 validation, and 10,000 test images
- Backbone: torchvision ResNet-18 with official ImageNet weights
  (`ResNet18_Weights.DEFAULT`)
- Backbone policy: frozen, including BatchNorm running statistics
- Input: 224 x 224, ImageNet normalization
- Train augmentation: random resized crop and horizontal flip
- Head: 512 -> 128 projection with ReLU, then a 10-class linear classifier
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0005
- Scheduler: cosine annealing over 10 epochs
- Loss: cross-entropy with 0.1 label smoothing

## Reproduction

```bash
uv sync --extra dev
uv run semantic-drift --config configs/baseline/cifar10.yaml
```

The resolved configuration, per-epoch metrics, best checkpoint, and machine-
readable summary are stored in:

```text
outputs/cifar10-resnet18-pretrained-baseline/
```

`outputs/`, `data/`, and model checkpoints are intentionally ignored by Git.
This tracked report records the canonical clean-baseline number used by later
drift and realignment comparisons.
