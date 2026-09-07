# Aligned CIFAR-10 class-prototype codebook

## Result

| Field | Value |
| --- | ---: |
| Codebook shape | 10 x 128 |
| Training embeddings used | 45,000 |
| Validation codebook accuracy | **86.56%** |
| Test codebook accuracy | **85.36%** |
| Validation classifier accuracy (checkpoint check) | 88.04% |
| Test classifier accuracy (checkpoint check) | 87.07% |
| Transmitter/receiver agreement | **100.00%** |
| Initial cosine drift | **0.0** |
| Initial Euclidean drift | **0.0** |
| Maximum absolute difference | **0.0** |
| Source checkpoint epoch | 9 |
| Seed | 42 |
| Device | Apple MPS |

## Construction

- Source model: the best validation checkpoint from the clean CIFAR-10 baseline.
- Source data: only the 45,000-image training split; validation and test images
  were never used to construct prototypes.
- Input transform: deterministic resize, center crop, tensor conversion, and
  ImageNet normalization. No random training augmentation was used.
- Representation: the 128-dimensional output of the trained projection head.
- Prototype: the mean training embedding for one class, followed by L2
  normalization.
- Comparison rule: cosine nearest prototype.
- Initial state: one canonical codebook was cloned into separate transmitter and
  receiver tensors. The tensors have independent storage but exactly equal values.
- Update policy: neither codebook was trained or updated in this phase.

The class counts in the deterministic 45,000-image training split are 4,470,
4,517, 4,522, 4,522, 4,507, 4,503, 4,483, 4,482, 4,489, and 4,505. They sum to
45,000.

The codebook tensor SHA-256 is:

```text
144bf74665c951e470b361081082e83645685a94d343b2f873047574ab3c5599
```

The canonical, transmitter, and receiver tensors all have this same checksum.

## Reproduction

```bash
uv run semantic-codebook \
  --config configs/baseline/cifar10.yaml \
  --checkpoint outputs/cifar10-resnet18-pretrained-baseline/best.pt
```

The machine-readable artifacts are stored in:

```text
outputs/cifar10-resnet18-pretrained-baseline/aligned_codebook/
```

The **85.36% test codebook accuracy** is the aligned codebook baseline `L0` for
future drift experiments. It is intentionally reported separately from the
original linear-classifier baseline of 87.07%.
