# Aligned codebook report working notes

## Reporting job

- Audience: technical (student supervisor / research review)
- Question: Was a reproducible, initially aligned semantic codebook produced for
  the CIFAR-10 transmitter-receiver system?
- Decision-useful answer: yes; the 10 x 128 codebook is equal at both endpoints,
  achieves 85.36% test accuracy, and provides the zero-drift reference for the
  next phase.
- Comparison basis: original linear classifier versus cosine nearest-prototype
  decoding on the same epoch-9 checkpoint and fixed validation/test splits.

## Required-structure map

- Technical summary: page 1
- Key findings and visual evidence: pages 3-4
- Scope, data, and definitions: pages 1-2
- Methodology and model specification: page 2
- Limitations and robustness checks: pages 4-5
- Recommended next steps and further questions: page 5

## Chart map

| Section | Question | Form | Fields | Supported claim | Palette |
| --- | --- | --- | --- | --- | --- |
| Performance evidence | What accuracy cost does prototype decoding introduce? | Grouped vertical bar | split, classifier accuracy, codebook accuracy, n | Codebook retains 98.04% of test classifier accuracy | Hard two-root: teal and orange |
| Prototype geometry | Are the class prototypes distinct and structured? | 10 x 10 heatmap | class pair, cosine similarity | Off-diagonal similarity ranges from 0.319 to 0.872 | Single-root blue scale |

Both charts are rendered directly inside the final PDF and inspected in page context.

## Evidence sources

- `outputs/cifar10-resnet18-pretrained-baseline/summary.json`
- `outputs/cifar10-resnet18-pretrained-baseline/aligned_codebook/summary.json`
- `outputs/cifar10-resnet18-pretrained-baseline/aligned_codebook/class_prototypes.csv`
- `outputs/cifar10-resnet18-pretrained-baseline/aligned_codebook/initial_codebook.pt`
- `src/semantic_drift/codebook.py`

## Interpretation constraints

- The result is descriptive and based on one random seed.
- The codebook is a single-prototype-per-class baseline, not a claim of optimal
  CIFAR-10 accuracy.
- No drift, channel noise, detector, or realignment method is evaluated in this phase.
