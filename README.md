# KBs Mismatch: Semantic Codebook Drift

Research code and reports for **Keeping Meaning Aligned on a Budget: Drift
Detection and Realignment for AI-Native Communication**.

The project studies what happens when a transmitter and receiver start with the
same semantic codebook but later learn from different local data streams. The
current repository establishes the clean image-classification baseline and the
shared transmitter/receiver codebook that future drift experiments will use as
their aligned reference state.

## Status

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | CIFAR-10 ResNet-18 classifier baseline | Complete |
| 2 | Identical initial transmitter/receiver codebooks | Complete |
| 3 | Abstract channel and asymmetric local updates | Planned |
| 4 | Drift detection and realignment methods | Planned |

No semantic drift is introduced in the current implementation. The two saved
codebooks have independent storage but exactly equal initial values.

## Main results

| Metric | Result |
| --- | ---: |
| Best validation classifier accuracy | **88.04%** |
| Test classifier accuracy | **87.07%** |
| Validation codebook accuracy | **86.56%** |
| Test codebook accuracy (`L0`) | **85.36%** |
| Transmitter/receiver agreement | **100.00%** |
| Initial cosine drift | **0.0** |
| Initial Euclidean drift | **0.0** |

These values are from the canonical seed-42 run. The best classifier checkpoint
was selected at epoch 9 of 10.

## Method overview

1. CIFAR-10 is split deterministically into 45,000 training, 5,000 validation,
   and 10,000 test images.
2. An ImageNet-pretrained ResNet-18 is used as a frozen feature extractor.
3. A trainable head maps the 512-dimensional backbone feature to a
   128-dimensional representation and then to 10 class logits.
4. After classifier training, one normalized prototype is computed for each
   CIFAR-10 class by averaging only its training-set representations.
5. The resulting `10 x 128` tensor is cloned into independent transmitter and
   receiver codebook files.
6. Codebook prediction selects the class prototype with maximum cosine
   similarity to the image representation.

The classifier/codebook accuracy gap is not semantic drift. It comes from using
two different decision rules: a trained linear classifier versus nearest class
prototype decoding.

## Repository layout

```text
configs/
  baseline/cifar10.yaml        canonical CIFAR-10 experiment
  smoke/synthetic.yaml         fast, download-free pipeline check
src/semantic_drift/
  cli.py                       classifier training command
  codebook_cli.py              aligned-codebook command
  config.py                    typed YAML configuration loader
  data.py                      CIFAR-10 and synthetic data loaders
  models.py                    ResNet-18 and smoke-test models
  training.py                  training, evaluation, and checkpoint logging
  codebook.py                  prototype construction and aligned evaluation
tests/                         unit and end-to-end smoke tests
scripts/                       PDF report builders
results/                       tracked experiment summaries
phase1/                        baseline reports
phase2/                        aligned-codebook reports
output/pdf/                    generated English and Persian reports
Nafas_Mohebi_work-1.pdf        original project brief
```

The recorded CIFAR-10 files, model checkpoint, and codebook tensors are included
through Git Large File Storage (Git LFS). Virtual environments, caches, and
temporary renders remain excluded because they are machine-specific and can be
recreated from `uv.lock`.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended)
- [Git LFS](https://git-lfs.com/)
- Internet access for the first CIFAR-10 and pretrained-weight download
- CPU, CUDA GPU, or Apple Silicon MPS

The full baseline resizes CIFAR-10 images to `224 x 224`. CPU execution is
supported but can be slow. The recorded run used Apple MPS.

## Installation

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/dani-dehghani/KBs-mismatch.git
cd KBs-mismatch
git lfs install
git lfs pull
uv sync --extra dev --extra report
```

Activate the environment only if you want to run commands without `uv run`:

```bash
source .venv/bin/activate
```

### Installation with pip

If `uv` is not available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,report]'
```

### Optional local macOS environment snapshot

The repository's Releases page also contains a compressed snapshot of the
original Apple Silicon development environment, caches, and temporary report
renders. This archive is not required to run the project and is not portable to
Linux, Windows, or Intel macOS. Prefer `uv sync` for normal installation.

See [`LOCAL_ENVIRONMENT.md`](LOCAL_ENVIRONMENT.md) for the download link,
contents, checksum, and restoration instructions.

## Quick verification

Run the synthetic smoke experiment. It exercises the complete software pipeline
without downloading CIFAR-10:

```bash
uv run semantic-drift --config configs/smoke/synthetic.yaml
uv run pytest
uv run ruff check .
```

The synthetic dataset is only a software check and must not be reported as a
research result.

## Reproduce the CIFAR-10 classifier baseline

The repository includes the recorded CIFAR-10 data through Git LFS. If those
objects were not pulled, torchvision downloads CIFAR-10 automatically. The
official torchvision ImageNet weights are obtained if they are not cached:

```bash
uv run semantic-drift --config configs/baseline/cifar10.yaml
```

The command trains for 10 epochs, selects the checkpoint with the highest
validation accuracy, evaluates it once on the test set, and writes:

```text
outputs/cifar10-resnet18-pretrained-baseline/
  config.json       resolved experiment configuration
  metrics.csv       train/validation metrics for every epoch
  best.pt           best-validation model checkpoint
  summary.json      best epoch and final test metrics
```

Useful overrides:

```bash
# Force CPU execution
uv run semantic-drift --config configs/baseline/cifar10.yaml --device cpu

# Short diagnostic run
uv run semantic-drift --config configs/baseline/cifar10.yaml --epochs 1

# Write artifacts to another directory
uv run semantic-drift \
  --config configs/baseline/cifar10.yaml \
  --output-dir /path/to/experiment-output
```

See [`results/baseline.md`](results/baseline.md) for the canonical run definition
and recorded result.

## Build and evaluate the aligned codebooks

After the classifier checkpoint exists, build the class prototypes:

```bash
uv run semantic-codebook \
  --config configs/baseline/cifar10.yaml \
  --checkpoint outputs/cifar10-resnet18-pretrained-baseline/best.pt
```

The command uses deterministic evaluation transforms and only the 45,000-image
training split to create the prototypes. It then verifies exact tensor equality,
independent storage, transmitter/receiver agreement, classifier accuracy, and
nearest-prototype accuracy.

Generated artifacts, including the recorded run committed through Git LFS:

```text
outputs/cifar10-resnet18-pretrained-baseline/aligned_codebook/
  initial_codebook.pt        canonical normalized prototypes
  transmitter_codebook.pt    independent transmitter copy
  receiver_codebook.pt       independent receiver copy
  class_prototypes.csv       class counts and prototype norms
  summary.json               equality, drift, and accuracy metrics
```

See [`results/aligned_codebook.md`](results/aligned_codebook.md) for the complete
tracked experiment record.

## Generate the PDF reports

Run the experiments first so the report builders can read their metrics and
artifacts. Then execute:

```bash
uv run python scripts/build_baseline_report.py
uv run python scripts/build_baseline_explanation_fa.py
uv run python scripts/build_aligned_codebook_report.py
uv run python scripts/build_aligned_codebook_explanation_fa.py
```

The generated files are written to `output/pdf/`. Ready-to-read copies are also
included in `phase1/` and `phase2/`:

- Phase 1: English baseline report and Persian explanation
- Phase 2: English aligned-codebook report and Persian explanation

## Configuration

Experiments are controlled by YAML files. The canonical baseline configuration
defines:

- deterministic seed and train/validation split
- dataset roots and transforms
- pretrained model and frozen-backbone policy
- projection dimension and class count
- optimizer, learning rate, weight decay, and scheduler
- epoch count, batch size, device policy, and output directory

Create a new YAML file under `configs/` rather than editing a reported canonical
configuration in place.

## Testing and code quality

```bash
uv run pytest
uv run ruff check .
```

The GitHub Actions workflow runs these checks automatically on pushes and pull
requests. Tests use the synthetic dataset and do not download CIFAR-10.

## Reproducibility notes

- Python dependencies are locked in `uv.lock`.
- Random generators and deterministic data splits are seeded.
- Validation selects the checkpoint; the test set is reserved for final
  evaluation.
- Codebook prototypes use training embeddings only.
- The recorded raw outputs, checkpoint, and dataset are stored with Git LFS;
  future runs should only be committed when they represent a deliberate result.
- The current numerical results are a single-seed reference, not a variance
  estimate. Future research runs should report 3-5 seeds.

## Roadmap

1. CIFAR-10 classifier baseline - complete
2. Shared initial transmitter/receiver codebooks - complete
3. Abstract task-message channel
4. Asymmetric transmitter and receiver updates
5. Drift metrics and no-alignment/full-synchronization bounds
6. Prototype, anchor/Procrustes, and adapter realignment
7. Receiver-side detection and cost-aware triggering
8. Multi-seed experiment and plotting pipeline

## فارسی - راه‌اندازی سریع

برای نصب، آزمایش سریع، آموزش مدل و ساخت کدبوک به‌ترتیب اجرا کنید:

```bash
uv sync --extra dev --extra report
git lfs pull
uv run semantic-drift --config configs/smoke/synthetic.yaml
uv run semantic-drift --config configs/baseline/cifar10.yaml
uv run semantic-codebook \
  --config configs/baseline/cifar10.yaml \
  --checkpoint outputs/cifar10-resnet18-pretrained-baseline/best.pt
```

در وضعیت فعلی هنوز رانش ایجاد نشده است. فاصله کسینوسی و اقلیدسی کدبوک فرستنده
و گیرنده صفر و توافق آن‌ها 100 درصد است. دقت `85.36%` نقطه مرجع کدبوک برای
آزمایش‌های رانش مراحل بعدی است.
