import csv
from pathlib import Path

from semantic_drift.codebook import run_aligned_codebook
from semantic_drift.config import load_config
from semantic_drift.training import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_experiment_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs/smoke/synthetic.yaml")
    config.output_dir = str(tmp_path)
    config.training.epochs = 1
    config.training.limit_train_batches = 2
    config.training.limit_eval_batches = 1

    summary = run_experiment(config)
    run_directory = tmp_path / config.experiment_name

    assert 0.0 <= summary["test_accuracy"] <= 1.0
    assert (run_directory / "config.json").exists()
    assert (run_directory / "best.pt").exists()
    assert (run_directory / "summary.json").exists()

    with (run_directory / "metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1

    codebook_summary = run_aligned_codebook(config, run_directory / "best.pt")
    codebook_directory = run_directory / "aligned_codebook"
    assert codebook_summary["codebook_shape"] == [config.data.num_classes, 32]
    assert codebook_summary["exactly_equal"] is True
    assert codebook_summary["independent_storage"] is True
    assert codebook_summary["drift"]["mean_cosine_distance"] == 0.0
    assert codebook_summary["drift"]["mean_euclidean_distance"] == 0.0
    assert (codebook_directory / "initial_codebook.pt").exists()
    assert (codebook_directory / "transmitter_codebook.pt").exists()
    assert (codebook_directory / "receiver_codebook.pt").exists()
    assert (codebook_directory / "class_prototypes.csv").exists()
    assert (codebook_directory / "summary.json").exists()
