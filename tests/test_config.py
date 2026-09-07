from pathlib import Path

import pytest

from semantic_drift.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_loads() -> None:
    config = load_config(PROJECT_ROOT / "configs/smoke/synthetic.yaml")

    assert config.experiment_name == "synthetic-smoke"
    assert config.data.name == "synthetic"
    assert config.model.name == "tiny_cnn"
    assert config.training.epochs == 2


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("data:\n  unknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_config(config_path)
