"""R5.7: multi-dataset e2e — every synthetic dataset × both evaluations.

Runs the full orchestration pipeline (compose → validate → run →
finalize) on the three synthetic datasets under both evaluation
protocols with tiny overridden dimensions. Real datasets are covered
by compose/validation tests only (data-access gated, R6).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from csd_observer.config.store import register_configs
from csd_observer.config.validate import validate_config

_REPO = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = os.path.relpath(_REPO / "configs", Path(__file__).resolve().parent)

SYNTHETIC_DATASETS = ("synthetic_fold", "synthetic_hopf", "synthetic_logistic")
EVALUATIONS = ("persistenceaware", "baseline_classic")


def _compose(dataset: str, evaluation: str) -> dict:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    register_configs()
    with initialize(version_base=None, config_path=_CONFIG_PATH):
        cfg = compose(
            config_name="run",
            overrides=[
                f"dataset={dataset}",
                f"evaluation={evaluation}",
                "+dataset_overrides.n_trajectories=12",
                "+dataset_overrides.max_length=128",
            ],
        )
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_config(config)
    return config


@pytest.mark.parametrize("dataset", SYNTHETIC_DATASETS)
@pytest.mark.parametrize("evaluation", EVALUATIONS)
def test_synthetic_dataset_e2e(dataset: str, evaluation: str, tmp_path: Path) -> None:
    from csd_observer.orchestration.runner import run_benchmark

    config = _compose(dataset, evaluation)
    config["output"]["base_dir"] = str(tmp_path)
    writer = run_benchmark(config)
    assert writer.status == "completed"

    rows = [
        json.loads(line)
        for line in (writer.root / "results" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["dataset"] == dataset
    assert rows[0]["method"] == "VAR-CSD"
    assert rows[0]["k_persist"] == (1 if evaluation == "baseline_classic" else 5)
    assert rows[0]["bif_type"] in {"fold", "hopf", "logistic"}

    checks = json.loads(
        (writer.root / "metrics" / "protocol_checks.json").read_text(encoding="utf-8")
    )
    assert checks[0]["method"] == "VAR-CSD"
    assert checks[0]["k_persist"] == (1 if evaluation == "baseline_classic" else 5)

    # R4: metrics.json + protocol_checks.csv are written by finalize
    metrics = json.loads(
        (writer.root / "metrics" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["rows"] == 1
    assert (writer.root / "tables" / "protocol_checks.csv").is_file()

    # R2.2: the hopf dataset declares the radial mode; fold/logistic
    # declare channel_0 — the composed dataset block carries it.
    assert writer.root.parent.parent is not None
    resolved = (writer.root / "resolved_config" / "resolved.yaml").read_text(encoding="utf-8")
    if dataset == "synthetic_hopf":
        assert "feature_mode: radial" in resolved
    else:
        assert "feature_mode: channel_0" in resolved
