"""L2.8: governance driver end-to-end with a deterministic dummy method.

Runs the full per-method governance pipeline (fit → score → calibrate →
persist → metrics → row) on a synthetic bundle, and verifies the R2.3
score validator rejects a misbehaving method (Inf buffers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from csd_observer.evaluation.persistence.governance import evaluate_method
from csd_observer.models.common.interface import MethodMeta
from csd_observer.outputs.writer import OutputWriter


class DummyMethod:
    """Deterministic constant-score method (0.5 everywhere)."""

    def __init__(self) -> None:
        self._meta = MethodMeta(
            name="Dummy", family="indicator", is_learned=False,
            scope_caveat="test double",
            default_params={}, config_path=("dummy",),
        )

    @property
    def meta(self) -> MethodMeta:
        return self._meta

    def fit(self, train_arrays: dict[str, Any], val_arrays: dict[str, Any], cfg: dict[str, Any]) -> None:
        pass

    def score(self, features: Any, seq_lengths: Any, cfg: dict[str, Any]) -> np.ndarray:
        return np.full((np.asarray(features).shape[0], np.asarray(features).shape[1]), 0.5, dtype=np.float32)


class InfMethod(DummyMethod):
    """R2.3 violation: Inf in the score buffer."""

    def score(self, features: Any, seq_lengths: Any, cfg: dict[str, Any]) -> np.ndarray:
        out = np.full((np.asarray(features).shape[0], np.asarray(features).shape[1]), 0.5, dtype=np.float32)
        out[:, 0] = np.inf
        return out


class WrongShapeMethod(DummyMethod):
    """R2.3 violation: wrong ndim."""

    def score(self, features: Any, seq_lengths: Any, cfg: dict[str, Any]) -> np.ndarray:
        B = np.asarray(features).shape[0]
        return np.full((B,), 0.5, dtype=np.float32)


def _tiny_bundle(n: int = 12, length: int = 64) -> dict[str, Any]:
    rng = np.random.default_rng(5)
    split = {
        "train": np.arange(0, int(n * 0.6)),
        "val": np.arange(int(n * 0.6), int(n * 0.8)),
        "test": np.arange(int(n * 0.8), n),
    }
    return {
        "features": rng.standard_normal((n, length, 1)).astype(np.float32),
        "seq_lengths": np.full(n, length, dtype=np.int64),
        "bifurcation_times": np.full(n, length * 0.8, dtype=np.float64),
        "is_positive": np.ones(n, dtype=bool),
        "split_indices": split,
    }


@pytest.fixture()
def writer(tmp_path: Path) -> OutputWriter:
    return OutputWriter("test", tmp_path, timestamp="2026-01-01_00-00-00-000000")


def _run_eval(writer: OutputWriter, method: Any) -> dict[str, Any]:
    return evaluate_method(
        method,
        _tiny_bundle(),
        _tiny_bundle(),
        system="fold", bif_type="fold", dataset="synthetic_fold",
        run_name="synthetic_fold", run_id="r1", timestamp="2026-01-01_00-00-00-000000",
        seed=0, replicate="s0",
        config={"model": {}}, writer=writer,
        k_persist=5, fpr_target=0.05,
        evaluation="persistenceaware",
        git_sha="", config_hash="",
    )


def test_dummy_method_produces_valid_row_and_artifacts(writer: OutputWriter) -> None:
    row = _run_eval(writer, DummyMethod())
    assert row["method"] == "Dummy"
    assert row["dataset"] == "synthetic_fold"
    assert np.isfinite(row["fpr"])
    assert np.isfinite(row["ew_auc"])
    assert row["k_persist"] == 5

    checks_path = writer.paths.metrics / "protocol_checks.json"
    assert checks_path.is_file()
    checks = __import__("json").loads(checks_path.read_text(encoding="utf-8"))
    assert checks[0]["null_anchor_upper_bound"] > 0.0
    assert "null_anchor_upper_bound_50step" not in checks[0]
    assert checks[0]["drift"] in {"ok", "PROTOCOL_DRIFT"}

    results = writer.paths.results / "results.jsonl"
    assert results.is_file()
    assert (writer.paths.artifacts / "calibration").is_dir()


def test_inf_scores_rejected_by_score_validator(writer: OutputWriter) -> None:
    with pytest.raises(ValueError, match="Inf"):
        _run_eval(writer, InfMethod())


def test_wrong_shape_scores_rejected(writer: OutputWriter) -> None:
    with pytest.raises(ValueError, match="scores must be"):
        _run_eval(writer, WrongShapeMethod())


def test_baseline_classic_forces_k1(writer: OutputWriter) -> None:
    row = evaluate_method(
        DummyMethod(),
        _tiny_bundle(), _tiny_bundle(),
        system="fold", bif_type="fold", dataset="synthetic_fold",
        run_name="synthetic_fold", run_id="r2", timestamp="2026-01-01_00-00-00-000000",
        seed=0, replicate="s0",
        config={"model": {}}, writer=writer,
        k_persist=5, fpr_target=0.05,
        evaluation="baseline_classic",
        git_sha="", config_hash="",
    )
    assert row["k_persist"] == 1
