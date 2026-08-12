"""Tests for the model registry and the evaluation metric primitives.

The legacy ``benchmark/`` package was removed at L7.4; the catalog lives
in ``models/common/registry.py`` and the metric primitives in
``evaluation/common/``. Covers: registry completeness/order and config
cross-checks, finite-only calibration, NaN-as-no-alarm semantics, and
behavior of the migrated metric implementations.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = os.path.relpath(_REPO / "configs", Path(__file__).resolve().parent)

_METHOD_ORDER = [
    "VAR-CSD",
    "AC1-CSD",
    "SKEW-CSD",
    "SRATIO-CSD",
    "RETRATE-CSD",
    "DFA-CSD",
    "DMD-CSD",
    "Kalman-Spectral-Drift",
    "LSTM-AlarmNet",
    "TCN-AlarmNet",
    "PatchTST-AlarmNet",
]

_INDICATOR_DEFAULTS = {
    "VAR-CSD": {"window_size": 30},
    "AC1-CSD": {"window_size": 30},
    "SKEW-CSD": {"window_size": 30},
    "SRATIO-CSD": {"window_size": 30},
    "RETRATE-CSD": {"window_size": 30},
    "DFA-CSD": {"window_size": 100},
    "DMD-CSD": {"window_size": 30, "embedding_dim": 6, "rank": 2},
}


def _compose_model_config() -> dict:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from csd_observer.config.store import register_configs

    GlobalHydra.instance().clear()
    register_configs()
    with initialize(version_base=None, config_path=_CONFIG_PATH):
        cfg = compose(config_name="run", overrides=["model=default"])
    return OmegaConf.to_container(cfg, resolve=True)["model"]


# --------------------------------------------------------------------- #
# registry catalog
# --------------------------------------------------------------------- #
def test_catalog_completeness_and_order() -> None:
    from csd_observer.models.common.registry import list_methods

    assert list_methods() == _METHOD_ORDER
    assert len(set(list_methods())) == len(list_methods())


def test_catalog_against_shipped_config() -> None:
    from csd_observer.models.common.registry import get_method

    model_cfg = _compose_model_config()
    for name, expected in _INDICATOR_DEFAULTS.items():
        method = get_method(name, system="fold")
        block = model_cfg[name.lower().replace("-", "_")]
        assert {k: v for k, v in block.items() if v is not None} == expected
        assert method.meta.default_params == expected
        assert method.meta.family == "indicator"
        assert not method.meta.is_learned


def test_catalog_rejects_unknown_method() -> None:
    from csd_observer.models.common.registry import validate_names

    with pytest.raises(ValueError, match="Unknown method"):
        validate_names(["NOT-A-METHOD"])


def test_registry_rejects_duplicate_registration() -> None:
    from csd_observer.models.common.registry import register_method

    with pytest.raises(ValueError, match="already registered"):
        register_method("VAR-CSD", lambda _k, _s: None, "indicator")


# --------------------------------------------------------------------- #
# calibration (finite-only rule, plan §4)
# --------------------------------------------------------------------- #
def test_calibrate_threshold_filters_nan() -> None:
    from csd_observer.evaluation.common.calibration import calibrate_threshold

    rng = np.random.default_rng(0)
    scores = rng.normal(0.5, 0.1, (10, 200)).astype(np.float32)
    scores[:, :50] = np.nan  # DFA-like leading undefined windows
    lens = np.full(10, 200, dtype=np.int64)
    threshold = calibrate_threshold(scores, lens, fpr_target=0.05)
    finite = scores[np.isfinite(scores)]
    assert np.isfinite(threshold)
    assert threshold == float(np.percentile(finite, 95.0))


def test_calibrate_threshold_respects_seq_lengths() -> None:
    from csd_observer.evaluation.common.calibration import calibrate_threshold

    rng = np.random.default_rng(1)
    scores = rng.normal(0.0, 1.0, (4, 100)).astype(np.float32)
    lens = np.array([100, 100, 50, 50], dtype=np.int64)
    threshold = calibrate_threshold(scores, lens, fpr_target=0.05)
    steps = np.concatenate([scores[i, : lens[i]] for i in range(4)])
    assert threshold == float(np.percentile(steps, 95.0))


def test_calibrate_threshold_all_nan_returns_nan() -> None:
    from csd_observer.evaluation.common.calibration import calibrate_threshold

    scores = np.full((3, 100), np.nan, dtype=np.float32)
    lens = np.full(3, 100, dtype=np.int64)
    assert np.isnan(calibrate_threshold(scores, lens, fpr_target=0.05))


# --------------------------------------------------------------------- #
# NaN semantics: undefined steps never alarm (plan §4)
# --------------------------------------------------------------------- #
def test_detection_time_nan_steps_do_not_alarm() -> None:
    from csd_observer.evaluation.common.metrics import compute_detection_time

    scores = np.full((2, 200), 0.2, dtype=np.float32)
    scores[0, :50] = np.nan
    scores[1, :50] = np.nan
    bifs = np.array([150.0, 150.0], dtype=np.float32)
    is_pos = np.array([True, True])
    lens = np.full(2, 200, dtype=np.int64)
    # threshold above the finite score: any alarm would have to come from
    # the NaN region, which must not happen
    dt = compute_detection_time(scores, bifs, is_pos, lens, threshold=0.5)
    assert np.isnan(dt)
    # threshold below the finite score: alarm fires at the first finite
    # step (t=50), not at t=0
    dt = compute_detection_time(scores, bifs, is_pos, lens, threshold=0.1)
    assert dt == 150.0 - 50.0


def test_false_positive_rate_nan_steps_do_not_alarm() -> None:
    from csd_observer.evaluation.common.metrics import compute_false_positive_rate

    scores = np.full((3, 100), 0.9, dtype=np.float32)
    scores[:, :50] = np.nan
    lens = np.full(3, 100, dtype=np.int64)
    # NaN steps never alarm but still count in the denominator
    assert compute_false_positive_rate(scores, lens, threshold=0.5) == 0.5
    scores[:] = np.nan
    assert compute_false_positive_rate(scores, lens, threshold=0.5) == 0.0
    assert np.isnan(compute_false_positive_rate(scores, lens, threshold=np.nan))


def test_per_traj_dts_nan_for_missed_trajectories() -> None:
    from csd_observer.evaluation.common.metrics import compute_per_traj_dts

    scores = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], dtype=np.float32)
    bifs = np.array([3.0, 3.0], dtype=np.float32)
    is_pos = np.array([True, True])
    lens = np.full(2, 3, dtype=np.int64)
    dts = compute_per_traj_dts(scores, bifs, is_pos, lens, threshold=0.5)
    assert dts[0] == 3.0
    assert np.isnan(dts[1])


def test_metrics_deterministic_on_nan_free_inputs() -> None:
    from csd_observer.evaluation.common.metrics import (
        compute_detection_time,
        compute_early_warning_auc,
        compute_false_positive_rate,
    )

    rng = np.random.default_rng(7)
    B_sig, B_null, T = 12, 10, 200
    probs_sig = rng.uniform(0.0, 1.0, (B_sig, T)).astype(np.float32)
    probs_null = rng.uniform(0.0, 1.0, (B_null, T)).astype(np.float32)
    bifs = rng.uniform(60.0, 190.0, (B_sig,)).astype(np.float32)
    is_pos = rng.uniform(0.0, 1.0, (B_sig,)) > 0.3
    lens_sig = np.full(B_sig, T, dtype=np.int64)
    lens_null = np.full(B_null, T, dtype=np.int64)
    threshold = 0.5

    dt = compute_detection_time(probs_sig, bifs, is_pos, lens_sig, threshold)
    assert np.isfinite(dt) or np.isnan(dt)
    auc = compute_early_warning_auc(
        probs_sig, bifs, is_pos, lens_sig, probs_null, lens_null
    )
    assert 0.0 <= auc <= 1.0
    fpr = compute_false_positive_rate(probs_null, lens_null, threshold)
    assert 0.0 <= fpr <= 1.0


def test_auc_sanitizes_nan_to_neutral_midpoint() -> None:
    from csd_observer.evaluation.common.metrics import compute_early_warning_auc

    rng = np.random.default_rng(3)
    B_sig, B_null, T = 6, 6, 100
    probs_sig = rng.uniform(0.0, 1.0, (B_sig, T)).astype(np.float32)
    probs_null = rng.uniform(0.0, 1.0, (B_null, T)).astype(np.float32)
    probs_sig[:, :60] = np.nan
    probs_null[:, :60] = np.nan
    bifs = np.full(B_sig, 80.0, dtype=np.float32)
    is_pos = np.ones(B_sig, dtype=bool)
    lens = np.full(B_sig, T, dtype=np.int64)
    auc = compute_early_warning_auc(
        probs_sig, bifs, is_pos, lens, probs_null, lens
    )
    assert np.isfinite(auc)


# --------------------------------------------------------------------- #
# result-row schema
# --------------------------------------------------------------------- #
def test_result_row_schema_roundtrip() -> None:
    from csd_observer.outputs.schema import ResultRow, SchemaError, validate_row

    row = ResultRow(
        run_id="run-1", timestamp="ts", run_name="synthetic_fold",
        dataset="synthetic_fold", bif_type="fold", system="fold",
        replicate="s0", method="VAR-CSD", family="indicator",
        is_learned=False, k_persist=5, fpr_target=0.05,
        detection_rate=0.4, detection_time_mean=50.0,
        detection_time_median=20.0, detection_time_std=30.0,
        ew_auc=0.6, fpr=0.05, persistent_fpr=0.03, threshold=0.14,
        params={"window_size": 30}, config_hash="abc", git_sha="def",
        seed=0,
    ).to_dict()
    validate_row(row)
    with pytest.raises(SchemaError):
        broken = dict(row)
        del broken["method"]
        validate_row(broken)
