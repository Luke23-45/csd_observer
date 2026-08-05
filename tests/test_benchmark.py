"""Tests for the benchmark suite: catalog, evaluation governance, and
end-to-end orchestration (``csd_observer.benchmark``).

Covers plan §7 items 6-7: the finite-only calibration rule (DFA NaN
handling), NaN-as-no-alarm semantics, parity with the legacy metric
implementations on NaN-free inputs, seed determinism, and a full-suite
smoke run emitting one row per (system, method).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from csd_observer.config.load import load_config
from csd_observer.utils.io import OutputWriter

_METHOD_ORDER = [
    "Kalman-Spectral-Drift",
    "VAR-CSD",
    "AC1-CSD",
    "SKEW-CSD",
    "SRATIO-CSD",
    "DFA-CSD",
    "RETRATE-CSD",
    "DMD-CSD",
]


# --------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------- #
def test_catalog_completeness_and_order() -> None:
    from csd_observer.benchmark.methods import METHODS

    assert [spec.name for spec in METHODS] == _METHOD_ORDER
    assert len({spec.name for spec in METHODS}) == len(METHODS)
    for spec in METHODS:
        assert spec.family in ("indicator", "spectral")
        assert spec.config_path


def test_catalog_against_shipped_config() -> None:
    from csd_observer.benchmark.methods import METHODS, resolve_params, validate_catalog

    model_cfg = load_config("default")["model"]
    validate_catalog(model_cfg)
    expected = {
        "VAR-CSD": {"window_size": 30},
        "AC1-CSD": {"window_size": 30},
        "SKEW-CSD": {"window_size": 30},
        "SRATIO-CSD": {"window_size": 30},
        "RETRATE-CSD": {"window_size": 30},
        "DFA-CSD": {"window_size": 100},
        "DMD-CSD": {"window_size": 30, "embedding_dim": 6, "rank": 2},
    }
    for spec in METHODS:
        if spec.family != "indicator":
            continue
        assert resolve_params(spec, model_cfg) == expected[spec.name]


def test_catalog_rejects_unknown_method() -> None:
    from csd_observer.benchmark.methods import get_method

    with pytest.raises(ValueError, match="Unknown method"):
        get_method("NOT-A-METHOD")


def test_catalog_cross_check_fails_fast() -> None:
    from csd_observer.benchmark.methods import validate_catalog

    model_cfg = load_config("default")["model"]

    missing = {k: v for k, v in model_cfg["csd_indicators"].items() if k != "var"}
    broken = dict(model_cfg)
    broken["csd_indicators"] = missing
    with pytest.raises(KeyError, match="VAR-CSD"):
        validate_catalog(broken)

    unknown = dict(model_cfg)
    unknown["csd_indicators"] = dict(model_cfg["csd_indicators"], not_a_method={"window_size": 30})
    with pytest.raises(ValueError, match="not_a_method"):
        validate_catalog(unknown)


# --------------------------------------------------------------------- #
# calibration (finite-only rule, plan §4)
# --------------------------------------------------------------------- #
def test_calibrate_threshold_filters_nan() -> None:
    from csd_observer.utils.evaluation import calibrate_threshold

    rng = np.random.default_rng(0)
    scores = rng.normal(0.5, 0.1, (10, 200)).astype(np.float32)
    scores[:, :50] = np.nan  # DFA-like leading undefined windows
    lens = np.full(10, 200, dtype=np.int64)
    threshold = calibrate_threshold(scores, lens, fpr_target=0.05)
    finite = scores[np.isfinite(scores)]
    assert np.isfinite(threshold)
    assert threshold == float(np.percentile(finite, 95.0))


def test_calibrate_threshold_respects_seq_lengths() -> None:
    from csd_observer.utils.evaluation import calibrate_threshold

    rng = np.random.default_rng(1)
    scores = rng.normal(0.0, 1.0, (4, 100)).astype(np.float32)
    lens = np.array([100, 100, 50, 50], dtype=np.int64)
    threshold = calibrate_threshold(scores, lens, fpr_target=0.05)
    steps = np.concatenate([scores[i, : lens[i]] for i in range(4)])
    assert threshold == float(np.percentile(steps, 95.0))


def test_calibrate_threshold_all_nan_returns_nan() -> None:
    from csd_observer.utils.evaluation import calibrate_threshold

    scores = np.full((3, 100), np.nan, dtype=np.float32)
    lens = np.full(3, 100, dtype=np.int64)
    assert np.isnan(calibrate_threshold(scores, lens, fpr_target=0.05))


# --------------------------------------------------------------------- #
# NaN semantics: undefined steps never alarm (plan §4)
# --------------------------------------------------------------------- #
def test_detection_time_nan_steps_do_not_alarm() -> None:
    from csd_observer.utils.evaluation import compute_detection_time

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
    from csd_observer.utils.evaluation import compute_false_positive_rate

    scores = np.full((3, 100), 0.9, dtype=np.float32)
    scores[:, :50] = np.nan
    lens = np.full(3, 100, dtype=np.int64)
    # NaN steps never alarm but still count in the denominator
    assert compute_false_positive_rate(scores, lens, threshold=0.5) == 0.5
    scores[:] = np.nan
    assert compute_false_positive_rate(scores, lens, threshold=0.5) == 0.0
    assert np.isnan(compute_false_positive_rate(scores, lens, threshold=np.nan))


def test_per_traj_dts_nan_for_missed_trajectories() -> None:
    from csd_observer.utils.evaluation import compute_per_traj_dts

    scores = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], dtype=np.float32)
    bifs = np.array([3.0, 3.0], dtype=np.float32)
    is_pos = np.array([True, True])
    lens = np.full(2, 3, dtype=np.int64)
    dts = compute_per_traj_dts(scores, bifs, is_pos, lens, threshold=0.5)
    assert dts[0] == 3.0
    assert np.isnan(dts[1])


# --------------------------------------------------------------------- #
# parity with the legacy metric implementations on NaN-free inputs
# --------------------------------------------------------------------- #
def test_metrics_parity_with_legacy_implementations() -> None:
    from csd_observer.utils.evaluation import (
        compute_detection_time,
        compute_early_warning_auc,
        compute_false_positive_rate,
    )
    from csd_observer.utils.metrics import (
        compute_detection_time as legacy_dt,
    )
    from csd_observer.utils.metrics import (
        compute_early_warning_auc as legacy_auc,
    )
    from csd_observer.utils.metrics import (
        compute_false_positive_rate as legacy_fpr,
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

    assert compute_detection_time(probs_sig, bifs, is_pos, lens_sig, threshold) == legacy_dt(
        probs_sig, bifs, is_pos, lens_sig, threshold
    )
    assert compute_early_warning_auc(
        probs_sig, bifs, is_pos, lens_sig, probs_null, lens_null
    ) == legacy_auc(probs_sig, bifs, is_pos, lens_sig, probs_null, lens_null)
    assert compute_false_positive_rate(probs_null, lens_null, threshold) == legacy_fpr(
        probs_null, lens_null, threshold
    )


def test_auc_sanitizes_nan_to_neutral_midpoint() -> None:
    from csd_observer.utils.evaluation import compute_early_warning_auc

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
# end-to-end suite
# --------------------------------------------------------------------- #
def _tiny_suite_kwargs(tmp_path, name: str, n_patients: int = 32) -> dict:
    return dict(
        n_seeds=1,
        data_overrides={"n_patients": n_patients},
        model_overrides={
            "spectral_drift": {
                "n_particles": 100,
                "c_min": 1e-3,
                "delta": 0.05,
                "center_window": 50,
                "q_drift_grid": [1e-3],
                "sigma_u_grid": [0.3],
                "fpr_target": 0.05,
            }
        },
        writer=OutputWriter(experiment_name=name, base_dir=tmp_path),
    )


def _read_rows(writer: OutputWriter) -> list[dict]:
    with (writer.path / "results" / "results.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_suite_smoke_all_methods(tmp_path) -> None:
    from csd_observer.benchmark.suite import run_config

    kwargs = _tiny_suite_kwargs(tmp_path, "smoke_a")
    all_metrics = run_config("default", **kwargs)
    rows = _read_rows(kwargs["writer"])

    systems = ["fold", "hopf", "logistic"]
    assert sorted({r["system"] for r in rows}) == systems
    assert sorted({r["method"] for r in rows}) == sorted(_METHOD_ORDER)
    assert len(rows) == 3 * len(_METHOD_ORDER)
    for row in rows:
        assert np.isfinite(row["ew_auc"]), row
        assert 0.0 <= row["fpr"] <= 1.0 + 1e-9, row
    # every indicator row carries its resolved window size
    for row in rows:
        if row["method"] == "DFA-CSD":
            assert row["window_size"] == 100
        elif row["method"] == "DMD-CSD":
            assert row["window_size"] == 30 and row["rank"] == 2
        elif row["method"] != "Kalman-Spectral-Drift":
            assert row["window_size"] == 30
    # aggregated metrics per system per method
    assert all_metrics["fold"]["VAR-CSD"]["ew_auc"] > 0.0
    assert all_metrics["fold"]["DFA-CSD"]["ew_auc"] > 0.0
    # threshold calibrates FPR to ~5% on val nulls by construction;
    # per-seed test-null FPR carries sampling noise at this sample size
    for system in systems:
        for method in _METHOD_ORDER:
            assert 0.0 <= all_metrics[system][method]["fpr"] <= 0.2


def test_suite_method_filter(tmp_path) -> None:
    from csd_observer.benchmark.suite import run_config

    kwargs = _tiny_suite_kwargs(tmp_path, "smoke_filter", n_patients=16)
    run_config(
        "default",
        enabled_methods={"AC1-CSD", "DMD-CSD"},
        **kwargs,
    )
    rows = _read_rows(kwargs["writer"])
    assert sorted({r["method"] for r in rows}) == ["AC1-CSD", "DMD-CSD"]
    assert len(rows) == 3 * 2


def test_suite_seed_determinism(tmp_path) -> None:
    from csd_observer.benchmark.suite import run_config

    a_kwargs = _tiny_suite_kwargs(tmp_path, "det_a", n_patients=16)
    b_kwargs = _tiny_suite_kwargs(tmp_path, "det_b", n_patients=16)
    run_config("default", **a_kwargs)
    run_config("default", **b_kwargs)
    assert _read_rows(a_kwargs["writer"]) == _read_rows(b_kwargs["writer"])


def test_cli_parsing() -> None:
    from csd_observer.benchmark.__main__ import parse_args

    names, n_seeds, gen, diff, methods = parse_args(
        ["patients_100", "n_seeds=2", "generator=bury", "difficulty=hard", "methods=VAR-CSD,DMD-CSD"]
    )
    assert names == ["patients_100"]
    assert n_seeds == 2
    assert gen == "bury"
    assert diff == "hard"
    assert methods == {"VAR-CSD", "DMD-CSD"}

    names, n_seeds, gen, diff, methods = parse_args(["n_seeds=1", "methods=all"])
    assert names == ["patients_100", "patients_200", "patients_300", "patients_400", "patients_500", "high_noise"]
    assert methods is None

    with pytest.raises(ValueError, match="Unknown method"):
        parse_args(["methods=NOPE"])
