"""Smoke tests for csd_observer package."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_build_dataset() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset("fold", n_trajectories=20, max_length=50, noise_scale=0.1, seed=42, null=False)
    assert data["features"].shape == (20, 50, 1)
    assert data["is_positive"].all()


def test_build_dataset_hopf() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset("hopf", n_trajectories=10, max_length=30, noise_scale=0.05, seed=42, null=False)
    assert data["features"].shape == (10, 30, 2)


def test_build_dataset_logistic() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset("logistic", n_trajectories=10, max_length=30, noise_scale=0.02, seed=42, null=False)
    assert data["features"].shape == (10, 30, 1)


def test_build_dataset_null() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset("fold", n_trajectories=10, max_length=50, noise_scale=0.1, seed=42, null=True)
    assert not data["is_positive"].any()


def test_build_dataset_invalid_system() -> None:
    import pytest

    from csd_observer.data.bifurcation import build_dataset
    with pytest.raises(ValueError, match="Unknown system"):
        build_dataset("nonexistent")


def test_build_dataset_invalid_generator() -> None:
    import pytest

    from csd_observer.data.bifurcation import build_dataset
    with pytest.raises(ValueError, match="Unknown generator"):
        build_dataset("fold", generator="nope")


def test_build_dataset_bury_shapes() -> None:
    from csd_observer.data.bifurcation import build_dataset
    for system, channels in (("fold", 1), ("hopf", 2), ("logistic", 1)):
        data = build_dataset(
            system, generator="bury", n_trajectories=30, max_length=60,
            noise_scale=0.1, seed=7,
        )
        assert data["features"].shape == (30, 60, channels)
        assert data["features"].shape == data["true_states"].shape
        assert data["seq_lengths"].shape == (30,)
        assert data["is_positive"].all()


def test_build_dataset_bury_varying_bifurcation_times() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset("fold", generator="bury", n_trajectories=100, max_length=80, seed=3)
    bifs = data["bifurcation_times"]
    assert np.ptp(bifs) > 5.0
    assert bifs.min() >= 0.0
    assert bifs.max() <= 80.0


def test_build_dataset_bury_varying_starts_and_noise() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset("fold", generator="bury", n_trajectories=60, max_length=80, seed=11)
    assert np.ptp(data["r_values"][:, 0]) > 0.05
    per_traj_std = np.std(data["features"][:, :, 0], axis=1)
    assert np.ptp(per_traj_std) > 1e-4


def test_build_dataset_bury_null() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset("fold", generator="bury", n_trajectories=20, max_length=50, seed=3, null=True)
    assert not data["is_positive"].any()
    assert (data["bifurcation_times"] == 51.0).all()
    assert np.ptp(data["r_values"][:, 0]) > 0.05


def test_build_dataset_bury_deterministic() -> None:
    from csd_observer.data.bifurcation import build_dataset
    a = build_dataset("fold", generator="bury", n_trajectories=10, max_length=40, seed=5)
    b = build_dataset("fold", generator="bury", n_trajectories=10, max_length=40, seed=5)
    assert np.array_equal(a["features"], b["features"])
    assert np.array_equal(a["bifurcation_times"], b["bifurcation_times"])
    assert np.array_equal(a["r_values"], b["r_values"])


def test_build_dataset_hard_shapes() -> None:
    from csd_observer.data.bifurcation import build_dataset
    for system, channels in (("fold", 1), ("hopf", 2), ("logistic", 1)):
        data = build_dataset(
            system, generator="bury", difficulty="hard", n_trajectories=30,
            max_length=60, noise_scale=0.1, seed=7,
        )
        assert data["features"].shape == (30, 60, channels)
        assert data["features"].shape == data["true_states"].shape
        assert data["seq_lengths"].shape == (30,)


def test_build_dataset_hard_signals_cross() -> None:
    from csd_observer.data.bifurcation import build_dataset
    for system in ("fold", "hopf", "logistic"):
        data = build_dataset(
            system, generator="bury", difficulty="hard", n_trajectories=120,
            max_length=200, seed=3,
        )
        assert data["is_positive"].all()
        bifs = data["bifurcation_times"]
        assert (bifs > 0.0).all()
        assert (bifs < 200.0).all()
        assert np.ptp(bifs) > 5.0


def test_build_dataset_hard_null_sentinels() -> None:
    from csd_observer.data.bifurcation import build_dataset
    data = build_dataset(
        "fold", generator="bury", difficulty="hard", n_trajectories=120,
        max_length=200, seed=3, null=True,
    )
    assert not data["is_positive"].any()
    assert (data["bifurcation_times"] == 201.0).all()


def test_build_dataset_hard_co_moving_nulls() -> None:
    from csd_observer.data.bifurcation import build_dataset
    n = build_dataset(
        "logistic", generator="bury", difficulty="hard", n_trajectories=300,
        max_length=200, seed=5, null=True,
    )
    ends = n["mu_values"][:, -1]
    assert (ends > 2.85).sum() > 10
    assert (ends < 3.0).all()
    s = build_dataset(
        "logistic", generator="bury", difficulty="hard", n_trajectories=300,
        max_length=200, seed=5,
    )
    assert (s["mu_values"][:, -1] > 3.0).all()


def test_build_dataset_hard_deterministic() -> None:
    from csd_observer.data.bifurcation import build_dataset
    a = build_dataset("fold", generator="bury", difficulty="hard", n_trajectories=10, max_length=40, seed=5)
    b = build_dataset("fold", generator="bury", difficulty="hard", n_trajectories=10, max_length=40, seed=5)
    assert np.array_equal(a["features"], b["features"])
    assert np.array_equal(a["bifurcation_times"], b["bifurcation_times"])


def test_build_dataset_invalid_difficulty() -> None:
    import pytest

    from csd_observer.data.bifurcation import build_dataset
    with pytest.raises(ValueError, match="Unknown difficulty"):
        build_dataset("fold", generator="bury", difficulty="nope")


def test_load_config() -> None:
    from csd_observer.config.load import load_config
    config = load_config("default")
    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert config["data"]["noise_scale"] == 0.15
    assert config["data"]["n_patients"] == 500
    assert "spectral_drift" in config["model"]


def test_load_config_high_noise() -> None:
    from csd_observer.config.load import load_config
    config = load_config("high_noise")
    assert config["data"]["noise_scale"] == 0.30
    assert config["data"]["n_patients"] == 500


def test_load_config_low_data() -> None:
    from csd_observer.config.load import load_config
    config = load_config("low_data")
    assert config["data"]["n_patients"] == 200
    assert config["training"]["epochs"] == 50


def test_load_config_missing_file() -> None:
    import pytest

    from csd_observer.config.load import load_config
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_run")


def test_config_validation_types() -> None:
    import pytest

    from csd_observer.config.load import _validate_config
    data = {"noise_scale": "0.15", "n_patients": 500, "systems": ["fold"], "max_length": 200, "n_seeds": 5}
    model = {"spectral_drift": {"n_particles": 500}}
    training = {"epochs": 30, "batch_size": 256, "lr": 0.001, "patience": 5, "spectral_radius_weight": 0.1, "spectral_threshold": 0.95}
    with pytest.raises(TypeError, match="must be numeric"):
        _validate_config(data, model, training)


def test_config_validation_missing() -> None:
    import pytest

    from csd_observer.config.load import _validate_config
    with pytest.raises(ValueError, match="missing required"):
        _validate_config({}, {}, {})


def test_output_writer(tmp_path: Path) -> None:
    from csd_observer.utils.io import OutputWriter
    writer = OutputWriter(experiment_name="test", base_dir=str(tmp_path))
    assert writer.path.exists()
    assert (writer.path / "configs").exists()
    assert (writer.path / "metrics").exists()
    assert (writer.path / "results").exists()

    writer.write_config({"key": "value"})
    writer.write_result_row({"seed": 1, "dt": 10.0})
    writer.write_metrics({"system": {"dt": 10.0}})

    assert (writer.path / "configs" / "resolved.yaml").exists()
    assert (writer.path / "results" / "results.jsonl").exists()
    assert (writer.path / "metrics" / "metrics.json").exists()


def test_output_writer_multiple_rows(tmp_path: Path) -> None:
    from csd_observer.utils.io import OutputWriter
    writer = OutputWriter(experiment_name="multi", base_dir=str(tmp_path))
    for i in range(5):
        writer.write_result_row({"seed": i, "dt": float(i * 10)})
    lines = (writer.path / "results" / "results.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5


def test_metrics_functions() -> None:
    from csd_observer.utils.metrics import (
        compute_detection_time,
        compute_false_positive_rate,
        raw_csd_indicator,
        raw_lag2_indicator,
    )
    B, T, C = 4, 50, 1
    probs = np.random.rand(B, T).astype(np.float32)
    bif_times = np.full(B, 30.0, dtype=np.float32)
    is_pos = np.ones(B, dtype=bool)
    seq_lens = np.full(B, T, dtype=np.int64)

    dt = compute_detection_time(probs, bif_times, is_pos, seq_lens, threshold=0.5)
    assert np.isfinite(dt) or np.isnan(dt)

    fpr = compute_false_positive_rate(probs, seq_lens, threshold=0.5)
    assert 0.0 <= fpr <= 1.0

    features = np.random.randn(B, T, C).astype(np.float32)
    scores = raw_csd_indicator(features, seq_lens, window_size=10)
    assert scores.shape == (B, T)
    scores_lag2 = raw_lag2_indicator(features, seq_lens, window_size=10)
    assert scores_lag2.shape == (B, T)


def test_select_threshold_returns_float() -> None:
    from csd_observer.utils.metrics import select_threshold
    B, T = 20, 100
    probs = np.random.rand(B, T).astype(np.float32)
    bif_times = np.full(B, 60.0, dtype=np.float32)
    is_pos = np.ones(B, dtype=bool)
    seq_lens = np.full(B, T, dtype=np.int64)
    thresh = select_threshold(probs, bif_times, is_pos, seq_lens, target_sensitivity=0.8)
    assert isinstance(thresh, float)
    assert 0.0 <= thresh <= 1.0


def test_select_threshold_no_positives() -> None:
    from csd_observer.utils.metrics import select_threshold
    B, T = 10, 50
    probs = np.random.rand(B, T).astype(np.float32)
    bif_times = np.full(B, 0.0, dtype=np.float32)
    is_pos = np.zeros(B, dtype=bool)
    seq_lens = np.full(B, T, dtype=np.int64)
    thresh = select_threshold(probs, bif_times, is_pos, seq_lens)
    assert thresh == 0.5


def test_early_warning_auc_both_classes() -> None:
    from csd_observer.utils.metrics import compute_early_warning_auc
    B, T = 10, 100
    probs_signal = np.random.rand(B, T).astype(np.float32)
    probs_null = np.random.rand(B, T).astype(np.float32)
    bif_times = np.full(B, 70.0, dtype=np.float32)
    is_pos = np.ones(B, dtype=bool)
    seq_lens = np.full(B, T, dtype=np.int64)

    auc = compute_early_warning_auc(
        probs_signal, bif_times, is_pos, seq_lens,
        probs_null, seq_lens,
    )
    assert np.isfinite(auc) or np.isnan(auc)
    if np.isfinite(auc):
        assert 0.0 <= auc <= 1.0


def test_compute_null_metrics() -> None:
    from csd_observer.utils.metrics import compute_null_metrics
    B, T = 10, 50
    probs = np.random.rand(B, T).astype(np.float32)
    seq_lens = np.full(B, T, dtype=np.int64)
    metrics = compute_null_metrics(probs, 0.5, seq_lens)
    assert "fpr" in metrics
    assert 0.0 <= metrics["fpr"] <= 1.0


def test_linear_detrend_zero_trend() -> None:
    from csd_observer.utils.metrics import _linear_detrend
    seg = np.ones(30, dtype=np.float32) * 5.0
    detrended = _linear_detrend(seg)
    assert np.allclose(detrended, np.zeros(30), atol=1e-5)


def test_linear_detrend_line() -> None:
    from csd_observer.utils.metrics import _linear_detrend
    x = np.arange(30, dtype=np.float32)
    seg = 2.0 * x + 1.0
    detrended = _linear_detrend(seg)
    assert np.allclose(detrended, np.zeros(30), atol=1e-4)


def test_lag2_detrended_equals_original_on_flat() -> None:
    from csd_observer.utils.metrics import raw_lag2_indicator, raw_lag2_indicator_detrended
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (4, 50, 1)).astype(np.float32)
    seq_lengths = np.full(4, 50, dtype=np.int64)
    orig = raw_lag2_indicator(features, seq_lengths, window_size=10, detrend=False)
    det = raw_lag2_indicator_detrended(features, seq_lengths, window_size=10)
    assert not np.allclose(orig, det, atol=1e-5)
    const_features = np.full((4, 50, 1), 5.0, dtype=np.float32)
    orig_c = raw_lag2_indicator(const_features, seq_lengths, window_size=10, detrend=False)
    det_c = raw_lag2_indicator_detrended(const_features, seq_lengths, window_size=10)
    assert np.allclose(orig_c, det_c, atol=1e-6)
