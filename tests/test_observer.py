"""Smoke tests for csd_observer package."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_build_dataset() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset("fold", n_trajectories=20, max_length=50, noise_scale=0.1, seed=42, null=False)
    assert data["features"].shape == (20, 50, 1)
    assert data["is_positive"].all()


def test_build_dataset_hopf() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset("hopf", n_trajectories=10, max_length=30, noise_scale=0.05, seed=42, null=False)
    assert data["features"].shape == (10, 30, 2)


def test_build_dataset_logistic() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset("logistic", n_trajectories=10, max_length=30, noise_scale=0.02, seed=42, null=False)
    assert data["features"].shape == (10, 30, 1)


def test_build_dataset_null() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset("fold", n_trajectories=10, max_length=50, noise_scale=0.1, seed=42, null=True)
    assert not data["is_positive"].any()


def test_build_dataset_invalid_system() -> None:
    import pytest

    from csd_observer.datasets.synthetic.common.generators import build_dataset
    with pytest.raises(ValueError, match="Unknown system"):
        build_dataset("nonexistent")


def test_build_dataset_invalid_generator() -> None:
    import pytest

    from csd_observer.datasets.synthetic.common.generators import build_dataset
    with pytest.raises(ValueError, match="Unknown generator"):
        build_dataset("fold", generator="nope")


def test_build_dataset_bury_shapes() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
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
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset("fold", generator="bury", n_trajectories=100, max_length=80, seed=3)
    bifs = data["bifurcation_times"]
    assert np.ptp(bifs) > 5.0
    assert bifs.min() >= 0.0
    assert bifs.max() <= 80.0


def test_build_dataset_bury_varying_starts_and_noise() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset("fold", generator="bury", n_trajectories=60, max_length=80, seed=11)
    assert np.ptp(data["r_values"][:, 0]) > 0.05
    per_traj_std = np.std(data["features"][:, :, 0], axis=1)
    assert np.ptp(per_traj_std) > 1e-4


def test_build_dataset_bury_null() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset("fold", generator="bury", n_trajectories=20, max_length=50, seed=3, null=True)
    assert not data["is_positive"].any()
    assert (data["bifurcation_times"] == 51.0).all()
    assert np.ptp(data["r_values"][:, 0]) > 0.05


def test_build_dataset_bury_deterministic() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    a = build_dataset("fold", generator="bury", n_trajectories=10, max_length=40, seed=5)
    b = build_dataset("fold", generator="bury", n_trajectories=10, max_length=40, seed=5)
    assert np.array_equal(a["features"], b["features"])
    assert np.array_equal(a["bifurcation_times"], b["bifurcation_times"])
    assert np.array_equal(a["r_values"], b["r_values"])


def test_build_dataset_hard_shapes() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    for system, channels in (("fold", 1), ("hopf", 2), ("logistic", 1)):
        data = build_dataset(
            system, generator="bury", difficulty="hard", n_trajectories=30,
            max_length=60, noise_scale=0.1, seed=7,
        )
        assert data["features"].shape == (30, 60, channels)
        assert data["features"].shape == data["true_states"].shape
        assert data["seq_lengths"].shape == (30,)


def test_build_dataset_hard_signals_cross() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
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
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    data = build_dataset(
        "fold", generator="bury", difficulty="hard", n_trajectories=120,
        max_length=200, seed=3, null=True,
    )
    assert not data["is_positive"].any()
    assert (data["bifurcation_times"] == 201.0).all()


def test_build_dataset_hard_co_moving_nulls() -> None:
    from csd_observer.datasets.synthetic.common.generators import build_dataset
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
    from csd_observer.datasets.synthetic.common.generators import build_dataset
    a = build_dataset("fold", generator="bury", difficulty="hard", n_trajectories=10, max_length=40, seed=5)
    b = build_dataset("fold", generator="bury", difficulty="hard", n_trajectories=10, max_length=40, seed=5)
    assert np.array_equal(a["features"], b["features"])
    assert np.array_equal(a["bifurcation_times"], b["bifurcation_times"])


def test_build_dataset_invalid_difficulty() -> None:
    import pytest

    from csd_observer.datasets.synthetic.common.generators import build_dataset
    with pytest.raises(ValueError, match="Unknown difficulty"):
        build_dataset("fold", generator="bury", difficulty="nope")


def test_output_writer(tmp_path: Path) -> None:
    from csd_observer.outputs.writer import OutputWriter
    writer = OutputWriter("test", base_dir=str(tmp_path))
    assert writer.root.exists()
    for sub in ("resolved_config", "metadata", "metrics", "results", "artifacts", "logs", "times", "tables"):
        assert (writer.root / sub).is_dir()

    writer.write_resolved_config({"key": "value"})
    writer.write_result_row({
        "run_id": "r1", "timestamp": writer.timestamp, "run_name": "test",
        "dataset": "synthetic_fold", "bif_type": "fold", "system": "fold",
        "replicate": "s0", "method": "VAR-CSD", "family": "indicator",
        "is_learned": False, "k_persist": 5, "fpr_target": 0.05,
        "detection_rate": 0.4, "detection_time_mean": 50.0,
        "detection_time_median": 20.0, "detection_time_std": 30.0,
        "ew_auc": 0.6, "fpr": 0.05, "persistent_fpr": 0.03,
        "threshold": 0.14, "params": {"window_size": 30},
        "config_hash": "abc", "git_sha": "def", "seed": 0,
    })
    writer.write_metrics({"system": {"ew_auc": 0.6}})

    assert (writer.root / "resolved_config" / "resolved.yaml").exists()
    assert (writer.root / "results" / "results.jsonl").exists()
    assert (writer.root / "metrics" / "metrics.json").exists()


def test_output_writer_multiple_rows(tmp_path: Path) -> None:
    from csd_observer.outputs.writer import OutputWriter
    writer = OutputWriter("multi", base_dir=str(tmp_path))
    for i in range(5):
        writer.write_result_row({
            "run_id": f"r{i}", "timestamp": writer.timestamp, "run_name": "multi",
            "dataset": "synthetic_fold", "bif_type": "fold", "system": "fold",
            "replicate": "s0", "method": "VAR-CSD", "family": "indicator",
            "is_learned": False, "k_persist": 5, "fpr_target": 0.05,
            "detection_rate": 0.4, "detection_time_mean": 50.0,
            "detection_time_median": 20.0, "detection_time_std": 30.0,
            "ew_auc": 0.6, "fpr": 0.05, "persistent_fpr": 0.03,
            "threshold": 0.14, "params": {"window_size": 30},
            "config_hash": "abc", "git_sha": "def", "seed": i,
        })
    lines = (writer.root / "results" / "results.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5
    writer.mark_completed()
    assert (writer.root.with_suffix(writer.root.suffix + ".completed")).exists()


def test_metrics_functions() -> None:
    from csd_observer.evaluation.common.metrics import (
        compute_detection_time,
        compute_early_warning_auc,
        compute_false_positive_rate,
        compute_null_metrics,
    )
    B, T = 4, 50
    probs = np.random.rand(B, T).astype(np.float32)
    bif_times = np.full(B, 30.0, dtype=np.float32)
    is_pos = np.ones(B, dtype=bool)
    seq_lens = np.full(B, T, dtype=np.int64)

    dt = compute_detection_time(probs, bif_times, is_pos, seq_lens, threshold=0.5)
    assert np.isfinite(dt) or np.isnan(dt)

    fpr = compute_false_positive_rate(probs, seq_lens, threshold=0.5)
    assert 0.0 <= fpr <= 1.0

    auc = compute_early_warning_auc(probs, bif_times, is_pos, seq_lens, probs, seq_lens)
    assert 0.0 <= auc <= 1.0

    null_metrics = compute_null_metrics(probs, 0.5, seq_lens)
    assert "fpr" in null_metrics
    assert 0.0 <= null_metrics["fpr"] <= 1.0


def test_early_warning_auc_both_classes() -> None:
    from csd_observer.evaluation.common.metrics import compute_early_warning_auc
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


def test_linear_detrend_zero_trend() -> None:
    from csd_observer.models.common.detrend import _linear_detrend
    seg = np.ones(30, dtype=np.float32) * 5.0
    detrended = _linear_detrend(seg)
    assert np.allclose(detrended, np.zeros(30), atol=1e-5)


def test_linear_detrend_line() -> None:
    from csd_observer.models.common.detrend import _linear_detrend
    x = np.arange(30, dtype=np.float32)
    seg = 2.0 * x + 1.0
    detrended = _linear_detrend(seg)
    assert np.allclose(detrended, np.zeros(30), atol=1e-4)
