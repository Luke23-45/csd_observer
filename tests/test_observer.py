"""Smoke tests for csd_observer package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def test_model_creation() -> None:
    from csd_observer.models.csd_observer import CSDKalmanObserver
    model = CSDKalmanObserver(input_dim=2, latent_dim=4, lstm_head=True)
    assert model.latent_dim == 4
    assert model.lstm_head is True
    assert model.count_parameters() > 0

    model_bce = CSDKalmanObserver(input_dim=2, latent_dim=4, lstm_head=False)
    assert model_bce.lstm_head is False


def test_model_forward() -> None:
    from csd_observer.models.csd_observer import CSDKalmanObserver
    model = CSDKalmanObserver(input_dim=1, latent_dim=4, lstm_head=True, aux_head=True)
    x = torch.randn(8, 200, 1)
    logits, zs, A, K, C, aux = model(x)
    assert logits.shape == (8, 200)
    assert zs.shape == (8, 200, 4)
    assert A.shape == (4, 4)
    assert K.shape == (4, 1)
    assert C.shape == (1, 4)
    assert aux is not None
    assert aux.shape == (8, 200, 2)


def test_model_bce_forward() -> None:
    from csd_observer.models.csd_observer import CSDKalmanObserver
    model = CSDKalmanObserver(input_dim=1, latent_dim=4, lstm_head=False)
    x = torch.randn(4, 100, 1)
    logits, zs, A, K, C, aux = model(x)
    assert logits.shape == (4, 100)
    assert aux is None


def test_loss_creation() -> None:
    from csd_observer.utils.losses import SpectralRadiusLoss
    loss_fn = SpectralRadiusLoss(weight=0.1, threshold=0.95)
    assert loss_fn.weight == 0.1
    assert loss_fn.threshold == 0.95


def test_loss_forward() -> None:
    from csd_observer.utils.losses import SpectralRadiusLoss
    loss_fn = SpectralRadiusLoss()
    A = torch.eye(4) * 0.95
    K = torch.randn(4, 1).mul(0.1)
    C = torch.randn(1, 4).mul(0.1)
    out = loss_fn(A, K, C)
    assert "loss" in out
    assert out["loss"].ndim == 0
    assert out["loss"].item() >= 0.0


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


def test_load_config() -> None:
    from csd_observer.config.load import load_config
    config = load_config("default")
    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert config["data"]["noise_scale"] == 0.15
    assert config["data"]["n_patients"] == 500
    assert config["model"]["latent_dim"] == 4


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
    model = {"latent_dim": 4, "lstm_dim": 8}
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


def test_evaluate_raw_csd() -> None:
    from csd_observer.utils.metrics import evaluate_raw_csd, evaluate_raw_lag2
    B, T = 10, 100
    scores = np.random.rand(B, T).astype(np.float32)
    bif_times = np.full(B, 60.0, dtype=np.float32)
    is_pos = np.ones(B, dtype=bool)
    seq_lens = np.full(B, T, dtype=np.int64)
    scores_null = np.random.rand(B, T).astype(np.float32)

    metrics = evaluate_raw_csd(scores, bif_times, is_pos, seq_lens, scores_null, seq_lens, threshold=0.5)
    assert "detection_time" in metrics
    assert "ew_auc" in metrics

    metrics_lag2 = evaluate_raw_lag2(scores, bif_times, is_pos, seq_lens, scores_null, seq_lens, threshold=0.5)
    assert "detection_time" in metrics_lag2
    assert "ew_auc" in metrics_lag2
    assert "fpr" in metrics_lag2


def test_compute_null_metrics() -> None:
    from csd_observer.utils.metrics import compute_null_metrics
    B, T = 10, 50
    probs = np.random.rand(B, T).astype(np.float32)
    seq_lens = np.full(B, T, dtype=np.int64)
    metrics = compute_null_metrics(probs, 0.5, seq_lens)
    assert "fpr" in metrics
    assert 0.0 <= metrics["fpr"] <= 1.0


def test_tensorize() -> None:
    from csd_observer.data.bifurcation import build_dataset
    from csd_observer.training.trainer import tensorize
    data = build_dataset("fold", n_trajectories=10, max_length=30, noise_scale=0.1, seed=42, null=False)
    tensors = tensorize(data, torch.device("cpu"))
    assert tensors.features.shape == (10, 30, 1)
    assert tensors.masks.shape == (10, 30, 1)
    assert tensors.seq_lengths.shape == (10,)
    assert tensors.bifurcation_times.shape == (10,)
    assert tensors.is_positive.shape == (10,)


def test_tensorize_augmented_features() -> None:
    from csd_observer.training.trainer import tensorize

    dataset = {
        "features": np.random.randn(2, 30, 1).astype(np.float32),
        "seq_lengths": np.array([30, 24], dtype=np.int64),
        "bifurcation_times": np.array([20.0, 18.0], dtype=np.float32),
        "is_positive": np.array([True, True], dtype=np.bool_),
        "split_indices": {"train": np.array([0]), "val": np.array([1]), "test": np.array([0, 1])},
        "augment_features": True,
    }

    tensors = tensorize(dataset, torch.device("cpu"))
    assert tensors.features.shape == (2, 30, 5)
    assert tensors.masks.shape == (2, 30, 5)


def test_tensorize_phase_features() -> None:
    from csd_observer.training.trainer import tensorize

    dataset = {
        "features": np.random.randn(2, 30, 1).astype(np.float32),
        "seq_lengths": np.array([30, 24], dtype=np.int64),
        "bifurcation_times": np.array([20.0, 18.0], dtype=np.float32),
        "is_positive": np.array([True, True], dtype=np.bool_),
        "split_indices": {"train": np.array([0]), "val": np.array([1]), "test": np.array([0, 1])},
        "augment_features": True,
        "phase_features": True,
    }

    tensors = tensorize(dataset, torch.device("cpu"))
    assert tensors.features.shape == (2, 30, 9)
    assert tensors.masks.shape == (2, 30, 9)


def test_train_csd_observer() -> None:
    from csd_observer.config.load import load_config
    from csd_observer.data.bifurcation import build_dataset
    from csd_observer.training.trainer import build_probs, tensorize, train_kalman

    config = load_config("default")
    config["data"]["max_length"] = 30
    config["data"]["n_patients"] = 20
    config["training"]["epochs"] = 2

    data = build_dataset("fold", n_trajectories=20, max_length=30, noise_scale=0.1, seed=42, null=False)
    device = torch.device("cpu")
    tensors = tensorize(data, device)
    model = train_kalman(
        tensors, data["split_indices"]["train"], data["split_indices"]["val"],
        loss_type="lstm", seed=42, config=config, device=device,
    )
    assert model.lstm_head
    probs = build_probs(model, tensors, data["split_indices"]["test"])
    assert probs.shape[0] == len(data["split_indices"]["test"])


def test_train_kalman_bce() -> None:
    from csd_observer.config.load import load_config
    from csd_observer.data.bifurcation import build_dataset
    from csd_observer.training.trainer import build_probs, tensorize, train_kalman

    config = load_config("default")
    config["data"]["max_length"] = 30
    config["data"]["n_patients"] = 20
    config["training"]["epochs"] = 2

    data = build_dataset("fold", n_trajectories=20, max_length=30, noise_scale=0.1, seed=42, null=False)
    device = torch.device("cpu")
    tensors = tensorize(data, device)
    model = train_kalman(
        tensors, data["split_indices"]["train"], data["split_indices"]["val"],
        loss_type="bce", seed=42, config=config, device=device,
    )
    assert not model.lstm_head
    probs = build_probs(model, tensors, data["split_indices"]["test"])
    assert probs.shape[0] == len(data["split_indices"]["test"])


def test_train_kalman_spec() -> None:
    from csd_observer.config.load import load_config
    from csd_observer.data.bifurcation import build_dataset
    from csd_observer.training.trainer import build_probs, tensorize, train_kalman

    config = load_config("default")
    config["data"]["max_length"] = 30
    config["data"]["n_patients"] = 20
    config["training"]["epochs"] = 2

    data = build_dataset("fold", n_trajectories=20, max_length=30, noise_scale=0.1, seed=42, null=False)
    device = torch.device("cpu")
    tensors = tensorize(data, device)
    model = train_kalman(
        tensors, data["split_indices"]["train"], data["split_indices"]["val"],
        loss_type="lstm_spec", seed=42, config=config, device=device,
    )
    assert model.lstm_head
    probs = build_probs(model, tensors, data["split_indices"]["test"])
    assert probs.shape[0] == len(data["split_indices"]["test"])


def test_train_kalman_aux() -> None:
    from csd_observer.config.load import load_config
    from csd_observer.data.bifurcation import build_dataset
    from csd_observer.training.trainer import build_probs, tensorize, train_kalman

    config = load_config("default")
    config["data"]["max_length"] = 30
    config["data"]["n_patients"] = 20
    config["training"]["epochs"] = 2
    config["training"]["aux_loss_weight"] = 0.3

    data = build_dataset("fold", n_trajectories=20, max_length=30, noise_scale=0.1, seed=42, null=False)
    device = torch.device("cpu")
    tensors = tensorize(data, device)
    model = train_kalman(
        tensors, data["split_indices"]["train"], data["split_indices"]["val"],
        loss_type="lstm_aux", seed=42, config=config, device=device,
    )
    assert model.lstm_head
    assert model.aux_head
    probs = build_probs(model, tensors, data["split_indices"]["test"])
    assert probs.shape[0] == len(data["split_indices"]["test"])


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
    from csd_observer.utils.metrics import _linear_detrend
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


def test_classical_kalman_shapes() -> None:
    from csd_observer.models.kalman_lag2 import ClassicalKalmanLag2
    B, T = 4, 100
    y = torch.randn(B, T)
    model = ClassicalKalmanLag2(q=1e-3, r=1.0)
    out = model(y)
    assert out["mu_hat"].shape == (B, T)
    assert out["delta_hat"].shape == (B, T)
    assert out["innovation"].shape == (B, T)
    assert out["y"].shape == (B, T)


def test_classical_kalman_no_params() -> None:
    from csd_observer.models.kalman_lag2 import ClassicalKalmanLag2
    model = ClassicalKalmanLag2(q=1e-3, r=1.0)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 0, f"Expected 0 trainable params, got {n_params}"


def test_classical_kalman_q_low_smooths() -> None:
    from csd_observer.models.kalman_lag2 import ClassicalKalmanLag2
    B, T = 1, 200
    rng = np.random.default_rng(42)
    y = np.zeros((B, T), dtype=np.float32)
    y[0, 100:] = 0.8
    y[0] += rng.normal(0, 0.05, T).astype(np.float32)
    y_t = torch.from_numpy(y)
    model = ClassicalKalmanLag2(q=1e-6, r=1.0)
    out = model(y_t)
    mu = out["mu_hat"].numpy()
    var_raw = np.var(y[0, -50:])
    var_mu = np.var(mu[0, -50:])
    assert var_mu < var_raw, f"Smoothed variance {var_mu} >= raw {var_raw}"


def test_classical_kalman_q_high_follows() -> None:
    from csd_observer.models.kalman_lag2 import ClassicalKalmanLag2
    B, T = 1, 100
    y = torch.zeros(B, T)
    y[0, 50:] = 1.0
    model = ClassicalKalmanLag2(q=1e-1, r=1.0)
    out = model(y)
    mu = out["mu_hat"].numpy()
    assert mu[0, -1] > 0.8, f"mu[-1] = {mu[0, -1]}"


def test_grid_search_q_returns_valid() -> None:
    from csd_observer.models.kalman_lag2 import grid_search_q
    B, T = 10, 100
    rng = np.random.default_rng(42)
    _ = rng.normal(0, 0.1, (B, T)).astype(np.float32)
    y_val = rng.normal(0, 0.1, (B, T)).astype(np.float32)
    bifs = np.full(B, 80.0, dtype=np.float32)
    is_pos = np.concatenate([np.ones(B // 2, dtype=bool), np.zeros(B - B // 2, dtype=bool)])
    lens = np.full(B, T, dtype=np.int64)
    best_q = grid_search_q(y_val, bifs, is_pos, lens)
    assert best_q in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def test_kalman_lag2_mlp_count() -> None:
    from csd_observer.models.kalman_lag2 import KalmanLag2MLPHead
    head = KalmanLag2MLPHead(input_dim=4, hidden_dim=4)
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == 25, f"Expected 25 params, got {n_params}"


def test_kalman_lag2_net_forward() -> None:
    from csd_observer.models.kalman_lag2 import KalmanLag2Net
    B, T = 4, 100
    y = torch.randn(B, T)
    model = KalmanLag2Net(q=1e-3, r=1.0)
    logits = model(y)
    assert logits.shape == (B, T)


def test_kalman_lag2_net_forward_numpy() -> None:
    from csd_observer.models.kalman_lag2 import KalmanLag2Net
    B, T = 4, 100
    y = np.random.randn(B, T).astype(np.float32)
    model = KalmanLag2Net(q=1e-3, r=1.0)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(y))
    assert logits.shape == (B, T)


def test_acko_creation() -> None:
    from csd_observer.models.csd_observer import CSDKalmanObserver
    model = CSDKalmanObserver(input_dim=2, latent_dim=4, parity_aware=True)
    assert model.parity_aware
    assert model.parity_channel == 1
    assert model.lstm_head is False
    assert model.aux_head is False
    assert model.count_parameters() > 0


def test_acko_forward() -> None:
    from csd_observer.models.csd_observer import CSDKalmanObserver
    model = CSDKalmanObserver(input_dim=2, latent_dim=4, parity_aware=True)
    B, T = 4, 100
    x = torch.randn(B, T, 2)
    x[:, :, 1] = 1.0  # default parity = +1 (even)
    x[:, 1::2, 1] = -1.0  # odd beats = -1
    logits, zs, A, K, C, alt = model(x)
    assert logits.shape == (B, T)
    assert zs.shape == (B, T, 4)
    assert A is None
    assert K is None
    assert C is None
    assert alt is None


def test_acko_forward_masked() -> None:
    from csd_observer.models.csd_observer import CSDKalmanObserver
    model = CSDKalmanObserver(input_dim=2, latent_dim=4, parity_aware=True)
    B, T = 4, 100
    x = torch.randn(B, T, 2)
    x[:, :, 1] = 1.0
    x[:, 1::2, 1] = -1.0
    mask = torch.ones(B, T, 2)
    mask[:, 50:, :] = 0.0
    logits, zs, _, _, _, alt = model(x, mask)
    assert logits.shape == (B, T)
    assert torch.isfinite(logits).all()
    assert alt is None


def test_acko_odd_latent_dim_raises() -> None:
    import pytest
    from csd_observer.models.csd_observer import CSDKalmanObserver
    with pytest.raises(ValueError, match="latent_dim even"):
        CSDKalmanObserver(input_dim=2, latent_dim=3, parity_aware=True)


def test_train_kalman_parity() -> None:
    from csd_observer.config.load import load_config
    from csd_observer.data.bifurcation import build_dataset
    from csd_observer.training.trainer import build_probs, tensorize, train_kalman

    config = load_config("default")
    config["data"]["max_length"] = 30
    config["data"]["n_patients"] = 20
    config["training"]["epochs"] = 2

    data = build_dataset("fold", n_trajectories=20, max_length=30, noise_scale=0.1, seed=42, null=False)
    data["augment_features"] = True
    data["phase_features"] = True
    device = torch.device("cpu")
    tensors = tensorize(data, device)
    model = train_kalman(
        tensors, data["split_indices"]["train"], data["split_indices"]["val"],
        loss_type="parity", seed=42, config=config, device=device,
    )
    assert model.parity_aware
    assert not model.lstm_head
    probs = build_probs(model, tensors, data["split_indices"]["test"])
    assert probs.shape[0] == len(data["split_indices"]["test"])
    assert np.isfinite(probs).all()
