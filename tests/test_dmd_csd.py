"""Tests for the DMD-CSD baseline (dominant-eigenvalue dynamic-mode-decomposition)."""

from __future__ import annotations

import numpy as np


def _dmd(features, seq_lengths, window_size=30, embedding_dim=6, rank=2):
    from csd_observer.models.dmd_csd.indicator import raw_dmd_indicator
    return raw_dmd_indicator(features, seq_lengths, window_size, embedding_dim, rank)


def _ar1(rng, phi, B, T):
    x = np.empty((B, T))
    e = rng.normal(0.0, 1.0, (B, T))
    x[:, 0] = e[:, 0]
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def _ar2(rng, rho, omega, B, T):
    a = 2.0 * rho * np.cos(omega)
    b = -rho**2
    x = np.empty((B, T))
    e = rng.normal(0.0, 1.0, (B, T))
    x[:, 0] = e[:, 0]
    x[:, 1] = a * x[:, 0] + e[:, 1]
    for t in range(2, T):
        x[:, t] = a * x[:, t - 1] + b * x[:, t - 2] + e[:, t]
    return x


def _manual_dmd(seg, m=6, rank=2):
    """Independent from-scratch SVD-DMD: |lambda_max| of U^T X2 V S^{-1}."""
    from csd_observer.utils.metrics import _linear_detrend

    seg = _linear_detrend(seg)
    snap = np.stack([seg[k : k + m] for k in range(len(seg) - m + 1)])
    x1 = snap[:-1].T
    x2 = snap[1:].T
    u, s, vt = np.linalg.svd(x1, full_matrices=False)
    r = min(rank, len(s), int((s > 1e-6 * s[0]).sum()))
    if r == 0:
        return float("nan")
    a = (u[:, :r].T @ x2 @ vt[:r].T) @ np.diag(1.0 / s[:r])
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def test_dmd_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, (4, 50, 1)).astype(np.float32)
    seq_lengths = np.full(4, 50, dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=10)
    assert scores.shape == (4, 50)
    assert scores.dtype == np.float32


def test_dmd_invalid_ndim() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be \\(B, T, C\\)"):
        _dmd(np.zeros((4, 50), dtype=np.float32), np.full(4, 50, dtype=np.int64))


def test_dmd_invalid_seq_lengths() -> None:
    import pytest

    features = np.zeros((4, 50, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="seq_lengths length"):
        _dmd(features, np.full(3, 50, dtype=np.int64))


def test_dmd_short_prefix_is_nan() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 1.0, (1, 20, 1)).astype(np.float32)
    seq_lengths = np.full(1, 20, dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=30)
    assert np.isnan(scores[0, :6]).all()
    assert np.isfinite(scores[0, 6:]).all()


def test_dmd_beyond_seq_length_is_nan() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(0.0, 1.0, (3, 50, 1)).astype(np.float32)
    seq_lengths = np.array([10, 25, 50], dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=30)
    for b, L in enumerate(seq_lengths):
        assert np.isfinite(scores[b, 6:L]).all()
        assert np.isnan(scores[b, L:]).all()


def test_dmd_degenerate_windows_are_nan() -> None:
    constant = np.full((1, 30, 1), 5.0, dtype=np.float32)
    linear = (2.0 * np.arange(30) + 1.0).reshape(1, 30, 1).astype(np.float32)
    seq_lengths = np.full(1, 30, dtype=np.int64)
    assert np.isnan(_dmd(constant, seq_lengths, window_size=30)).all()
    assert np.isnan(_dmd(linear, seq_lengths, window_size=30)).all()


def test_dmd_is_causal() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (2, 50, 1)).astype(np.float32)
    seq_lengths = np.full(2, 50, dtype=np.int64)
    scores_base = _dmd(base, seq_lengths, window_size=10)
    for t in range(5, 45, 5):
        modified = base.copy()
        modified[:, t + 1 :] = 1000.0 * rng.normal(size=modified[:, t + 1 :].shape)
        scores_mod = _dmd(modified, seq_lengths, window_size=10)
        assert np.array_equal(np.isnan(scores_base), np.isnan(scores_mod))
        assert np.allclose(scores_base[:, 6 : t + 1], scores_mod[:, 6 : t + 1], atol=1e-6)


def test_dmd_deterministic() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 1.0, (3, 40, 1)).astype(np.float32)
    seq_lengths = np.full(3, 40, dtype=np.int64)
    a = _dmd(features, seq_lengths, window_size=15)
    b = _dmd(features, seq_lengths, window_size=15)
    assert np.array_equal(a, b, equal_nan=True)


def test_dmd_matches_manual_svd_formula() -> None:
    rng = np.random.default_rng(9)
    B, n = 4, 30
    x = _ar1(rng, 0.6, B, n)
    seq_lengths = np.full(B, n, dtype=np.int64)
    scores = _dmd(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    for b in range(B):
        expected = _manual_dmd(x[b])
        assert np.allclose(scores[b, n - 1], expected, rtol=1e-5, atol=1e-5)


def test_dmd_pure_sine_modulus_is_one() -> None:
    t = 60
    seq_lengths = np.array([t], dtype=np.int64)
    for P in (5, 8, 10):
        x = np.sin(2.0 * np.pi * np.arange(t) / P)
        scores = _dmd(x[None, :, None].astype(np.float32), seq_lengths, window_size=30)
        assert abs(float(scores[0, t - 1]) - 1.0) < 1e-2


def test_dmd_rank_one_collapses_oscillatory_mode() -> None:
    t = 60
    seq_lengths = np.array([t], dtype=np.int64)
    x = np.sin(2.0 * np.pi * np.arange(t) / 5.0)
    s1 = _dmd(x[None, :, None].astype(np.float32), seq_lengths, window_size=30, rank=1)
    s2 = _dmd(x[None, :, None].astype(np.float32), seq_lengths, window_size=30, rank=2)
    assert float(s1[0, t - 1]) < 0.95
    assert float(s2[0, t - 1]) > float(s1[0, t - 1]) + 0.05


def test_dmd_scores_order_by_ar1_coefficient() -> None:
    rng = np.random.default_rng(2)
    B, W = 300, 30
    seq_lengths = np.full(B, W, dtype=np.int64)
    means = {}
    for phi in (0.9, 0.5, 0.2, 0.0):
        x = _ar1(rng, phi, B, W)
        scores = _dmd(x[..., None].astype(np.float32), seq_lengths, window_size=W)
        means[phi] = float(scores[:, W - 1].mean())
    assert means[0.9] > means[0.5] + 0.02
    assert means[0.5] > means[0.2] + 0.04
    assert means[0.2] > means[0.0] + 0.02


def test_dmd_white_noise_baseline_below_strong_slowing() -> None:
    rng = np.random.default_rng(2)
    B, W = 300, 30
    seq_lengths = np.full(B, W, dtype=np.int64)
    x_white = _ar1(rng, 0.0, B, W)
    x_slow = _ar1(rng, 0.9, B, W)
    s_white = _dmd(x_white[..., None].astype(np.float32), seq_lengths, window_size=W)
    s_slow = _dmd(x_slow[..., None].astype(np.float32), seq_lengths, window_size=W)
    assert float(s_slow[:, W - 1].mean()) > float(s_white[:, W - 1].mean()) + 0.1


def test_dmd_ar2_pole_modulus_rises() -> None:
    rng = np.random.default_rng(11)
    B, W = 500, 30
    seq_lengths = np.full(B, W, dtype=np.int64)
    x_hi = _ar2(rng, 0.95, np.pi / 8.0, B, W)
    x_lo = _ar2(rng, 0.7, np.pi / 8.0, B, W)
    s_hi = _dmd(x_hi[..., None].astype(np.float32), seq_lengths, window_size=W)
    s_lo = _dmd(x_lo[..., None].astype(np.float32), seq_lengths, window_size=W)
    assert float(s_hi[:, W - 1].mean()) > float(s_lo[:, W - 1].mean()) + 0.01


def test_dmd_embedding_dim_changes_min_prefix() -> None:
    rng = np.random.default_rng(13)
    features = rng.normal(0.0, 1.0, (1, 20, 1)).astype(np.float32)
    seq_lengths = np.full(1, 20, dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=30, embedding_dim=4)
    assert np.isnan(scores[0, :4]).all()
    assert np.isfinite(scores[0, 4:]).all()


def test_dmd_partial_window_equals_whole_prefix() -> None:
    rng = np.random.default_rng(23)
    B, T, W = 2, 60, 30
    features = _ar1(rng, 0.6, B, T)[..., None].astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=W)
    for t in range(7, W):
        prefix = features[:, : t + 1, :]
        lens = np.full(B, t + 1, dtype=np.int64)
        scores_prefix = _dmd(prefix, lens, window_size=1000)
        assert np.allclose(scores[:, t], scores_prefix[:, t], rtol=1e-5, atol=1e-6)


def test_dmd_max_over_channels() -> None:
    rng = np.random.default_rng(29)
    B, T, W = 3, 60, 30
    ch0 = rng.normal(0.0, 1.0, (B, T))
    ch1 = _ar1(rng, 0.8, B, T)
    features = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=W)
    s0 = _dmd(ch0[..., None], seq_lengths, window_size=W)
    s1 = _dmd(ch1[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], np.maximum(s0, s1)[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_dmd_max_over_channels_ignores_degenerate() -> None:
    rng = np.random.default_rng(31)
    B, T, W = 2, 40, 30
    noisy = _ar1(rng, 0.7, B, T)
    constant = np.full((B, T), 3.0)
    features = np.stack([noisy, constant], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=W)
    s_noisy = _dmd(noisy[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], s_noisy[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_dmd_float_window_size() -> None:
    rng = np.random.default_rng(33)
    features = rng.normal(0.0, 1.0, (2, 40, 1)).astype(np.float32)
    seq_lengths = np.full(2, 40, dtype=np.int64)
    a = _dmd(features, seq_lengths, window_size=30)
    b = _dmd(features, seq_lengths, window_size=30.0)
    assert np.array_equal(a, b, equal_nan=True)


def test_dmd_tiny_window_all_nan() -> None:
    rng = np.random.default_rng(37)
    features = rng.normal(0.0, 1.0, (2, 40, 1)).astype(np.float32)
    seq_lengths = np.full(2, 40, dtype=np.int64)
    scores = _dmd(features, seq_lengths, window_size=3)
    assert np.isnan(scores).all()
