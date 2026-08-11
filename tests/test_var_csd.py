"""Tests for the VAR-CSD baseline (windowed-variance indicator)."""

from __future__ import annotations

import numpy as np


def _var(features, seq_lengths, window_size=30):
    from csd_observer.models.indicators.var_csd.indicator import raw_var_indicator
    return raw_var_indicator(features, seq_lengths, window_size)


def test_var_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, (4, 50, 1)).astype(np.float32)
    seq_lengths = np.full(4, 50, dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=10)
    assert scores.shape == (4, 50)
    assert scores.dtype == np.float32


def test_var_invalid_ndim() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be \\(B, T, C\\)"):
        _var(np.zeros((4, 50), dtype=np.float32), np.full(4, 50, dtype=np.int64))


def test_var_invalid_seq_lengths() -> None:
    import pytest

    features = np.zeros((4, 50, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="seq_lengths length"):
        _var(features, np.full(3, 50, dtype=np.int64))


def test_var_short_prefix_is_nan() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 1.0, (1, 20, 1)).astype(np.float32)
    seq_lengths = np.full(1, 20, dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=30)
    assert np.isnan(scores[0, :3]).all()
    assert np.isfinite(scores[0, 3:]).all()


def test_var_beyond_seq_length_is_nan() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(0.0, 1.0, (3, 50, 1)).astype(np.float32)
    seq_lengths = np.array([10, 25, 50], dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=30)
    for b, L in enumerate(seq_lengths):
        assert np.isfinite(scores[b, 4:L]).all()
        assert np.isnan(scores[b, L:]).all()


def test_var_constant_window_is_zero() -> None:
    features = np.full((2, 30, 1), 5.0, dtype=np.float32)
    seq_lengths = np.full(2, 30, dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=30)
    assert np.isnan(scores[:, :3]).all()
    assert np.allclose(scores[:, 3:], 0.0, atol=1e-6)


def test_var_linear_trend_is_zero() -> None:
    x = np.arange(30, dtype=np.float32)
    features = (2.0 * x + 1.0).reshape(1, 30, 1)
    seq_lengths = np.full(1, 30, dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=30)
    assert np.allclose(scores[0, 3:], 0.0, atol=1e-4)


def test_var_is_causal() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (2, 50, 1)).astype(np.float32)
    seq_lengths = np.full(2, 50, dtype=np.int64)
    scores_base = _var(base, seq_lengths, window_size=10)
    for t in range(5, 45, 5):
        modified = base.copy()
        modified[:, t + 1 :] = 1000.0 * rng.normal(size=modified[:, t + 1 :].shape)
        scores_mod = _var(modified, seq_lengths, window_size=10)
        assert np.array_equal(np.isnan(scores_base), np.isnan(scores_mod))
        assert np.allclose(scores_base[:, 3 : t + 1], scores_mod[:, 3 : t + 1], atol=1e-6)


def test_var_deterministic() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 1.0, (3, 40, 1)).astype(np.float32)
    seq_lengths = np.full(3, 40, dtype=np.int64)
    a = _var(features, seq_lengths, window_size=15)
    b = _var(features, seq_lengths, window_size=15)
    assert np.array_equal(a, b, equal_nan=True)


def test_var_matches_manual_formula() -> None:
    from csd_observer.models.common.detrend import _linear_detrend

    rng = np.random.default_rng(9)
    n = 30
    x = rng.normal(0.0, 1.0, (4, n))
    seq_lengths = np.full(4, n, dtype=np.int64)
    scores = _var(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    for b in range(4):
        seg = _linear_detrend(x[b])
        expected = float(np.mean((seg - seg.mean()) ** 2))
        assert np.allclose(scores[b, n - 1], expected, atol=1e-5)


def test_var_white_noise_recovers_sigma_sq() -> None:
    rng = np.random.default_rng(13)
    B, W = 200, 100
    x = rng.normal(0.0, 1.0, (B, W))
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _var(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    vals = scores[:, W - 1]
    assert np.allclose(vals.mean(), 1.0 - 2.0 / W, atol=0.05)


def test_var_monotone_in_sigma() -> None:
    rng = np.random.default_rng(17)
    B, W = 200, 30
    means: dict = {}
    for sigma in (0.5, 1.0, 2.0):
        x = rng.normal(0.0, sigma, (B, W))
        seq_lengths = np.full(B, W, dtype=np.int64)
        scores = _var(x[..., None].astype(np.float32), seq_lengths, window_size=W)
        means[sigma] = float(scores[:, W - 1].mean())
    assert means[0.5] < means[1.0] < means[2.0]
    assert np.allclose(means[2.0] / means[1.0], 4.0, rtol=0.15)


def test_var_linear_trend_does_not_inflate() -> None:
    rng = np.random.default_rng(19)
    B, W = 100, 30
    noise = rng.normal(0.0, 1.0, (B, W))
    trend = 0.2 * np.arange(W, dtype=np.float64)[None, :]
    seq_lengths = np.full(B, W, dtype=np.int64)
    s_clean = _var(noise[..., None].astype(np.float32), seq_lengths, window_size=W)
    s_trended = _var((noise + trend)[..., None].astype(np.float32), seq_lengths, window_size=W)
    v_clean = float(s_clean[:, W - 1].mean())
    v_trended = float(s_trended[:, W - 1].mean())
    assert np.allclose(v_clean, 1.0 - 2.0 / W, atol=0.05)
    assert np.allclose(v_trended, v_clean, atol=0.03)
    assert float(np.var(noise + trend)) > 2.0 * v_trended


def test_var_partial_window_equals_whole_prefix() -> None:
    rng = np.random.default_rng(23)
    B, T, W = 2, 60, 30
    features = rng.normal(0.0, 1.0, (B, T, 1)).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=W)
    for t in range(4, W):
        prefix = features[:, : t + 1, :]
        lens = np.full(B, t + 1, dtype=np.int64)
        scores_prefix = _var(prefix, lens, window_size=1000)
        assert np.allclose(scores[:, t], scores_prefix[:, t], atol=1e-6)


def test_var_max_over_channels() -> None:
    rng = np.random.default_rng(29)
    B, T, W = 3, 60, 30
    ch0 = rng.normal(0.0, 1.0, (B, T))
    ch1 = 2.0 * rng.normal(0.0, 1.0, (B, T))
    features = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=W)
    s0 = _var(ch0[..., None], seq_lengths, window_size=W)
    s1 = _var(ch1[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], np.maximum(s0, s1)[:, W - 1 :], atol=1e-6)


def test_var_max_over_channels_ignores_degenerate() -> None:
    rng = np.random.default_rng(31)
    B, T, W = 2, 40, 30
    noisy = rng.normal(0.0, 1.0, (B, T))
    constant = np.full((B, T), 3.0)
    features = np.stack([noisy, constant], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _var(features, seq_lengths, window_size=W)
    s_noisy = _var(noisy[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], s_noisy[:, W - 1 :], atol=1e-6)
