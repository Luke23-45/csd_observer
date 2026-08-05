"""Tests for the AC1-CSD baseline (lag-1 Pearson autocorrelation indicator)."""

from __future__ import annotations

import numpy as np


def _ac1(features, seq_lengths, window_size=30):
    from csd_observer.models.ac1_csd.indicator import raw_ac1_indicator
    return raw_ac1_indicator(features, seq_lengths, window_size)


def test_ac1_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, (4, 50, 1)).astype(np.float32)
    seq_lengths = np.full(4, 50, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=10)
    assert scores.shape == (4, 50)
    assert scores.dtype == np.float32


def test_ac1_invalid_ndim() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be \\(B, T, C\\)"):
        _ac1(np.zeros((4, 50), dtype=np.float32), np.full(4, 50, dtype=np.int64))


def test_ac1_invalid_seq_lengths() -> None:
    import pytest

    features = np.zeros((4, 50, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="seq_lengths length"):
        _ac1(features, np.full(3, 50, dtype=np.int64))


def test_ac1_short_prefix_is_nan() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 1.0, (1, 20, 1)).astype(np.float32)
    seq_lengths = np.full(1, 20, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=30)
    assert np.isnan(scores[0, :3]).all()
    assert np.isfinite(scores[0, 3:]).all()


def test_ac1_beyond_seq_length_is_nan() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(0.0, 1.0, (3, 50, 1)).astype(np.float32)
    seq_lengths = np.array([10, 25, 50], dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=30)
    for b, L in enumerate(seq_lengths):
        assert np.isfinite(scores[b, 4:L]).all()
        assert np.isnan(scores[b, L:]).all()


def test_ac1_constant_window_is_nan() -> None:
    features = np.full((2, 30, 1), 5.0, dtype=np.float32)
    seq_lengths = np.full(2, 30, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=30)
    assert np.isnan(scores).all()


def test_ac1_linear_trend_is_nan() -> None:
    x = np.arange(30, dtype=np.float32)
    features = (2.0 * x + 1.0).reshape(1, 30, 1)
    seq_lengths = np.full(1, 30, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=30)
    assert np.isnan(scores).all()


def test_ac1_is_causal() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (2, 50, 1)).astype(np.float32)
    seq_lengths = np.full(2, 50, dtype=np.int64)
    scores_base = _ac1(base, seq_lengths, window_size=10)
    for t in range(5, 45, 5):
        modified = base.copy()
        modified[:, t + 1 :] = 1000.0 * rng.normal(size=modified[:, t + 1 :].shape)
        scores_mod = _ac1(modified, seq_lengths, window_size=10)
        assert np.array_equal(np.isnan(scores_base), np.isnan(scores_mod))
        assert np.allclose(scores_base[:, 3 : t + 1], scores_mod[:, 3 : t + 1], atol=1e-6)


def test_ac1_deterministic() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 1.0, (3, 40, 1)).astype(np.float32)
    seq_lengths = np.full(3, 40, dtype=np.int64)
    a = _ac1(features, seq_lengths, window_size=15)
    b = _ac1(features, seq_lengths, window_size=15)
    assert np.array_equal(a, b, equal_nan=True)


def test_ac1_white_noise_mean_is_zero() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(0.0, 1.0, (5, 200, 1)).astype(np.float32)
    seq_lengths = np.full(5, 200, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=50)
    assert np.allclose(scores[:, 49:].mean(), 0.0, atol=0.12)


def test_ac1_ar1_monotone_in_phi() -> None:
    rng = np.random.default_rng(13)
    B, T, W = 200, 30, 30
    means: dict = {}
    for phi in (0.9, 0.5, 0.1):
        x = np.zeros((B, T))
        eps = rng.normal(0.0, 0.3, (B, T))
        for b in range(B):
            for t in range(1, T):
                x[b, t] = phi * x[b, t - 1] + eps[b, t]
        features = x[..., None].astype(np.float32)
        seq_lengths = np.full(B, T, dtype=np.int64)
        scores = _ac1(features, seq_lengths, window_size=W)
        means[phi] = float(scores[:, T - 1].mean())
    assert means[0.9] > means[0.5] > means[0.1]
    assert means[0.9] - means[0.1] > 0.5


def test_ac1_period2_is_strongly_negative() -> None:
    x = np.tile([1.0, -1.0], 25).astype(np.float32)
    features = x.reshape(1, 50, 1)
    seq_lengths = np.full(1, 50, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=30)
    assert (scores[0, 30:] < -0.9).all()


def test_ac1_sine_period8_recovers_cos() -> None:
    t = np.arange(64, dtype=np.float64)
    x = np.sin(2.0 * np.pi * t / 8.0).astype(np.float32)
    features = x.reshape(1, 64, 1)
    seq_lengths = np.full(1, 64, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=32)
    assert np.allclose(scores[0, 63], np.cos(2.0 * np.pi / 8.0), atol=0.03)


def test_ac1_linear_trend_does_not_inflate() -> None:
    rng = np.random.default_rng(17)
    B, T, W = 8, 120, 30
    noise = rng.normal(0.0, 1.0, (B, T))
    trended = (noise + 0.2 * np.arange(T, dtype=np.float64)[None, :])[..., None]
    trended = trended.astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _ac1(trended, seq_lengths, window_size=W)
    tail = scores[:, W - 1 :]
    assert np.abs(tail).max() < 0.7
    assert np.allclose(tail.mean(), 0.0, atol=0.15)


def test_ac1_is_pearson_not_acf_estimator() -> None:
    rng = np.random.default_rng(21)
    phi = 0.9
    n = 30
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, 0.2)
    from csd_observer.utils.metrics import _linear_detrend

    seg = _linear_detrend(x)
    a = seg[:-1] - seg[:-1].mean()
    b = seg[1:] - seg[1:].mean()
    pearson = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
    acf_est = float(np.dot(seg[:-1], seg[1:]) / np.dot(seg, seg))
    features = x[None, :, None].astype(np.float32)
    seq_lengths = np.full(1, n, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=n)
    assert np.allclose(scores[0, n - 1], pearson, atol=1e-6)
    assert abs(pearson - acf_est) > 1e-3


def test_ac1_partial_window_equals_whole_prefix() -> None:
    rng = np.random.default_rng(23)
    B, T, W = 2, 60, 30
    features = rng.normal(0.0, 1.0, (B, T, 1)).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=W)
    for t in range(4, W):
        prefix = features[:, : t + 1, :]
        lens = np.full(B, t + 1, dtype=np.int64)
        scores_prefix = _ac1(prefix, lens, window_size=1000)
        assert np.allclose(scores[:, t], scores_prefix[:, t], atol=1e-6)


def test_ac1_max_over_channels() -> None:
    rng = np.random.default_rng(29)
    B, T, W = 3, 60, 30
    ch0 = rng.normal(0.0, 1.0, (B, T))
    phi = 0.8
    ch1 = np.zeros((B, T))
    for b in range(B):
        for t in range(1, T):
            ch1[b, t] = phi * ch1[b, t - 1] + rng.normal(0.0, 0.3)
    features = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=W)
    s0 = _ac1(ch0[..., None], seq_lengths, window_size=W)
    s1 = _ac1(ch1[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], np.maximum(s0, s1)[:, W - 1 :], atol=1e-6)


def test_ac1_max_over_channels_ignores_degenerate() -> None:
    rng = np.random.default_rng(31)
    B, T, W = 2, 40, 30
    noisy = rng.normal(0.0, 1.0, (B, T))
    constant = np.full((B, T), 3.0)
    features = np.stack([noisy, constant], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _ac1(features, seq_lengths, window_size=W)
    s_noisy = _ac1(noisy[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], s_noisy[:, W - 1 :], atol=1e-6)
