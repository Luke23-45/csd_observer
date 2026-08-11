"""Tests for the SKEW-CSD baseline (absolute-skewness indicator)."""

from __future__ import annotations

import numpy as np


def _skew(features, seq_lengths, window_size=30):
    from csd_observer.models.indicators.skew_csd.indicator import raw_skew_indicator
    return raw_skew_indicator(features, seq_lengths, window_size)


def test_skew_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, (4, 50, 1)).astype(np.float32)
    seq_lengths = np.full(4, 50, dtype=np.int64)
    scores = _skew(features, seq_lengths, window_size=10)
    assert scores.shape == (4, 50)
    assert scores.dtype == np.float32


def test_skew_invalid_ndim() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be \\(B, T, C\\)"):
        _skew(np.zeros((4, 50), dtype=np.float32), np.full(4, 50, dtype=np.int64))


def test_skew_invalid_seq_lengths() -> None:
    import pytest

    features = np.zeros((4, 50, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="seq_lengths length"):
        _skew(features, np.full(3, 50, dtype=np.int64))


def test_skew_short_prefix_is_nan() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 1.0, (1, 20, 1)).astype(np.float32)
    seq_lengths = np.full(1, 20, dtype=np.int64)
    scores = _skew(features, seq_lengths, window_size=30)
    assert np.isnan(scores[0, :3]).all()
    assert np.isfinite(scores[0, 3:]).all()


def test_skew_beyond_seq_length_is_nan() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(0.0, 1.0, (3, 50, 1)).astype(np.float32)
    seq_lengths = np.array([10, 25, 50], dtype=np.int64)
    scores = _skew(features, seq_lengths, window_size=30)
    for b, L in enumerate(seq_lengths):
        assert np.isfinite(scores[b, 4:L]).all()
        assert np.isnan(scores[b, L:]).all()


def test_skew_degenerate_windows_are_nan() -> None:
    constant = np.full((1, 30, 1), 5.0, dtype=np.float32)
    linear = (2.0 * np.arange(30) + 1.0).reshape(1, 30, 1).astype(np.float32)
    seq_lengths = np.full(1, 30, dtype=np.int64)
    assert np.isnan(_skew(constant, seq_lengths, window_size=30)).all()
    assert np.isnan(_skew(linear, seq_lengths, window_size=30)).all()


def test_skew_is_causal() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (2, 50, 1)).astype(np.float32)
    seq_lengths = np.full(2, 50, dtype=np.int64)
    scores_base = _skew(base, seq_lengths, window_size=10)
    for t in range(5, 45, 5):
        modified = base.copy()
        modified[:, t + 1 :] = 1000.0 * rng.normal(size=modified[:, t + 1 :].shape)
        scores_mod = _skew(modified, seq_lengths, window_size=10)
        assert np.array_equal(np.isnan(scores_base), np.isnan(scores_mod))
        assert np.allclose(scores_base[:, 3 : t + 1], scores_mod[:, 3 : t + 1], atol=1e-6)


def test_skew_deterministic() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 1.0, (3, 40, 1)).astype(np.float32)
    seq_lengths = np.full(3, 40, dtype=np.int64)
    a = _skew(features, seq_lengths, window_size=15)
    b = _skew(features, seq_lengths, window_size=15)
    assert np.array_equal(a, b, equal_nan=True)


def test_skew_matches_manual_formula() -> None:
    from csd_observer.models.common.detrend import _linear_detrend

    rng = np.random.default_rng(9)
    n = 30
    x = rng.gamma(2.0, 1.0, (4, n))
    seq_lengths = np.full(4, n, dtype=np.int64)
    scores = _skew(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    for b in range(4):
        seg = _linear_detrend(x[b])
        seg = seg - seg.mean()
        m2 = float(np.mean(seg * seg))
        m3 = float(np.mean(seg * seg * seg))
        expected = abs(m3 / (m2 ** 1.5))
        assert np.allclose(scores[b, n - 1], expected, atol=1e-5)


def test_skew_sign_invariant() -> None:
    rng = np.random.default_rng(13)
    B, T, W = 2, 60, 30
    x = rng.gamma(2.0, 1.0, (B, T))
    seq_lengths = np.full(B, T, dtype=np.int64)
    pos = _skew(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    neg = _skew((-x)[..., None].astype(np.float32), seq_lengths, window_size=W)
    assert np.allclose(pos, neg, atol=1e-5, equal_nan=True)


def test_skew_normal_is_small() -> None:
    rng = np.random.default_rng(17)
    B, W = 200, 100
    x = rng.normal(0.0, 1.0, (B, W))
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _skew(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    mean_abs = float(scores[:, W - 1].mean())
    assert 0.1 < mean_abs < 0.3


def test_skew_skewed_is_large() -> None:
    rng = np.random.default_rng(19)
    B, W = 200, 100
    x = rng.gamma(1.0, 1.0, (B, W))
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _skew(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    mean_abs = float(scores[:, W - 1].mean())
    assert mean_abs > 1.5


def test_skew_more_skewed_beats_less_skewed() -> None:
    rng = np.random.default_rng(21)
    B, W = 200, 100
    x_mild = rng.gamma(5.0, 1.0, (B, W))
    x_strong = rng.gamma(0.5, 1.0, (B, W))
    seq_lengths = np.full(B, W, dtype=np.int64)
    s_mild = float(_skew(x_mild[..., None].astype(np.float32), seq_lengths, window_size=W)[:, W - 1].mean())
    s_strong = float(_skew(x_strong[..., None].astype(np.float32), seq_lengths, window_size=W)[:, W - 1].mean())
    assert s_strong > s_mild


def test_skew_partial_window_equals_whole_prefix() -> None:
    rng = np.random.default_rng(23)
    B, T, W = 2, 60, 30
    features = rng.gamma(2.0, 1.0, (B, T, 1)).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _skew(features, seq_lengths, window_size=W)
    for t in range(4, W):
        prefix = features[:, : t + 1, :]
        lens = np.full(B, t + 1, dtype=np.int64)
        scores_prefix = _skew(prefix, lens, window_size=1000)
        assert np.allclose(scores[:, t], scores_prefix[:, t], atol=1e-6)


def test_skew_max_over_channels() -> None:
    rng = np.random.default_rng(29)
    B, T, W = 3, 60, 30
    ch0 = rng.normal(0.0, 1.0, (B, T))
    ch1 = rng.gamma(0.5, 1.0, (B, T))
    features = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _skew(features, seq_lengths, window_size=W)
    s0 = _skew(ch0[..., None], seq_lengths, window_size=W)
    s1 = _skew(ch1[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], np.maximum(s0, s1)[:, W - 1 :], atol=1e-6)


def test_skew_max_over_channels_ignores_degenerate() -> None:
    rng = np.random.default_rng(31)
    B, T, W = 2, 40, 30
    noisy = rng.gamma(0.5, 1.0, (B, T))
    constant = np.full((B, T), 3.0)
    features = np.stack([noisy, constant], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _skew(features, seq_lengths, window_size=W)
    s_noisy = _skew(noisy[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], s_noisy[:, W - 1 :], atol=1e-6)
