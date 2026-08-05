"""Tests for the RETRATE-CSD baseline (return-rate / recovery-time indicator)."""

from __future__ import annotations

import numpy as np


def _retrate(features, seq_lengths, window_size=30):
    from csd_observer.models.retrate_csd.indicator import raw_retrate_indicator
    return raw_retrate_indicator(features, seq_lengths, window_size)


def _ar1(rng, phi, B, T):
    x = np.empty((B, T))
    e = rng.normal(0.0, 1.0, (B, T))
    x[:, 0] = e[:, 0]
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def test_retrate_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, (4, 50, 1)).astype(np.float32)
    seq_lengths = np.full(4, 50, dtype=np.int64)
    scores = _retrate(features, seq_lengths, window_size=10)
    assert scores.shape == (4, 50)
    assert scores.dtype == np.float32


def test_retrate_invalid_ndim() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be \\(B, T, C\\)"):
        _retrate(np.zeros((4, 50), dtype=np.float32), np.full(4, 50, dtype=np.int64))


def test_retrate_invalid_seq_lengths() -> None:
    import pytest

    features = np.zeros((4, 50, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="seq_lengths length"):
        _retrate(features, np.full(3, 50, dtype=np.int64))


def test_retrate_short_prefix_is_nan() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 1.0, (1, 20, 1)).astype(np.float32)
    seq_lengths = np.full(1, 20, dtype=np.int64)
    scores = _retrate(features, seq_lengths, window_size=30)
    assert np.isnan(scores[0, :3]).all()
    assert np.isfinite(scores[0, 3:]).all()


def test_retrate_beyond_seq_length_is_nan() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(0.0, 1.0, (3, 50, 1)).astype(np.float32)
    seq_lengths = np.array([10, 25, 50], dtype=np.int64)
    scores = _retrate(features, seq_lengths, window_size=30)
    for b, L in enumerate(seq_lengths):
        assert np.isfinite(scores[b, 4:L]).all()
        assert np.isnan(scores[b, L:]).all()


def test_retrate_degenerate_windows_are_nan() -> None:
    constant = np.full((1, 30, 1), 5.0, dtype=np.float32)
    linear = (2.0 * np.arange(30) + 1.0).reshape(1, 30, 1).astype(np.float32)
    seq_lengths = np.full(1, 30, dtype=np.int64)
    assert np.isnan(_retrate(constant, seq_lengths, window_size=30)).all()
    assert np.isnan(_retrate(linear, seq_lengths, window_size=30)).all()


def test_retrate_is_causal() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (2, 50, 1)).astype(np.float32)
    seq_lengths = np.full(2, 50, dtype=np.int64)
    scores_base = _retrate(base, seq_lengths, window_size=10)
    for t in range(5, 45, 5):
        modified = base.copy()
        modified[:, t + 1 :] = 1000.0 * rng.normal(size=modified[:, t + 1 :].shape)
        scores_mod = _retrate(modified, seq_lengths, window_size=10)
        assert np.array_equal(np.isnan(scores_base), np.isnan(scores_mod))
        assert np.allclose(scores_base[:, 3 : t + 1], scores_mod[:, 3 : t + 1], atol=1e-6)


def test_retrate_deterministic() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 1.0, (3, 40, 1)).astype(np.float32)
    seq_lengths = np.full(3, 40, dtype=np.int64)
    a = _retrate(features, seq_lengths, window_size=15)
    b = _retrate(features, seq_lengths, window_size=15)
    assert np.array_equal(a, b, equal_nan=True)


def test_retrate_matches_manual_ols_formula() -> None:
    from csd_observer.utils.metrics import _linear_detrend

    rng = np.random.default_rng(9)
    B, n = 4, 30
    x = _ar1(rng, 0.6, B, n)
    seq_lengths = np.full(B, n, dtype=np.int64)
    scores = _retrate(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    for b in range(B):
        seg = _linear_detrend(x[b])
        seg = seg - seg.mean()
        phi = float(np.dot(seg[:-1], seg[1:])) / float(np.dot(seg[:-1], seg[:-1]))
        expected = float(np.log(max(phi, 1e-4)))
        assert np.allclose(scores[b, n - 1], expected, rtol=1e-5, atol=1e-5)


def test_retrate_uses_ols_not_yule_walker() -> None:
    from csd_observer.utils.metrics import _linear_detrend

    rng = np.random.default_rng(10)
    B, n = 10, 30
    x = _ar1(rng, 0.6, B, n)
    seq_lengths = np.full(B, n, dtype=np.int64)
    scores = _retrate(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    max_gap = 0.0
    for b in range(B):
        seg = _linear_detrend(x[b])
        seg = seg - seg.mean()
        phi_ols = float(np.dot(seg[:-1], seg[1:])) / float(np.dot(seg[:-1], seg[:-1]))
        phi_yw = float(np.dot(seg[:-1], seg[1:])) / float(np.dot(seg, seg))
        score_yw = float(np.log(max(phi_yw, 1e-4)))
        max_gap = max(max_gap, abs(float(scores[b, n - 1]) - score_yw))
        assert np.allclose(scores[b, n - 1], np.log(max(phi_ols, 1e-4)), rtol=1e-5, atol=1e-5)
    assert max_gap > 1e-3


def test_retrate_negative_phi_saturates_at_floor() -> None:
    rng = np.random.default_rng(12)
    B, W = 100, 30
    x = _ar1(rng, -0.8, B, W)
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _retrate(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    floor = float(np.log(1e-4))
    assert np.allclose(scores[:, W - 1], floor, atol=1e-6)


def test_retrate_recovery_time_orders_by_ar1_coefficient() -> None:
    rng = np.random.default_rng(17)
    B, W = 300, 30
    seq_lengths = np.full(B, W, dtype=np.int64)
    means = {}
    for phi in (0.9, 0.5, 0.2, 0.0):
        x = _ar1(rng, phi, B, W)
        scores = _retrate(x[..., None].astype(np.float32), seq_lengths, window_size=W)
        means[phi] = float(scores[:, W - 1].mean())
    assert means[0.9] > means[0.5] + 0.3
    assert means[0.5] > means[0.2] + 0.5
    assert means[0.2] > means[0.0] + 2.0
    assert means[0.0] < -4.0


def test_retrate_partial_window_equals_whole_prefix() -> None:
    rng = np.random.default_rng(23)
    B, T, W = 2, 60, 30
    features = _ar1(rng, 0.6, B, T)[..., None].astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _retrate(features, seq_lengths, window_size=W)
    for t in range(4, W):
        prefix = features[:, : t + 1, :]
        lens = np.full(B, t + 1, dtype=np.int64)
        scores_prefix = _retrate(prefix, lens, window_size=1000)
        assert np.allclose(scores[:, t], scores_prefix[:, t], rtol=1e-5, atol=1e-6)


def test_retrate_max_over_channels() -> None:
    rng = np.random.default_rng(29)
    B, T, W = 3, 60, 30
    ch0 = rng.normal(0.0, 1.0, (B, T))
    ch1 = _ar1(rng, 0.8, B, T)
    features = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _retrate(features, seq_lengths, window_size=W)
    s0 = _retrate(ch0[..., None], seq_lengths, window_size=W)
    s1 = _retrate(ch1[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], np.maximum(s0, s1)[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_retrate_max_over_channels_ignores_degenerate() -> None:
    rng = np.random.default_rng(31)
    B, T, W = 2, 40, 30
    noisy = _ar1(rng, 0.7, B, T)
    constant = np.full((B, T), 3.0)
    features = np.stack([noisy, constant], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _retrate(features, seq_lengths, window_size=W)
    s_noisy = _retrate(noisy[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], s_noisy[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_retrate_float_window_size() -> None:
    rng = np.random.default_rng(33)
    features = rng.normal(0.0, 1.0, (2, 40, 1)).astype(np.float32)
    seq_lengths = np.full(2, 40, dtype=np.int64)
    a = _retrate(features, seq_lengths, window_size=30)
    b = _retrate(features, seq_lengths, window_size=30.0)
    assert np.array_equal(a, b, equal_nan=True)
