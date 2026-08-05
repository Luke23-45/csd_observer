"""Tests for the SRATIO-CSD baseline (spectral-density-ratio indicator)."""

from __future__ import annotations

import numpy as np


def _sratio(features, seq_lengths, window_size=30):
    from csd_observer.models.sratio_csd.indicator import raw_sratio_indicator
    return raw_sratio_indicator(features, seq_lengths, window_size)


def _ar1(rng, phi, B, T):
    x = np.empty((B, T))
    e = rng.normal(0.0, 1.0, (B, T))
    x[:, 0] = e[:, 0]
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def test_sratio_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, (4, 50, 1)).astype(np.float32)
    seq_lengths = np.full(4, 50, dtype=np.int64)
    scores = _sratio(features, seq_lengths, window_size=10)
    assert scores.shape == (4, 50)
    assert scores.dtype == np.float32


def test_sratio_invalid_ndim() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be \\(B, T, C\\)"):
        _sratio(np.zeros((4, 50), dtype=np.float32), np.full(4, 50, dtype=np.int64))


def test_sratio_invalid_seq_lengths() -> None:
    import pytest

    features = np.zeros((4, 50, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="seq_lengths length"):
        _sratio(features, np.full(3, 50, dtype=np.int64))


def test_sratio_short_prefix_is_nan() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 1.0, (1, 20, 1)).astype(np.float32)
    seq_lengths = np.full(1, 20, dtype=np.int64)
    scores = _sratio(features, seq_lengths, window_size=30)
    assert np.isnan(scores[0, :3]).all()
    assert np.isfinite(scores[0, 3:]).all()


def test_sratio_beyond_seq_length_is_nan() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(0.0, 1.0, (3, 50, 1)).astype(np.float32)
    seq_lengths = np.array([10, 25, 50], dtype=np.int64)
    scores = _sratio(features, seq_lengths, window_size=30)
    for b, L in enumerate(seq_lengths):
        assert np.isfinite(scores[b, 4:L]).all()
        assert np.isnan(scores[b, L:]).all()


def test_sratio_degenerate_windows_are_nan() -> None:
    constant = np.full((1, 30, 1), 5.0, dtype=np.float32)
    linear = (2.0 * np.arange(30) + 1.0).reshape(1, 30, 1).astype(np.float32)
    seq_lengths = np.full(1, 30, dtype=np.int64)
    assert np.isnan(_sratio(constant, seq_lengths, window_size=30)).all()
    assert np.isnan(_sratio(linear, seq_lengths, window_size=30)).all()


def test_sratio_tiny_window_size_is_nan() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(0.0, 1.0, (2, 30, 1)).astype(np.float32)
    seq_lengths = np.full(2, 30, dtype=np.int64)
    for w in (1, 2, 3):
        assert np.isnan(_sratio(features, seq_lengths, window_size=w)).all()


def test_sratio_is_causal() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (2, 50, 1)).astype(np.float32)
    seq_lengths = np.full(2, 50, dtype=np.int64)
    scores_base = _sratio(base, seq_lengths, window_size=10)
    for t in range(5, 45, 5):
        modified = base.copy()
        modified[:, t + 1 :] = 1000.0 * rng.normal(size=modified[:, t + 1 :].shape)
        scores_mod = _sratio(modified, seq_lengths, window_size=10)
        assert np.array_equal(np.isnan(scores_base), np.isnan(scores_mod))
        assert np.allclose(scores_base[:, 3 : t + 1], scores_mod[:, 3 : t + 1], atol=1e-6)


def test_sratio_deterministic() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 1.0, (3, 40, 1)).astype(np.float32)
    seq_lengths = np.full(3, 40, dtype=np.int64)
    a = _sratio(features, seq_lengths, window_size=15)
    b = _sratio(features, seq_lengths, window_size=15)
    assert np.array_equal(a, b, equal_nan=True)


def test_sratio_matches_manual_formula() -> None:
    from csd_observer.utils.metrics import _linear_detrend

    rng = np.random.default_rng(9)
    B, n = 4, 30
    x = _ar1(rng, 0.6, B, n)
    seq_lengths = np.full(B, n, dtype=np.int64)
    scores = _sratio(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    for b in range(B):
        seg = _linear_detrend(x[b])
        seg = seg - seg.mean()
        phi = float(np.dot(seg[:-1], seg[1:])) / float(np.dot(seg, seg))
        freqs = np.linspace(0.0, 0.5, n)
        shape = 1.0 / (1.0 + phi * phi - 2.0 * phi * np.cos(2.0 * np.pi * freqs))
        expected = float(shape[1] / shape[-1])
        assert np.allclose(scores[b, n - 1], expected, rtol=1e-5, atol=1e-5)


def test_sratio_white_noise_ratio_is_one() -> None:
    rng = np.random.default_rng(15)
    B, W = 300, 30
    x = rng.normal(0.0, 1.0, (B, W))
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _sratio(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    mean_ratio = float(scores[:, W - 1].mean())
    assert 0.9 < mean_ratio < 1.1


def test_sratio_reddening_orders_by_ar1_coefficient() -> None:
    rng = np.random.default_rng(17)
    B, W = 300, 30
    seq_lengths = np.full(B, W, dtype=np.int64)
    g = {}
    for phi in (0.9, 0.5, 0.1, -0.5):
        x = _ar1(rng, phi, B, W)
        scores = _sratio(x[..., None].astype(np.float32), seq_lengths, window_size=W)
        g[phi] = float(np.log10(scores[:, W - 1]).mean())
    assert g[0.9] > 1.2
    assert g[0.9] > g[0.5] + 0.5
    assert g[0.5] > g[0.1] + 0.3
    assert g[0.1] > g[-0.5] + 0.5
    assert g[-0.5] < -0.5


def test_sratio_negative_phi_gives_ratio_below_one() -> None:
    rng = np.random.default_rng(18)
    B, W = 300, 30
    x = _ar1(rng, -0.6, B, W)
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _sratio(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    assert float(scores[:, W - 1].mean()) < 0.9


def test_sratio_partial_window_equals_whole_prefix() -> None:
    rng = np.random.default_rng(23)
    B, T, W = 2, 60, 30
    features = _ar1(rng, 0.7, B, T)[..., None].astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _sratio(features, seq_lengths, window_size=W)
    for t in range(4, W):
        prefix = features[:, : t + 1, :]
        lens = np.full(B, t + 1, dtype=np.int64)
        scores_prefix = _sratio(prefix, lens, window_size=1000)
        assert np.allclose(scores[:, t], scores_prefix[:, t], rtol=1e-5, atol=1e-6)


def test_sratio_window_size_changes_scores() -> None:
    rng = np.random.default_rng(25)
    B, T = 2, 60
    x = _ar1(rng, 0.7, B, T)[..., None].astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    s30 = _sratio(x, seq_lengths, window_size=30)
    s50 = _sratio(x, seq_lengths, window_size=50)
    s20 = _sratio(x, seq_lengths, window_size=20)
    assert np.isfinite(s30[:, T - 1]).all()
    assert np.isfinite(s50[:, T - 1]).all()
    assert np.isfinite(s20[:, T - 1]).all()
    assert not np.allclose(s30[:, T - 1], s50[:, T - 1], rtol=1e-5, atol=1e-6)
    assert not np.allclose(s30[:, T - 1], s20[:, T - 1], rtol=1e-5, atol=1e-6)


def test_sratio_max_over_channels() -> None:
    rng = np.random.default_rng(29)
    B, T, W = 3, 60, 30
    ch0 = rng.normal(0.0, 1.0, (B, T))
    ch1 = _ar1(rng, 0.8, B, T)
    features = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _sratio(features, seq_lengths, window_size=W)
    s0 = _sratio(ch0[..., None], seq_lengths, window_size=W)
    s1 = _sratio(ch1[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], np.maximum(s0, s1)[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_sratio_max_over_channels_ignores_degenerate() -> None:
    rng = np.random.default_rng(31)
    B, T, W = 2, 40, 30
    noisy = _ar1(rng, 0.7, B, T)
    constant = np.full((B, T), 3.0)
    features = np.stack([noisy, constant], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _sratio(features, seq_lengths, window_size=W)
    s_noisy = _sratio(noisy[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], s_noisy[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_sratio_float_window_size() -> None:
    rng = np.random.default_rng(33)
    features = rng.normal(0.0, 1.0, (2, 40, 1)).astype(np.float32)
    seq_lengths = np.full(2, 40, dtype=np.int64)
    a = _sratio(features, seq_lengths, window_size=30)
    b = _sratio(features, seq_lengths, window_size=30.0)
    assert np.array_equal(a, b, equal_nan=True)
