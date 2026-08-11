"""Tests for the DFA-CSD baseline (detrended-fluctuation-analysis exponent)."""

from __future__ import annotations

import numpy as np


def _dfa(features, seq_lengths, window_size=100):
    from csd_observer.models.indicators.dfa_csd.indicator import raw_dfa_indicator
    return raw_dfa_indicator(features, seq_lengths, window_size)


def _manual_dfa_alpha(seg, box_sizes):
    from csd_observer.models.common.detrend import _linear_detrend

    seg = _linear_detrend(seg)
    seg = seg - seg.mean()
    profile = np.cumsum(seg)
    n = len(seg)
    fluct = []
    for s in box_sizes:
        n_seg = n // s
        variances = []
        for j in range(n_seg):
            for part in (profile[j * s : (j + 1) * s], profile[n - (j + 1) * s : n - j * s]):
                x = np.arange(s, dtype=np.float64)
                X = np.vstack([x, np.ones(s)]).T
                coeffs, _, _, _ = np.linalg.lstsq(X, part, rcond=None)
                variances.append(float(np.mean((part - X @ coeffs) ** 2)))
        fluct.append(float(np.sqrt(np.mean(variances))))
    return float(np.polyfit(np.log(box_sizes), np.log(fluct), 1)[0])


def _ar1(rng, phi, B, T):
    x = np.empty((B, T))
    e = rng.normal(0.0, 1.0, (B, T))
    x[:, 0] = e[:, 0]
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def test_dfa_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, (4, 120, 1)).astype(np.float32)
    seq_lengths = np.full(4, 120, dtype=np.int64)
    scores = _dfa(features, seq_lengths)
    assert scores.shape == (4, 120)
    assert scores.dtype == np.float32


def test_dfa_invalid_ndim() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be \\(B, T, C\\)"):
        _dfa(np.zeros((4, 120), dtype=np.float32), np.full(4, 120, dtype=np.int64))


def test_dfa_invalid_seq_lengths() -> None:
    import pytest

    features = np.zeros((4, 120, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="seq_lengths length"):
        _dfa(features, np.full(3, 120, dtype=np.int64))


def test_dfa_nan_until_window_fills() -> None:
    rng = np.random.default_rng(7)
    B, T, W = 2, 120, 100
    features = rng.normal(0.0, 1.0, (B, T, 1)).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _dfa(features, seq_lengths, window_size=W)
    assert np.isnan(scores[:, : W - 1]).all()
    assert np.isfinite(scores[:, W - 1 :]).all()


def test_dfa_window_below_100_is_all_nan() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(0.0, 1.0, (2, 120, 1)).astype(np.float32)
    seq_lengths = np.full(2, 120, dtype=np.int64)
    assert np.isnan(_dfa(features, seq_lengths, window_size=30)).all()
    assert np.isnan(_dfa(features, seq_lengths, window_size=99)).all()


def test_dfa_short_trajectory_is_all_nan() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(0.0, 1.0, (2, 60, 1)).astype(np.float32)
    seq_lengths = np.full(2, 60, dtype=np.int64)
    assert np.isnan(_dfa(features, seq_lengths, window_size=100)).all()


def test_dfa_beyond_seq_length_is_nan() -> None:
    rng = np.random.default_rng(11)
    B, T = 3, 130
    features = rng.normal(0.0, 1.0, (B, T, 1)).astype(np.float32)
    seq_lengths = np.array([110, 120, 130], dtype=np.int64)
    scores = _dfa(features, seq_lengths, window_size=100)
    for b, L in enumerate(seq_lengths):
        assert np.isfinite(scores[b, 99:L]).all()
        assert np.isnan(scores[b, L:]).all()


def test_dfa_degenerate_windows_are_nan() -> None:
    constant = np.full((1, 120, 1), 5.0, dtype=np.float32)
    linear = (2.0 * np.arange(120) + 1.0).reshape(1, 120, 1).astype(np.float32)
    seq_lengths = np.full(1, 120, dtype=np.int64)
    assert np.isnan(_dfa(constant, seq_lengths, window_size=100)).all()
    assert np.isnan(_dfa(linear, seq_lengths, window_size=100)).all()


def test_dfa_is_causal() -> None:
    rng = np.random.default_rng(3)
    B, T, W = 2, 130, 100
    base = rng.normal(0.0, 1.0, (B, T, 1)).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores_base = _dfa(base, seq_lengths, window_size=W)
    for t in range(105, 125, 5):
        modified = base.copy()
        modified[:, t + 1 :] = 1000.0 * rng.normal(size=modified[:, t + 1 :].shape)
        scores_mod = _dfa(modified, seq_lengths, window_size=W)
        assert np.array_equal(np.isnan(scores_base), np.isnan(scores_mod))
        assert np.allclose(scores_base[:, W - 1 : t + 1], scores_mod[:, W - 1 : t + 1], atol=1e-6)


def test_dfa_deterministic() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 1.0, (3, 110, 1)).astype(np.float32)
    seq_lengths = np.full(3, 110, dtype=np.int64)
    a = _dfa(features, seq_lengths, window_size=100)
    b = _dfa(features, seq_lengths, window_size=100)
    assert np.array_equal(a, b, equal_nan=True)


def test_dfa_matches_manual_formula() -> None:
    rng = np.random.default_rng(9)
    B, n = 4, 100
    x = _ar1(rng, 0.5, B, n)
    seq_lengths = np.full(B, n, dtype=np.int64)
    scores = _dfa(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    for b in range(B):
        expected = _manual_dfa_alpha(x[b], (10, 20, 40, 80))
        assert np.allclose(scores[b, n - 1], expected, rtol=1e-5, atol=1e-5)


def test_dfa_white_noise_alpha_is_half() -> None:
    rng = np.random.default_rng(15)
    B, W = 200, 100
    x = rng.normal(0.0, 1.0, (B, W))
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _dfa(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    mean_alpha = float(scores[:, W - 1].mean())
    assert 0.4 < mean_alpha < 0.6


def test_dfa_brownian_alpha_is_super_diffusive() -> None:
    rng = np.random.default_rng(16)
    B, W = 200, 100
    x = np.cumsum(rng.normal(0.0, 1.0, (B, W)), axis=1)
    seq_lengths = np.full(B, W, dtype=np.int64)
    scores = _dfa(x[..., None].astype(np.float32), seq_lengths, window_size=W)
    mean_alpha = float(scores[:, W - 1].mean())
    assert mean_alpha > 1.05


def test_dfa_reddening_orders_by_ar1_coefficient() -> None:
    rng = np.random.default_rng(17)
    B, W = 200, 100
    seq_lengths = np.full(B, W, dtype=np.int64)
    means = {}
    for phi in (0.9, 0.5, 0.0):
        x = _ar1(rng, phi, B, W)
        scores = _dfa(x[..., None].astype(np.float32), seq_lengths, window_size=W)
        means[phi] = float(scores[:, W - 1].mean())
    assert means[0.9] > means[0.5] + 0.1
    assert means[0.5] > means[0.0] + 0.05


def test_dfa_partial_prefix_equals_whole_run() -> None:
    rng = np.random.default_rng(23)
    B, T, W = 2, 130, 100
    features = _ar1(rng, 0.6, B, T)[..., None].astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _dfa(features, seq_lengths, window_size=W)
    for t in range(W + 5, T - 5, 7):
        prefix = features[:, : t + 1, :]
        lens = np.full(B, t + 1, dtype=np.int64)
        scores_prefix = _dfa(prefix, lens, window_size=W)
        assert np.allclose(scores[:, t], scores_prefix[:, t], rtol=1e-5, atol=1e-6)


def test_dfa_box_sizes_scale_with_window() -> None:
    rng = np.random.default_rng(25)
    B, n = 3, 200
    x = _ar1(rng, 0.6, B, n)
    seq_lengths = np.full(B, n, dtype=np.int64)
    scores = _dfa(x[..., None].astype(np.float32), seq_lengths, window_size=n)
    for b in range(B):
        expected = _manual_dfa_alpha(x[b], (20, 40, 80, 160))
        assert np.allclose(scores[b, n - 1], expected, rtol=1e-5, atol=1e-5)


def test_dfa_window_size_changes_scores() -> None:
    rng = np.random.default_rng(27)
    B, T = 2, 220
    x = _ar1(rng, 0.7, B, T)[..., None].astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    s100 = _dfa(x, seq_lengths, window_size=100)
    s200 = _dfa(x, seq_lengths, window_size=200)
    assert np.isfinite(s100[:, 199:]).all()
    assert np.isfinite(s200[:, 199:]).all()
    assert np.isnan(s100[:, :99]).all()
    assert np.isnan(s200[:, :199]).all()
    assert not np.allclose(s100[:, 199], s200[:, 199], rtol=1e-5, atol=1e-6)


def test_dfa_max_over_channels() -> None:
    rng = np.random.default_rng(29)
    B, T, W = 3, 120, 100
    ch0 = rng.normal(0.0, 1.0, (B, T))
    ch1 = _ar1(rng, 0.8, B, T)
    features = np.stack([ch0, ch1], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _dfa(features, seq_lengths, window_size=W)
    s0 = _dfa(ch0[..., None], seq_lengths, window_size=W)
    s1 = _dfa(ch1[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], np.maximum(s0, s1)[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_dfa_max_over_channels_ignores_degenerate() -> None:
    rng = np.random.default_rng(31)
    B, T, W = 2, 120, 100
    noisy = _ar1(rng, 0.7, B, T)
    constant = np.full((B, T), 3.0)
    features = np.stack([noisy, constant], axis=-1).astype(np.float32)
    seq_lengths = np.full(B, T, dtype=np.int64)
    scores = _dfa(features, seq_lengths, window_size=W)
    s_noisy = _dfa(noisy[..., None], seq_lengths, window_size=W)
    assert np.allclose(scores[:, W - 1 :], s_noisy[:, W - 1 :], rtol=1e-5, atol=1e-6)


def test_dfa_float_window_size() -> None:
    rng = np.random.default_rng(33)
    features = rng.normal(0.0, 1.0, (2, 110, 1)).astype(np.float32)
    seq_lengths = np.full(2, 110, dtype=np.int64)
    a = _dfa(features, seq_lengths, window_size=100)
    b = _dfa(features, seq_lengths, window_size=100.0)
    assert np.array_equal(a, b, equal_nan=True)
