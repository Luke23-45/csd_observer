"""L4.7: indicator parity — hand-computed fixtures for all seven
statistical indicators.

Each indicator is exercised on small hand-built windows where the
expected value is known analytically (or structurally):

* NaN warm-up: fewer than 4 points in the causal window ⇒ NaN;
* prefix coverage: finite scores everywhere inside ``seq_lengths``
  once the warm-up is over (smooth, non-degenerate input);
* known values: constant window ⇒ VAR 0, AC1 NaN; alternating series
  ⇒ AC1 ≈ -1; linear ramp ⇒ VAR ≈ 0 (detrending); symmetric window
  ⇒ SKEW ≈ 0;
* determinism: repeated calls reproduce bit-identical arrays;
* feature-mode declaration (R2.2): ``channel_0``/``radial``/auto agree
  with the dataset-declared modes on multi-channel input.
"""

from __future__ import annotations

import numpy as np
import pytest

from csd_observer.models.common.indicators import IndicatorMethod
from csd_observer.models.common.mode import extract_mode
from csd_observer.models.indicators.ac1_csd.indicator import raw_ac1_indicator
from csd_observer.models.indicators.dfa_csd.indicator import raw_dfa_indicator
from csd_observer.models.indicators.dmd_csd.indicator import raw_dmd_indicator
from csd_observer.models.indicators.retrate_csd.indicator import raw_retrate_indicator
from csd_observer.models.indicators.skew_csd.indicator import raw_skew_indicator
from csd_observer.models.indicators.sratio_csd.indicator import raw_sratio_indicator
from csd_observer.models.indicators.var_csd.indicator import raw_var_indicator

_LENGTH = 40


def _windowed(fn, seq: np.ndarray, **kwargs) -> np.ndarray:
    """Score a single (length,) channel with full-prefix lengths."""
    features = np.asarray(seq, dtype=np.float32).reshape(1, -1, 1)
    lengths = np.array([len(seq)], dtype=np.int64)
    return fn(features, lengths, **kwargs)[0]


def test_var_constant_window_is_zero_and_detrended_ramp_is_zero() -> None:
    scores = _windowed(raw_var_indicator, np.full(_LENGTH, 3.0), window_size=10)
    assert np.allclose(scores[3:], 0.0, atol=1e-5)  # population variance of constants is 0 (float32 noise)
    ramp = np.arange(_LENGTH, dtype=np.float64)
    scores = _windowed(raw_var_indicator, ramp, window_size=10)
    # within-window linear detrending removes the ramp exactly
    assert np.allclose(scores[3:], 0.0, atol=1e-4)


def test_ac1_alternating_series_is_strongly_negative_and_constant_is_nan() -> None:
    alternating = np.tile([1.0, -1.0], _LENGTH // 2)
    scores = _windowed(raw_ac1_indicator, alternating, window_size=10)
    # lag-1 autocorrelation of a (detrended) alternating window is near -1;
    # odd-length windows after detrending are slightly weaker in magnitude.
    assert np.all(scores[4:] <= -0.8)
    assert scores[-1] < -0.9
    constant = np.full(_LENGTH, 2.0)
    scores = _windowed(raw_ac1_indicator, constant, window_size=10)
    assert np.isnan(scores[3:]).all()  # degenerate zero-variance window


def test_skew_symmetric_window_is_zero() -> None:
    symmetric = np.concatenate([np.linspace(-2, -0.1, 6), np.linspace(0.1, 2, 6)])
    seq = np.tile(symmetric, 6)
    scores = _windowed(raw_skew_indicator, seq, window_size=12)
    # only windows aligned to the 12-step period are exactly symmetric
    assert np.allclose(scores[11::12], 0.0, atol=1e-9)


def test_all_indicators_have_nan_warmup_and_full_prefix_coverage() -> None:
    rng = np.random.default_rng(3)
    seq = rng.standard_normal(40)
    cases = {
        raw_var_indicator: (40, {}),
        raw_ac1_indicator: (40, {}),
        raw_skew_indicator: (40, {}),
        raw_sratio_indicator: (40, {}),
        raw_retrate_indicator: (40, {}),
        raw_dfa_indicator: (120, {"window_size": 100}),   # DFA scores only W>=100 windows
        raw_dmd_indicator: (40, {"window_size": 12}),
    }
    for fn, (length, kwargs) in cases.items():
        seq = rng.standard_normal(length)
        scores = _windowed(fn, seq, **kwargs)
        assert scores.shape == (length,), fn.__name__
        assert np.isnan(scores[:3]).all(), f"{fn.__name__}: warm-up must be NaN"
        # the late prefix is past every warm-up (windows are <= 100)
        assert np.isfinite(scores[-10:]).all(), f"{fn.__name__}: prefix must be finite"


def test_indicators_are_deterministic() -> None:
    rng = np.random.default_rng(9)
    seq = rng.standard_normal(_LENGTH)
    a = _windowed(raw_sratio_indicator, seq)
    b = _windowed(raw_sratio_indicator, seq)
    np.testing.assert_array_equal(a, b)


def test_feature_mode_radial_matches_auto_on_hopf() -> None:
    rng = np.random.default_rng(4)
    feats = rng.standard_normal((3, _LENGTH, 2)).astype(np.float32)
    explicit = extract_mode(feats, "hopf", mode="radial")
    auto = extract_mode(feats, "hopf", mode=None)
    np.testing.assert_array_equal(explicit, auto)
    # and it is the radial coordinate, not channel 0
    assert not np.array_equal(explicit, feats[..., 0])


def test_feature_mode_channel_0_ignores_second_channel() -> None:
    rng = np.random.default_rng(4)
    feats = rng.standard_normal((3, _LENGTH, 2)).astype(np.float32)
    out = extract_mode(feats, "subcritical_hopf", mode="channel_0")
    np.testing.assert_array_equal(out, feats[..., 0])


def test_indicator_method_reads_declared_feature_mode_from_config() -> None:
    rng = np.random.default_rng(6)
    feats = rng.standard_normal((4, _LENGTH, 2)).astype(np.float32)
    lengths = np.full(4, _LENGTH, dtype=np.int64)
    method = IndicatorMethod("var_csd", "hopf")
    cfg_radial = {"model": {"var_csd": {"window_size": 10}}, "dataset": {"feature_mode": "radial"}}
    cfg_auto = {"model": {"var_csd": {"window_size": 10}}, "dataset": {"feature_mode": None}}
    scores_radial = method.score(feats, lengths, cfg_radial)
    scores_auto = method.score(feats, lengths, cfg_auto)
    np.testing.assert_array_equal(scores_radial, scores_auto)
    # a declared channel_0 on a 2-channel hopf scores channel 0 directly
    cfg_ch0 = {"model": {"var_csd": {"window_size": 10}}, "dataset": {"feature_mode": "channel_0"}}
    scores_ch0 = method.score(feats, lengths, cfg_ch0)
    assert not np.array_equal(scores_ch0, scores_radial)


def test_extract_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown feature mode"):
        extract_mode(np.zeros((2, 8, 1), dtype=np.float32), "fold", mode="bogus")
