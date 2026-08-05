"""Absolute skewness (SKEW) critical-slowing-down baseline.

Definition (benchmark plan §3, row SKEW): the **absolute value of the
third standardized moment** ``abs(g1)`` of the within-window linearly
detrended causal window, with population central moments
``g1 = m3 / m2^1.5``.

References:

* Dakos et al. (2012), *PLoS ONE* 7(7): e41010 — skewness of the
  detrended data within rolling windows.
* ``earlywarnings`` (R) ``sk`` indicator: ``abs(skewness(window,
  na.rm = TRUE))``, where ``skewness`` is ``moments::skewness``, i.e.
  the population third standardized moment ``(sum(x-xbar)^3/n) /
  (sum(x-xbar)^2/n)^(3/2)`` — the formula we implement verbatim on
  the detrended window.
* Guttal & Jayaprakash (2008), *Ecol. Lett.* 11: 450–460 — changing
  (signed) skewness as an early warning signal; Scheffer et al.
  (2009), *Nature* 461: 53–59 — increased asymmetry of fluctuations
  before catastrophic bifurcations.
* ewstools (Bury et al. 2021) computes the Fisher–Pearson corrected
  skewness (``pandas`` rolling ``skew``); for a fixed window size that
  is a constant positive multiple of ``g1``, hence rank-equivalent
  under the benchmark's fixed-FPR calibration (plan §4). We follow
  the ``earlywarnings``/plan definition.

Direction convention: the raw indicator value is the alarm score
(higher score => higher alarm). On fold systems the *signed* skewness
moves away from zero as the unstable equilibrium approaches the
attractor; the absolute transform is the canonical handling of the
``earlywarnings`` toolbox (plan §3) and is sign-agnostic by design.

The indicator is deterministic and has no parameters.
"""

from __future__ import annotations

import numpy as np

from csd_observer.utils.metrics import _linear_detrend

_MIN_POINTS = 4


def raw_skew_indicator(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 30,
) -> np.ndarray:
    """Absolute skewness of causal linearly detrended windows.

    For each trajectory ``b`` and step ``t`` the score is
    ``abs(m3 / m2^1.5)`` (population central moments) of the causal
    window ``x[max(0, t-W+1) : t+1]`` after within-window linear
    detrending (``_linear_detrend``, ``metrics.py``). The window ends
    at ``t`` and has effective size ``min(W, t+1)``, i.e. every step
    gets a score once enough data has accumulated (plan §4 — same
    causal convention as ``running_mean_center``).

    Scores are undefined (``NaN``) when fewer than 4 points are
    available in the causal window (the same guard as the other
    window-30 baselines), when the detrended window has zero variance
    (``g1`` is 0/0 — constant or perfectly linear windows), or when
    ``t >= seq_lengths[b]`` (beyond the valid prefix).

    For multi-channel inputs the channel-wise scores are combined by
    taking the maximum over the *finite* channel scores at each step.

    Args:
        features: ``(B, T, C)`` float array of observed features.
        seq_lengths: ``(B,)`` integer array of valid prefix lengths.
        window_size: causal window size (default 30).

    Returns:
        ``(B, T)`` float32 array of alarm scores (``NaN`` where
        undefined).
    """
    features = np.asarray(features)
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
    if features.ndim != 3:
        raise ValueError(f"features must be (B, T, C), got shape {features.shape}")
    if seq_lengths.shape[0] != features.shape[0]:
        raise ValueError(
            f"seq_lengths length {seq_lengths.shape[0]} != B {features.shape[0]}"
        )
    B, T, C = features.shape
    W = max(1, min(int(window_size), T))
    scores = np.full((B, T), np.nan, dtype=np.float32)
    for b in range(B):
        L = min(int(seq_lengths[b]), T)
        if L <= 0:
            continue
        for c in range(C):
            seq = np.asarray(features[b, :, c], dtype=np.float64)
            for t in range(L):
                seg = seq[max(0, t - W + 1) : t + 1]
                if len(seg) < _MIN_POINTS:
                    continue
                s = _window_abs_skewness(seg)
                if not np.isfinite(s):
                    continue
                if not np.isfinite(scores[b, t]):
                    scores[b, t] = s
                else:
                    scores[b, t] = max(float(scores[b, t]), s)
    return scores


def _window_abs_skewness(seg: np.ndarray) -> float:
    """``abs(m3 / m2^1.5)`` of the linearly detrended window.

    The linear detrend is applied inside the window first (plan §3:
    within-window linear detrending, causal because the window is
    causal). The OLS residual has zero mean, so the central moments
    are taken around the (numerically zero) residual mean. Returns
    ``NaN`` for a degenerate (zero-variance) window.
    """
    seg = _linear_detrend(seg)
    seg = seg - seg.mean()
    m2 = float(np.mean(seg * seg))
    if m2 <= 1e-12:
        return float("nan")
    m3 = float(np.mean(seg * seg * seg))
    return abs(m3 / (m2 ** 1.5))


__all__ = ["raw_skew_indicator"]
