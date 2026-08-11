"""Return-rate / recovery-time (RETRATE) critical-slowing-down baseline.

Definition (benchmark plan §3, row RETRATE): the score is the
**recovery time** ``-log(return_rate)`` of the within-window linearly
detrended causal window, where ``return_rate = 1/phi`` and ``phi`` is
the AR(1) coefficient estimated by ordinary least squares (the
``earlywarnings`` ``ar1`` estimator). The plan's direction convention
(§3 table) is that the raw indicator value is the alarm score, so for
RETRATE — whose *return rate* ``1/phi`` falls as slowing down
increases — the score is the monotone transform ``-log(return_rate)
= log(phi)`` (recovery time), which rises with slowing down; monotone
transforms do not change ranked metrics (plan §3).

References:

* Ives (1995), *Ecol. Monogr.* 65: 217–233 — resilience as recovery
  rate of a stochastic system; Ives et al. (2003), *Ecol. Monogr.*
  73: 301–330 — recovery time lengthens as the dominant eigenvalue
  approaches 1.
* Dakos et al. (2012), *PLoS ONE* 7(7): e41010 — return rate in
  rolling windows.
* ``earlywarnings`` (R) ``generic_ews``: ``ar1`` is fitted as
  ``ar.ols(x, aic = FALSE, order.max = 1, dmean = FALSE,
  intercept = FALSE)`` (verbatim from r-forge ``generic_ews.R``; the
  misspelled ``dmean`` argument is forwarded to and ignored by
  ``lm.fit``'s dots, so the effective call keeps ``ar.ols``'s default
  ``demean = TRUE``) and ``returnrate = 1/ar1`` in the code (the docs
  text says ``1 - ar(1)``; the code is authoritative, plan §3).
  ``ar.ols`` regresses ``x_{t+1}`` on ``x_t`` (no intercept) after
  demeaning, i.e. ``phi = sum(x[:-1]*x[1:]) / sum(x[:-1]^2)`` — the
  OLS estimator, which differs from the Yule-Walker/ACF estimator
  (``sum(x[:-1]*x[1:]) / sum(x^2)``) by ``O(1/n)``; we implement the
  OLS estimator verbatim.

``phi`` is estimated on the within-window linearly detrended window
(plan §3 global convention; ``earlywarnings`` computes indicators on
its detrended residuals). Because ``log(phi)`` is undefined for
``phi <= 0`` (oscillatory windows — e.g. the Hopf model — produce
negative AR(1) estimates), ``phi`` is floored at ``1e-4`` before the
log (a lower-only clip; the superseded plan's "(0,1) clip" was
written for the ``-1/log(ar1)`` recovery-time formula, which also
breaks above 1 — our ``log(phi)`` transform needs no upper clip).
The floor only saturates the lower tail of the score distribution and
does not affect the upper-tail alarm threshold (fixed-FPR
calibration, plan §4).

Direction convention: raw score = recovery time, higher score =>
higher alarm (rises as the system slows; monotone-inverse of AC1).
Like AC1/SRATIO/DFA it is monotone in persistence and inverts on the
Hopf model (documented caveat).

The indicator is deterministic and has no parameters.
"""

from __future__ import annotations

import numpy as np

from csd_observer.models.common.detrend import _linear_detrend

_MIN_POINTS = 4
_PHI_FLOOR = 1e-4


def raw_retrate_indicator(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 30,
) -> np.ndarray:
    """Recovery time ``log(phi)`` of causal linearly detrended windows.

    For each trajectory ``b`` and step ``t`` the score is
    ``log(phi)`` of the causal window ``x[max(0, t-W+1) : t+1]`` after
    within-window linear detrending (``_linear_detrend``,
    ``detrend.py``), where ``phi`` is the OLS AR(1) coefficient of the
    demeaned residual (``ar.ols`` estimator: numerator and denominator
    both over the ``n-1`` lagged pairs) floored at ``1e-4``. The
    window ends at ``t`` and has effective size ``min(W, t+1)``, i.e.
    every step gets a score once enough data has accumulated (plan §4
    — same causal convention as ``running_mean_center``).

    Scores are undefined (``NaN``) when fewer than 4 points are
    available in the causal window (the same guard as the other
    window-30 baselines), when the detrended window has zero variance
    (``phi`` is 0/0 — constant or perfectly linear windows), or when
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
                s = _window_retrate(seg)
                if not np.isfinite(s):
                    continue
                if not np.isfinite(scores[b, t]):
                    scores[b, t] = s
                else:
                    scores[b, t] = max(float(scores[b, t]), s)
    return scores


def _window_retrate(seg: np.ndarray) -> float:
    """``log(max(phi, 1e-4))`` of the linearly detrended window.

    The linear detrend is applied inside the window first (plan §3).
    The OLS AR(1) coefficient on the demeaned residual is
    ``phi = sum(seg[:-1]*seg[1:]) / sum(seg[:-1]^2)`` (R ``ar.ols``,
    intercept-free regression of the lagged pairs; note the
    denominator is over the ``n-1`` lagged values, not the full
    window). Returns ``NaN`` for a degenerate (zero-variance) window.
    """
    seg = _linear_detrend(seg)
    seg = seg - seg.mean()
    denom = float(np.dot(seg[:-1], seg[:-1]))
    if denom <= 1e-12:
        return float("nan")
    phi = float(np.dot(seg[:-1], seg[1:])) / denom
    if not np.isfinite(phi):
        return float("nan")
    return float(np.log(max(phi, _PHI_FLOOR)))


__all__ = ["raw_retrate_indicator"]
