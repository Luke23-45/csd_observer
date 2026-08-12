"""Windowed variance (VAR) critical-slowing-down baseline.

Definition (benchmark plan §3, row VAR): the **population variance**
``mean((x - mean)²)`` of the within-window linearly detrended causal
window. This is the variance-increase early-warning signal of critical
slowing down.

References:

* Scheffer et al. (2009), *Nature* 461: 53–59 — increased variance of
  the fluctuations as a generic consequence of critical slowing down
  (Box 3: the variance diverges as the dominant eigenvalue approaches
  zero).
* Carpenter & Brock (2006), *Ecol. Lett.* 9: 308–315 — "Rising
  variance: a leading indicator of ecological transition".
* Dakos et al. (2012), *PLoS ONE* 7(7): e41010 — variance estimated
  in rolling windows on detrended data.
* ``earlywarnings`` (R) ``sd`` indicator: the *sample standard
  deviation* of the detrended window. Note: the R toolbox reports the
  standard deviation, while the benchmark plan specifies the variance
  ``mean((x - mean)²)``; ewstools (Bury et al. 2021) computes the
  *sample* variance (``pandas`` rolling ``var``, ddof=1) of the
  detrended residuals. All three quantities are strictly monotone
  transforms of one another on non-negative values, hence
  rank-equivalent under the benchmark's fixed-FPR calibration (plan
  §4); we implement the plan's formula verbatim.

Direction convention: the raw indicator value is the alarm score
(higher score => higher alarm). On fold and logistic systems variance
rises towards the transition. Known caveat (plan §3): on the
bury-hard fold the *null* variance can exceed the signal's, so the
EW-AUC may invert; the benchmark reports the raw direction without
adaptive transforms (Dakos et al. 2012 likewise document conditions
under which variance decreases).

The indicator is deterministic and has no parameters.
"""

from __future__ import annotations

import numpy as np

from csd_observer.models.common.detrend import _linear_detrend

_MIN_POINTS = 4


def raw_var_indicator(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 30,
) -> np.ndarray:
    """Population variance of causal linearly detrended windows.

    For each trajectory ``b`` and step ``t`` the score is the
    population variance ``mean((x - mean)²)`` of the causal window
    ``x[max(0, t-W+1) : t+1]`` after within-window linear detrending
    (``_linear_detrend``, ``detrend.py``). The window ends at ``t`` and
    has effective size ``min(W, t+1)``, i.e. every step gets a score
    once enough data has accumulated (plan §4 — same causal convention
    as ``running_mean_center``).

    Scores are undefined (``NaN``) when fewer than 4 points are
    available in the causal window (the same guard as the other
    window-30 baselines) or when ``t >= seq_lengths[b]`` (beyond the
    valid prefix). A degenerate (constant) window is *not* NaN: the
    variance of a constant window is defined and equals 0.

    For multi-channel inputs the channel-wise scores are combined by
    taking the maximum over the *finite* channel scores at each step.

    Note: this replaces the legacy ``csd_observer.utils.metrics``
    ``raw_var_indicator`` (which does not detrend and forward-fills
    the first window) in the modular baseline suite.

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
                v = _window_variance(seg)
                if not np.isfinite(v):
                    continue
                if not np.isfinite(scores[b, t]):
                    scores[b, t] = v
                else:
                    scores[b, t] = max(float(scores[b, t]), v)
    return scores


def _window_variance(seg: np.ndarray) -> float:
    """Population variance of the linearly detrended window.

    The OLS residual of a first-order polynomial detrend has zero mean
    by construction, so the plan's formula
    ``mean((x - mean(x))^2)`` reduces to ``mean(x^2)`` after
    detrending. We compute the squared-mean directly without the
    redundant centering pass — numerically identical on finite inputs
    and one subtraction cheaper per window step.
    """
    seg = _linear_detrend(seg)
    return float(np.mean(seg * seg))


__all__ = ["raw_var_indicator"]
