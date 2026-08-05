"""Lag-1 autocorrelation (AC1) critical-slowing-down baseline.

Definition (benchmark plan §3, row AC1): the **Pearson correlation
coefficient between the lagged halves** ``x[t-1]`` and ``x[t]`` of the
within-window linearly detrended causal window. This is the definition
used by Bury et al. 2021 (``ewstools`` ``compute_auto``, which is
``pd.Series.autocorr``, i.e. the Pearson correlation of the two lagged
segments).

References:

* Dakos et al. (2012), *PLoS ONE* 7(7): e41010 — lag-1 autocorrelation
  on detrended windows as a critical-slowing-down indicator.
* Bury et al. (2021), *PNAS* 118(39): e2106140118 — lag-1
  autocorrelation of detrended residuals; the open-source
  implementation (ewstools) is the Pearson correlation of the two
  lagged segments.
* ``earlywarnings`` (R) ``acf1``: the lag-1 sample autocorrelation
  estimator ``acf(x, lag.max = 1, type = "correlation")$acf[2]``.
  Note: the R toolbox uses the ACF estimator
  ``sum_t (x_t - xbar)(x_{t+1} - xbar) / sum_t (x_t - xbar)^2``
  (single full-window mean in both numerator and denominator), which
  differs from the Pearson definition by ``O(1/n)`` in normalisation.
  Per the benchmark plan we follow the Pearson (Bury/ewstools)
  definition; the two are numerically close and rank-equivalent in
  practice.

Direction convention: the raw indicator value is the alarm score
(higher score => higher alarm). On fold and logistic systems AC1 rises
towards the transition; on Hopf (oscillatory component) it typically
*falls* — a documented inversion (Bury et al. 2021). The benchmark
reports the raw direction (plan §3, no adaptive sign flip).

The indicator is deterministic and has no parameters.
"""

from __future__ import annotations

import numpy as np

from csd_observer.utils.metrics import _linear_detrend

_MIN_POINTS = 4


def raw_ac1_indicator(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 30,
) -> np.ndarray:
    """Lag-1 Pearson autocorrelation on causal linearly detrended windows.

    For each trajectory ``b`` and step ``t`` the score is the Pearson
    correlation between ``x[t-1]`` and ``x[t]`` of the causal window
    ``x[max(0, t-W+1) : t+1]`` after within-window linear detrending
    (``_linear_detrend``, ``metrics.py``). The window ends at ``t`` and
    has effective size ``min(W, t+1)``, i.e. every step gets a score
    once enough data has accumulated (plan §4 — same causal convention
    as ``running_mean_center``).

    Scores are undefined (``NaN``) when:

    * fewer than 4 points are available in the causal window
      (the same guard as ``raw_lag2_indicator``);
    * the detrended window has zero variance (the Pearson
      correlation is degenerate — constant windows; both R's ``acf``
      and pandas ``autocorr`` return ``NA``/``NaN`` there);
    * ``t >= seq_lengths[b]`` (beyond the valid prefix).

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
                rho = _lag1_pearson(seg)
                if not np.isfinite(rho):
                    continue
                if not np.isfinite(scores[b, t]):
                    scores[b, t] = rho
                else:
                    scores[b, t] = max(float(scores[b, t]), rho)
    return scores


def _lag1_pearson(seg: np.ndarray) -> float:
    """Pearson correlation between ``seg[:-1]`` and ``seg[1:]``.

    The linear detrend is applied inside the window first (plan §3:
    within-window linear detrending, causal because the window is
    causal). Returns ``NaN`` for a degenerate (zero-variance)
    window.
    """
    seg = _linear_detrend(seg)
    a = seg[:-1] - seg[:-1].mean()
    b = seg[1:] - seg[1:].mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


__all__ = ["raw_ac1_indicator"]
