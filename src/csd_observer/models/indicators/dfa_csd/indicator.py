"""Detrended-fluctuation-analysis (DFA) critical-slowing-down baseline.

Definition (benchmark plan §3, row DFA): the DFA scaling exponent
``alpha`` — the slope of ``log F(s)`` vs ``log s`` — of the within-window
linearly detrended causal window, computed with the standard
(Kantelhardt) DFA-1 algorithm on box sizes ``s = 10, 20, 40, 80`` scaled
by ``window/100`` (the plan's 4-point discretization of the canonical
10–100-unit short-term regime).

References:

* Peng et al. (1994), *Phys. Rev. E* 49: 1685–1689 — detrended
  fluctuation analysis; ``F(s) ~ s^alpha``.
* Kantelhardt et al. (2001), *Physica A* 316: 87–114 — standard DFA
  algorithm: integrate the (mean-centered) series to a profile, split
  into non-overlapping segments of size ``s`` (and repeat from the end,
  covering the tail), detrend each segment with a first-order
  polynomial, ``F(s) = sqrt(mean of segment residual variances)``.
* Livina & Lenton (2007), *Geophys. Res. Lett.* 34: L03712 — DFA
  exponent as an incipient-bifurcation indicator; ``alpha -> 1.5``
  (random-walk state) as the decay rate of fluctuations vanishes; DFA
  is evaluated in the short-term regime of 10–100 time units.
* Dakos et al. (2012), *PLoS ONE* 7(7): e41010 — DFA indicator in
  rolling windows of half the record after removing a simple linear
  trend; the exponent is usually estimated between 10 and 100 time
  units and rescaled to reach 1 at criticality; DFA "requires >100
  points for robust estimation".
* Scheffer et al. (2009), *Nature* 461: 53–59 (Box 3) — spectral
  reddening and long-range correlation as slowing-down signals.

Data-demand convention (plan §5): DFA is only scored on *full* causal
windows of at least 100 points — partial windows get ``NaN`` (no score
for the first ``W-1`` steps; trajectories shorter than 100 points get
no score at all). ``earlywarnings`` has no DFA function; the window
default is 100 per the plan (Dakos' "requires >100 points"), not 30.

Direction convention: the raw indicator value is the alarm score
(higher score => higher alarm); ``alpha`` rises toward 1.5 (rescaled
toward 1) as the system approaches criticality (Livina & Lenton 2007).
Like AC1/SRATIO it is monotone in persistence and inverts on the Hopf
model (documented caveat). We report the raw exponent, not the
rescaled indicator (plan §3 row: "detrended-fluctuation-analysis
exponent alpha").

The indicator is deterministic and has no parameters.
"""

from __future__ import annotations

import numpy as np

from csd_observer.models.common.detrend import _linear_detrend

_BASE_BOX_SIZES = (10, 20, 40, 80)


def raw_dfa_indicator(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 100,
) -> np.ndarray:
    """DFA scaling exponent of causal linearly detrended windows.

    For each trajectory ``b`` and step ``t`` the score is the DFA-1
    exponent ``alpha`` of the causal window ``x[max(0, t-W+1) : t+1]``
    after within-window linear detrending (``_linear_detrend``,
    ``detrend.py``): the mean-centered window is integrated to a
    profile, the profile is split into non-overlapping segments of each
    box size ``s`` (twice — from the start and from the end, per
    Kantelhardt), each segment is detrended with a first-order
    polynomial, and ``alpha`` is the OLS slope of ``log F(s)`` vs
    ``log s`` over ``s = 10, 20, 40, 80`` scaled by ``W/100``.

    Unlike the window-30 baselines, DFA scores only *full* windows of
    at least 100 points (plan §5): steps with fewer than ``W`` points
    in the causal window, or any step when ``W < 100``, are ``NaN``.
    The first score appears at ``t = W - 1``. Scores are also ``NaN``
    when the detrended window has zero variance (constant or perfectly
    linear windows — ``F(s) = 0`` for all box sizes, so the log-log
    slope is undefined) or when ``t >= seq_lengths[b]`` (beyond the
    valid prefix).

    For multi-channel inputs the channel-wise scores are combined by
    taking the maximum over the *finite* channel scores at each step.

    Args:
        features: ``(B, T, C)`` float array of observed features.
        seq_lengths: ``(B,)`` integer array of valid prefix lengths.
        window_size: causal window size (default 100; plan §3: DFA
            window 100). Box sizes scale with the effective window
            size (``base * W/100``).

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
    if W < 100:
        return np.full((B, T), np.nan, dtype=np.float32)
    box_sizes = tuple(int(round(base * W / 100)) for base in _BASE_BOX_SIZES)
    scores = np.full((B, T), np.nan, dtype=np.float32)
    for b in range(B):
        L = min(int(seq_lengths[b]), T)
        if L <= 0:
            continue
        for c in range(C):
            seq = np.asarray(features[b, :, c], dtype=np.float64)
            for t in range(W - 1, L):
                seg = seq[t - W + 1 : t + 1]
                s = _window_dfa_alpha(seg, box_sizes)
                if not np.isfinite(s):
                    continue
                if not np.isfinite(scores[b, t]):
                    scores[b, t] = s
                else:
                    scores[b, t] = max(float(scores[b, t]), s)
    return scores


def _window_dfa_alpha(seg: np.ndarray, box_sizes: tuple[int, ...]) -> float:
    """DFA-1 scaling exponent of a linearly detrended window.

    The linear detrend is applied inside the window first (plan §3
    global convention; Dakos 2012 detrends before DFA). Returns
    ``NaN`` when the detrended window has zero variance or any box
    size yields a non-positive fluctuation ``F(s)``.
    """
    seg = _linear_detrend(seg)
    seg = seg - seg.mean()
    if float(np.mean(seg * seg)) <= 1e-12:
        return float("nan")
    n = len(seg)
    if n < box_sizes[-1]:
        return float("nan")
    profile = np.cumsum(seg)
    log_s = np.empty(len(box_sizes), dtype=np.float64)
    log_f = np.empty(len(box_sizes), dtype=np.float64)
    for i, s in enumerate(box_sizes):
        f = _fluctuation(profile, s)
        if not np.isfinite(f) or f <= 0.0:
            return float("nan")
        log_s[i] = np.log(float(s))
        log_f[i] = np.log(f)
    return float(np.polyfit(log_s, log_f, 1)[0])


def _fluctuation(profile: np.ndarray, s: int) -> float:
    """``F(s)``: sqrt of the mean squared detrended profile residual.

    Splits the profile into ``floor(N/s)`` non-overlapping segments
    starting at the beginning and the same number starting at the end
    (Kantelhardt et al. 2001), detrends each segment with a first-order
    polynomial fit, and returns ``sqrt(mean(residual^2))`` over all
    segments.
    """
    n = len(profile)
    n_seg = n // s
    x = np.arange(s, dtype=np.float64)
    X = np.vstack([x, np.ones(s)]).T
    idx = np.empty((2 * n_seg, s), dtype=np.intp)
    for j in range(n_seg):
        idx[j] = np.arange(j * s, (j + 1) * s)
        idx[n_seg + j] = np.arange(n - (j + 1) * s, n - j * s)
    Y = profile[idx]
    coeffs = np.linalg.lstsq(X, Y.T, rcond=None)[0]
    resid = Y - (X @ coeffs).T
    return float(np.sqrt(np.mean(resid * resid)))


__all__ = ["raw_dfa_indicator"]
