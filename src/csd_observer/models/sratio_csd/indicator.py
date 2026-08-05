"""Spectral-density-ratio (SRATIO) critical-slowing-down baseline.

Definition (benchmark plan §3, row SRATIO): fit a first-order
autoregressive model (Yule-Walker) to the within-window linearly
detrended causal window and take the ratio of its parametric power
spectrum at the **lowest non-zero frequency bin** over the spectrum at
the **Nyquist bin** (spectral reddening). The bands are window-relative:
with a window of ``n`` points the spectrum is evaluated on R's
``spec.ar`` grid ``freq = seq(0, 0.5, length.out = n)`` (cycles per
sample), so the low bin is at ``f = 0.5/(n-1)`` and the high bin at
``f = 0.5`` (Nyquist).

References:

* Dakos et al. (2012), *PLoS ONE* 7(7): e41010 — density ratio of the
  power spectrum at low over high frequencies within rolling windows.
* ``earlywarnings`` (R) ``generic_ews`` ``densratio``:
  ``spec.ar(window, n.freq = mw, order = 1)``; ``densratio =
  spec[low]/spec[high]`` with ``low = 2`` (lowest non-zero bin) and
  ``high = mw`` (last bin = Nyquist) in the current CRAN versions; the
  very early r-forge versions used ``low = 6`` (documented there only).
  We follow the current canonical convention (``low = 2``).
* R ``stats::spec.ar``: the parametric AR(1) spectral density is
  ``S(f) = var.pred / ((1 - cos(2*pi*f)*phi)^2 + (sin(2*pi*f)*phi)^2)
  = var.pred / (1 + phi^2 - 2*phi*cos(2*pi*f))``; the ``var.pred``
  scale cancels in the ratio, so only the Yule-Walker coefficient
  ``phi`` and the frequency grid matter.
* Yule-Walker AR(1) coefficient (R ``ar.yw``): the biased lag-1
  autocorrelation of the demeaned window,
  ``phi = sum((x-xbar)[:-1]*(x-xbar)[1:]) / sum((x-xbar)^2)``.
* Scheffer et al. (2009), *Nature* 461: 53–59 (Box 3) — spectral
  reddening (increasing low-frequency power) as a slowing-down signal.

Detrending follows the plan's global convention (within-window linear
detrending via ``_linear_detrend``); the toolbox instead applies
Gaussian-kernel smoothing before rolling windows. For a window of
``n`` points the grid is anchored to the *effective* window size
``n = min(W, t+1)`` (as ``earlywarnings`` uses ``n.freq = winsize``),
so partial windows are scored on their own grid and the score at step
``t`` depends only on the causal window (plan §4).

Direction convention: the raw indicator value is the alarm score
(higher score => higher alarm). It is a strictly increasing function
of ``phi`` on (-1, 1), so it behaves like AC1-CSD: it rises before
fold bifurcations but inverts on the Hopf model (documented caveat).

The indicator is deterministic and has no parameters.
"""

from __future__ import annotations

import numpy as np

from csd_observer.utils.metrics import _linear_detrend

_MIN_POINTS = 4


def raw_sratio_indicator(
    features: np.ndarray,
    seq_lengths: np.ndarray,
    window_size: int = 30,
) -> np.ndarray:
    """Low-over-Nyquist spectral-density ratio of causal detrended windows.

    For each trajectory ``b`` and step ``t`` the score is
    ``S(f_1) / S(f_{n-1})`` of the parametric AR(1) spectrum of the
    causal window ``x[max(0, t-W+1) : t+1]`` after within-window linear
    detrending (``_linear_detrend``, ``metrics.py``), where ``n = min(W,
    t+1)`` is the effective window size, ``f_k = k*0.5/(n-1)`` (R
    ``spec.ar`` grid), ``S(f) = 1/(1 + phi^2 - 2*phi*cos(2*pi*f))`` and
    ``phi`` is the Yule-Walker AR(1) coefficient of the demeaned
    detrended window. The window ends at ``t`` (plan §4 — same causal
    convention as ``running_mean_center``).

    Scores are undefined (``NaN``) when fewer than 4 points are
    available in the causal window (the same guard as the other
    window-30 baselines), when the detrended window has zero variance
    (``phi`` is 0/0 — constant or perfectly linear windows), when the
    spectral density is degenerate at the used bins (e.g. ``phi = -1``
    gives zero spectrum at Nyquist), or when ``t >= seq_lengths[b]``
    (beyond the valid prefix).

    For multi-channel inputs the channel-wise scores are combined by
    taking the maximum over the *finite* channel scores at each step.

    Args:
        features: ``(B, T, C)`` float array of observed features.
        seq_lengths: ``(B,)`` integer array of valid prefix lengths.
        window_size: causal window size (default 30); also anchors the
            frequency grid via the effective window size.

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
                s = _window_sratio(seg)
                if not np.isfinite(s):
                    continue
                if not np.isfinite(scores[b, t]):
                    scores[b, t] = s
                else:
                    scores[b, t] = max(float(scores[b, t]), s)
    return scores


def _window_sratio(seg: np.ndarray) -> float:
    """Low-over-Nyquist AR(1) spectral-density ratio of a detrended window.

    The linear detrend is applied inside the window first (plan §3).
    The OLS residual has zero mean, so the Yule-Walker coefficient is
    the biased lag-1 autocorrelation of the demeaned residual (R
    ``ar.yw`` / ``acf(type = "covariance")``). The AR(1) spectral
    density shape ``1/(1 + phi^2 - 2*phi*cos(2*pi*f))`` is evaluated on
    the ``n``-point grid ``seq(0, 0.5, length.out = n)`` (R
    ``spec.ar``), where ``n = len(seg)``; the ``var.pred`` scale
    cancels in the ratio. Returns ``NaN`` for a degenerate window or a
    degenerate spectrum (grid with fewer than 3 bins, zero variance, or
    a zero/non-finite spectral value at the used bins).
    """
    seg = _linear_detrend(seg)
    seg = seg - seg.mean()
    denom = float(np.dot(seg, seg))
    if denom <= 1e-12:
        return float("nan")
    phi = float(np.dot(seg[:-1], seg[1:])) / denom
    if not np.isfinite(phi):
        return float("nan")
    n_freq = len(seg)
    if n_freq < 3:
        return float("nan")
    freqs = np.linspace(0.0, 0.5, n_freq)
    shape = 1.0 / (1.0 + phi * phi - 2.0 * phi * np.cos(2.0 * np.pi * freqs))
    s_low = float(shape[1])
    s_high = float(shape[-1])
    if not np.isfinite(s_low) or not np.isfinite(s_high) or s_high <= 1e-12:
        return float("nan")
    return s_low / s_high


__all__ = ["raw_sratio_indicator"]
