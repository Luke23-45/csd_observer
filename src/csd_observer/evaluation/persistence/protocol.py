"""Persistence-aware evaluation protocol (§8 of the implementation plan).

The research contribution: a causal persistence filter that suppresses
transient, noise-driven threshold crossings before counting an alarm.

Formal definitions:

* alarm indicator      ``a_t = 1[s_t >= theta]``
* persistence statistic ``r_t = run length of consecutive alarms ending at t``
  (``r_t = r_{t-1} + 1 if a_t else 0``, ``r_0 = 0``) — **causal**, uses
  only ``s_{<=t}``, so persistent detection time is a valid online
  early-warning quantity.
* persistent alarm      ``r_t >= k_persist``
* P-DT (persistence-aware detection time) = first ``t`` with a persistent
  alarm, measured relative to ``bifurcation_time``; trajectories without
  a persistent alarm before the end are **censored** (reported separately
  as detection_rate).
* P-EW-AUC = AUC of the binary persistent-alarm stream over (positive
  early window ∪ null window) pooled steps. At ``k_persist=1`` this equals
  the classic step-alarm AUC (the legacy EW-AUC evaluated on the binarized
  alarm stream), so the ablation identity holds by construction.
* null anchor: for i.i.d. null steps with per-step alarm probability p,
  positive association (FKG) gives
  ``P(no persistent alarm in a W-step window) >= (1-p^k)^(W-k+1)``,
  hence an analytic upper bound on the per-window persistent-FPR.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def run_lengths(alarms: np.ndarray) -> np.ndarray:
    """Causal run-length statistic of a boolean ``(..., T)`` stream.

    ``out[..., t] = number of consecutive ``True`` steps ending at t``.
    """
    alarms = np.asarray(alarms, dtype=bool)
    out = np.zeros(alarms.shape, dtype=np.int64)
    # rolling accumulation along the time axis; works on any leading dims
    run = np.zeros(alarms.shape[:-1], dtype=np.int64)
    for t in range(alarms.shape[-1]):
        run = (run + 1) * alarms[..., t]
        out[..., t] = run
    return out


def persistent_alarm_stream(
    scores: np.ndarray,
    threshold: float,
    k_persist: int,
    *,
    seq_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Return a ``(B, T)`` boolean persistent-alarm stream.

    ``out[b, t]`` is True iff the causal run length of ``score >= threshold``
    ending at ``t`` is at least ``k_persist``. ``NaN >= threshold`` is False,
    so undefined early windows never start a run. Steps beyond ``seq_lengths``
    are forced to False (padding must never alarm).
    """
    if k_persist < 1:
        raise ValueError(f"k_persist must be >= 1, got {k_persist}")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B,T), got {scores.shape}")
    if seq_lengths is not None:
        lengths = np.asarray(seq_lengths, dtype=np.int64)
        if lengths.shape != (scores.shape[0],):
            raise ValueError(f"seq_lengths must have shape {(scores.shape[0],)}, got {lengths.shape}")
        if np.any(lengths < 0) or np.any(lengths > scores.shape[1]):
            raise ValueError("seq_lengths must be within [0, T]")
    alarms = scores >= threshold
    r = run_lengths(alarms)
    out = r >= k_persist
    if seq_lengths is not None:
        lengths = np.asarray(seq_lengths, dtype=np.int64)
        mask = np.zeros(scores.shape, dtype=bool)
        for i, L in enumerate(lengths):
            mask[i, : int(L)] = True
        out = out & mask
    return out


def compute_persistent_dts(
    scores: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
    k_persist: int,
) -> tuple[list[float], list[float]]:
    """Return (per-traj P-DT lead, per-traj censored flag).

    ``P-DT = tau - t_first_persistent_alarm`` restricted to the pre-transition
    prefix. Trajectories without a persistent alarm before ``tau`` are
    censored (flag True, value NaN). Censored and undetected are *not*
    averaged into the DT; detection_rate is reported separately.
    """
    stream = persistent_alarm_stream(
        scores, threshold, k_persist, seq_lengths=seq_lengths
    )
    leads: list[float] = []
    censored: list[float] = []
    for i in range(len(scores)):
        if not is_positive[i]:
            leads.append(float("nan"))
            censored.append(float("nan"))
            continue
        tau = float(bifurcation_times[i])
        if not np.isfinite(threshold) or tau <= 0:
            leads.append(float("nan"))
            censored.append(float("nan"))
            continue
        pre = stream[i, : int(tau)]
        hits = np.where(pre)[0]
        if len(hits) > 0:
            leads.append(tau - hits[0])
            censored.append(0.0)
        else:
            leads.append(float("nan"))
            censored.append(1.0)
    return leads, censored


def compute_persistent_detection_metrics(
    scores: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
    k_persist: int,
) -> dict[str, float]:
    """Detection-rate + DT aggregates with explicit censoring (§8.2)."""
    leads, censored = compute_persistent_dts(
        scores, bifurcation_times, is_positive, seq_lengths, threshold, k_persist
    )
    n_pos = sum(1 for i in range(len(is_positive)) if bool(is_positive[i]))
    if n_pos == 0:
        return {
            "detection_rate": float("nan"),
            "detection_time_mean": float("nan"),
            "detection_time_median": float("nan"),
            "detection_time_std": float("nan"),
            "n_detected": 0,
            "n_censored": 0,
        }
    detected = [v for v in leads if np.isfinite(v)]
    n_detected = len(detected)
    n_censored = int(round(sum(1 for v in censored if v == 1.0)))
    rate = n_detected / n_pos
    if detected:
        arr = np.array(detected, dtype=float)
        dt_mean = float(arr.mean())
        dt_median = float(np.median(arr))
        dt_std = float(arr.std()) if len(arr) > 1 else float("nan")
    else:
        dt_mean = dt_median = dt_std = float("nan")
    return {
        "detection_rate": rate,
        "detection_time_mean": dt_mean,
        "detection_time_median": dt_median,
        "detection_time_std": dt_std,
        "n_detected": n_detected,
        "n_censored": n_censored,
    }


def compute_persistent_fpr(
    scores_null: np.ndarray,
    seq_lengths_null: np.ndarray,
    threshold: float,
    k_persist: int,
) -> float:
    """Persistent-FPR: fraction of null steps that lie in a persistent run."""
    if not np.isfinite(threshold):
        return float("nan")
    stream = persistent_alarm_stream(
        scores_null, threshold, k_persist, seq_lengths=seq_lengths_null
    )
    total = 0
    alarms = 0
    for i, L in enumerate(seq_lengths_null):
        L = int(L)
        total += L
        alarms += int(stream[i, :L].sum())
    return alarms / max(total, 1)


def compute_persistent_trajectory_fpr(
    scores_null: np.ndarray,
    seq_lengths_null: np.ndarray,
    threshold: float,
    k_persist: int,
) -> float:
    """Fraction of null trajectories with at least one persistent alarm.

    This is the quantity comparable to :func:`null_anchor_upper_bound`,
    whose unit is a window/trajectory rather than a time step.
    """
    if not np.isfinite(threshold) or len(seq_lengths_null) == 0:
        return float("nan")
    stream = persistent_alarm_stream(scores_null, threshold, k_persist, seq_lengths=seq_lengths_null)
    detected = sum(bool(stream[i, : int(length)].any()) for i, length in enumerate(seq_lengths_null))
    return detected / len(seq_lengths_null)


def compute_persistent_ew_auc(
    scores_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    scores_null: np.ndarray,
    seq_lens_null: np.ndarray,
    *,
    threshold: float,
    k_persist: int,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> float:
    """P-EW-AUC on the binary persistent-alarm stream (§8.3)."""
    stream_sig = persistent_alarm_stream(
        scores_signal, threshold, k_persist, seq_lengths=seq_lens_signal
    )
    stream_null = persistent_alarm_stream(
        scores_null, threshold, k_persist, seq_lengths=seq_lens_null
    )
    vals: list[float] = []
    labels: list[int] = []
    for i in range(len(stream_sig)):
        tau = float(bif_times_signal[i])
        if is_pos_signal[i] and tau > 0:
            t_start = max(0, int(tau - early_start_delta))
            t_end = max(0, int(tau - early_end_delta))
            window = stream_sig[i, t_start:t_end]
            if len(window) > 0:
                vals.extend(window.astype(np.float64).tolist())
                labels.extend([1] * len(window))
    for i in range(len(stream_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            t_start = max(0, int(T - early_start_delta))
            t_end = max(0, int(T - early_end_delta))
            window = stream_null[i, t_start:t_end]
            if len(window) > 0:
                vals.extend(window.astype(np.float64).tolist())
                labels.extend([0] * len(window))
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, vals))


def null_anchor_upper_bound(
    p: float, k_persist: int, window_steps: int
) -> float:
    """FKG upper bound on per-window persistent-alarm probability.

    ``P(alarm in W-window) <= 1 - (1 - p^k)^(W - k + 1)`` for i.i.d.
    null steps with per-step alarm prob ``p``.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1], got {p}")
    if k_persist < 1:
        raise ValueError(f"k_persist must be >= 1, got {k_persist}")
    if window_steps < k_persist:
        return 0.0
    return 1.0 - (1.0 - p**k_persist) ** (window_steps - k_persist + 1)


__all__ = [
    "run_lengths",
    "persistent_alarm_stream",
    "compute_persistent_dts",
    "compute_persistent_detection_metrics",
    "compute_persistent_fpr",
    "compute_persistent_trajectory_fpr",
    "compute_persistent_ew_auc",
    "null_anchor_upper_bound",
]
