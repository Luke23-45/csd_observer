"""Metric primitives used by every method evaluation.

Migrated from ``csd_observer.utils.evaluation`` (and ``utils.metrics``) so
the canonical implementation lives under ``evaluation/`` instead of the
generic ``utils/`` package. Behaviour preserved bit-for-bit on NaN-free
inputs; only the package location changes.

Conventions:

* ``scores`` is ``(B, T)`` float32
* ``seq_lengths`` is ``(B,)`` int
* ``bifurcation_times`` is ``(B,)`` (per-trajectory transition index)
* ``is_positive`` is ``(B,)`` bool (True = signal)
* ``NaN`` scores never trigger an alarm (``NaN >= threshold`` is False)
* step-wise FPR denominator counts every in-prefix step, even NaN ones
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

W_LABEL = 60  # legacy constant (kept for parity with classic EW-AUC)


def _sanitize_scores(scores: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(scores, nan=fill, posinf=1.0, neginf=0.0)


def compute_per_traj_dts(
    scores: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> list[float]:
    """Per-trajectory detection lead ``tau - t_first_alarm`` (classic).

    ``NaN`` scores never trigger, so trajectories without an alarm — or
    with an undefined threshold — get ``NaN`` entries.
    """
    times: list[float] = []
    for i in range(len(scores)):
        if not is_positive[i] or not np.isfinite(threshold):
            times.append(float("nan"))
            continue
        tau = float(bifurcation_times[i])
        if tau <= 0:
            times.append(float("nan"))
            continue
        pre = scores[i, : int(tau)]
        alerts = np.where(pre >= threshold)[0]
        if len(alerts) > 0:
            times.append(tau - alerts[0])
        else:
            times.append(float("nan"))
    return times


def compute_detection_time(
    scores: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> float:
    """Mean detection lead over positive trajectories with an alarm."""
    times = compute_per_traj_dts(
        scores, bifurcation_times, is_positive, seq_lengths, threshold
    )
    finite = [t for t in times if np.isfinite(t)]
    return float(np.mean(finite)) if finite else float("nan")


def compute_early_warning_auc(
    scores_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    scores_null: np.ndarray,
    seq_lens_null: np.ndarray,
    *,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> float:
    """Classic EW-AUC: per-trajectory max over the early window vs null."""
    scores_signal = _sanitize_scores(scores_signal, fill=0.5)
    scores_null = _sanitize_scores(scores_null, fill=0.5)
    scores: list[float] = []
    labels: list[int] = []
    for i in range(len(scores_signal)):
        tau = float(bif_times_signal[i])
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0 and T > 0:
            t_start = max(0, int(tau - early_start_delta))
            t_end = min(max(0, int(tau - early_end_delta)), int(tau), T)
            if t_end > t_start:
                window = scores_signal[i, t_start:t_end]
                scores.append(float(np.max(window)))
                labels.append(1)
    for i in range(len(scores_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            t_start = max(0, int(T - early_start_delta))
            t_end = min(max(0, int(T - early_end_delta)), T)
            if t_end > t_start:
                window = scores_null[i, t_start:t_end]
                scores.append(float(np.max(window)))
                labels.append(0)
    if len(set(labels)) < 2:
        return float("nan")
    try:
        if len(set(scores)) < 2:
            return 0.5
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def compute_false_positive_rate(
    scores: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of valid steps with ``score >= threshold`` on nulls."""
    if not np.isfinite(threshold):
        return float("nan")
    total_steps = 0
    alert_steps = 0
    for i in range(len(scores)):
        L = int(seq_lengths[i])
        total_steps += L
        alert_steps += int((scores[i, :L] >= threshold).sum())
    return alert_steps / max(total_steps, 1)


def compute_null_metrics(
    scores: np.ndarray,
    threshold: float,
    seq_lengths: np.ndarray,
) -> dict[str, float]:
    return {"fpr": compute_false_positive_rate(scores, seq_lengths, threshold)}


__all__ = [
    "compute_detection_time",
    "compute_early_warning_auc",
    "compute_false_positive_rate",
    "compute_null_metrics",
    "compute_per_traj_dts",
]
