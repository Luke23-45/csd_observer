"""Alarm evaluation for the benchmark suite.

Single canonical implementation of the evaluation governance defined in
``docs/plan/benchmark_revision_plan.md`` §4 (the "unchanged pipeline"):

* **Fixed-FPR calibration:** ``threshold = percentile(all finite
  val-null score steps, 100 * (1 - fpr_target))``. Only *finite* scores
  enter the percentile so indicators with leading ``NaN`` windows (DFA
  requires 100 points) do not poison the calibration.
* **Detection time:** first step ``t < tau`` with ``score >= threshold``;
  ``NaN >= threshold`` evaluates ``False`` in numpy, so undefined early
  windows are treated as "no alarm" — conservative and identical across
  methods (plan §4).
* **Early-warning AUC:** per-trajectory max over ``[tau - 50, tau - 5)``
  for signal trajectories, ``[T - 50, T - 5)`` for null trajectories,
  ranked with ``roc_auc_score``; ``NaN`` scores are filled with the
  neutral mid-point ``0.5`` before ranking.
* **False-positive rate:** fraction of valid (in-prefix) null test
  steps with ``score >= threshold``; ``NaN`` steps remain in the
  denominator but never count as alarms (``NaN >= threshold`` is
  ``False``), so DFA-style leading ``NaN`` windows cannot inflate the
  FPR.

These functions are numerically identical to the legacy
``csd_observer.utils.metrics`` implementations on NaN-free inputs
(guarded by ``tests/test_benchmark.py`` parity tests); the difference is
exclusively the NaN semantics required by plan §4.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score


def _sanitize_scores(scores: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(scores, nan=fill, posinf=1.0, neginf=0.0)


def calibrate_threshold(
    scores_val_null: np.ndarray,
    seq_lengths: np.ndarray,
    fpr_target: float,
) -> float:
    """Fixed-FPR alarm threshold from the validation null scores.

    Concatenates the valid prefixes of all validation-null trajectories
    and takes the ``100 * (1 - fpr_target)`` percentile over the
    *finite* steps only.

    Args:
        scores_val_null: ``(B, T)`` alarm scores on validation nulls.
        seq_lengths: ``(B,)`` valid prefix lengths.
        fpr_target: target false-positive rate (e.g. 0.05).

    Returns:
        The threshold, or ``NaN`` when no finite score is available.
    """
    steps = np.concatenate(
        [
            scores_val_null[i, : int(length)]
            for i, length in enumerate(seq_lengths)
            if int(length) > 0
        ]
    )
    finite = steps[np.isfinite(steps)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 100.0 * (1.0 - fpr_target)))


def compute_per_traj_dts(
    scores: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> List[float]:
    """Per-trajectory detection lead ``tau - t_first_alarm``.

    ``NaN`` scores never trigger (``NaN >= threshold`` is ``False``), so
    trajectories without an alarm — or with an undefined threshold —
    get ``NaN`` entries (plan §4).
    """
    times: List[float] = []
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
    """AUC of early-window maxima: signal ``[tau - 50, tau - 5)`` vs
    null ``[T - 50, T - 5)``. ``NaN`` scores fill with 0.5 (neutral
    mid-point under ranked evaluation, plan §4)."""
    scores_signal = _sanitize_scores(scores_signal, fill=0.5)
    scores_null = _sanitize_scores(scores_null, fill=0.5)
    scores: List[float] = []
    labels: List[int] = []
    for i in range(len(scores_signal)):
        tau = float(bif_times_signal[i])
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0:
            t_start = max(0, int(tau - early_start_delta))
            t_end = max(0, int(tau - early_end_delta))
            window = scores_signal[i, t_start:t_end]
            if len(window) > 0:
                scores.append(float(np.max(window)))
                labels.append(1)
    for i in range(len(scores_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            t_start = max(0, int(T - early_start_delta))
            t_end = max(0, int(T - early_end_delta))
            window = scores_null[i, t_start:t_end]
            if len(window) > 0:
                scores.append(float(np.max(window)))
                labels.append(0)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_false_positive_rate(
    scores: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of valid steps with ``score >= threshold`` on nulls.

    ``NaN >= threshold`` is ``False``, so undefined steps never count as
    false alarms. Returns ``NaN`` for a non-finite threshold (nothing
    meaningful can be reported).
    """
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
) -> Dict[str, float]:
    return {"fpr": compute_false_positive_rate(scores, seq_lengths, threshold)}


__all__ = [
    "calibrate_threshold",
    "compute_detection_time",
    "compute_early_warning_auc",
    "compute_false_positive_rate",
    "compute_null_metrics",
    "compute_per_traj_dts",
]
