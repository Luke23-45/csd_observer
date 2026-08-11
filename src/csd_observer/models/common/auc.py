"""Model-side validation selection metric.

The spectral-drift hyperparameter grid search (``models/spectral_drift/
grid_search.py``) selects ``(sigma_u, Q_drift)`` by the early-warning AUC
on the *validation* split. Per the one-way import rule of the plan
(``models`` must not import ``evaluation``), the selection criterion
lives here instead of ``evaluation/common/metrics.py``.

The math mirrors ``evaluation.common.metrics.compute_early_warning_auc``
verbatim (both are migrated from the legacy ``utils.evaluation``
implementation); a parity test
(``tests/models/common/test_auc_parity.py``) guards the two against
silent drift. The *reported* metric remains
``evaluation.common.metrics.compute_early_warning_auc``; this module is
used only for model-level hyperparameter selection on validation data.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _sanitize_scores(scores: np.ndarray, *, fill: float = 0.5) -> np.ndarray:
    return np.nan_to_num(scores, nan=fill, posinf=1.0, neginf=0.0)


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
    """Early-warning AUC on the validation split (model selection).

    Per-trajectory max score over the early window ``[tau - 50, tau - 5)``
    for signal trajectories vs ``[T - 50, T - 5)`` for null trajectories,
    ranked with ``roc_auc_score``. ``NaN`` scores fill with the neutral
    mid-point ``0.5`` before ranking (plan A4 of the benchmark plan).

    Args:
        scores_signal: ``(B_sig, T)`` alarm scores on validation signal.
        bif_times_signal: ``(B_sig,)`` bifurcation times.
        is_pos_signal: ``(B_sig,)`` boolean positive flags.
        seq_lens_signal: ``(B_sig,)`` valid prefix lengths.
        scores_null: ``(B_null, T)`` alarm scores on validation nulls.
        seq_lens_null: ``(B_null,)`` valid prefix lengths.

    Returns:
        The AUC in ``[0, 1]``, or ``NaN`` when fewer than two label
        classes are present.
    """
    scores_signal = _sanitize_scores(scores_signal, fill=0.5)
    scores_null = _sanitize_scores(scores_null, fill=0.5)
    vals: list[float] = []
    labels: list[int] = []
    for i in range(len(scores_signal)):
        tau = float(bif_times_signal[i])
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0:
            t_start = max(0, int(tau - early_start_delta))
            t_end = max(0, int(tau - early_end_delta))
            window = scores_signal[i, t_start:t_end]
            if len(window) > 0:
                vals.append(float(np.max(window)))
                labels.append(1)
    for i in range(len(scores_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            t_start = max(0, int(T - early_start_delta))
            t_end = max(0, int(T - early_end_delta))
            window = scores_null[i, t_start:t_end]
            if len(window) > 0:
                vals.append(float(np.max(window)))
                labels.append(0)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, vals))


__all__ = ["compute_early_warning_auc"]
