"""Fixed-FPR alarm-threshold calibration.

Migrated from ``csd_observer.utils.evaluation`` so the canonical
implementation lives under ``evaluation/``; behaviour preserved
bit-for-bit on NaN-free inputs. Plan layout: calibration lives in
``evaluation/common/calibration.py``, distinct from the metric
primitives in ``evaluation/common/metrics.py``.
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(
    scores_val_null: np.ndarray,
    seq_lengths: np.ndarray,
    fpr_target: float,
) -> float:
    """Fixed-FPR alarm threshold from the validation null scores.

    Concatenates the valid prefixes of all validation-null trajectories
    and takes the ``100 * (1 - fpr_target)`` percentile over the
    *finite* steps only.
    """
    if not (0.0 < float(fpr_target) < 1.0):
        raise ValueError(f"fpr_target must be in (0, 1), got {fpr_target}")
    scores_val_null = np.asarray(scores_val_null)
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
    if scores_val_null.ndim != 2 or seq_lengths.shape != (scores_val_null.shape[0],):
        raise ValueError("scores_val_null must be (B,T) and seq_lengths must be (B,)")
    if np.any(seq_lengths < 0) or np.any(seq_lengths > scores_val_null.shape[1]):
        raise ValueError("seq_lengths must be within [0, T]")
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


__all__ = ["calibrate_threshold"]
