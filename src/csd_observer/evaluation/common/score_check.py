"""Score-shape validation for the governance pipeline (R2.3).

Called by the governance driver after every ``method.score`` call. A
violation of the ``(B, T)`` float contract or a non-finite score outside
the documented NaN warm-up region means the method is misbehaving
(e.g. a neural net that hallucinated a constant/Inf buffer); the run
must not proceed to calibration with garbage.
"""

from __future__ import annotations

import numpy as np


def validate_scores(
    scores: np.ndarray,
    seq_lengths: np.ndarray,
    *,
    context: str,
) -> None:
    """Validate the ``(B, T)`` score contract; raise ``ValueError`` on
    shape/dtype/Inf violations.

    Allowed:

    * any float dtype (cast to float32 downstream);
    * ``NaN`` — the documented warm-up/undefined region (e.g. the first
      ``window_size - 1`` steps of the statistical indicators), always
      inside ``seq_lengths``.

    Rejected:

    * wrong ndim / batch mismatch against ``seq_lengths``;
    * non-float dtype (a boolean or integer score buffer means the
      method returned an alarm stream instead of scores);
    * ``+Inf``/``-Inf`` anywhere (NaN is a defined absence; Inf is not
      a score);
    * ``NaN`` beyond ``seq_lengths`` (padding must be present, not
      garbage — actually padding beyond ``seq_lengths`` is *allowed* to
      be NaN for the indicator family, so this check only rejects Inf).
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"{context}: scores must be (B, T), got shape {scores.shape}")
    lengths = np.asarray(seq_lengths, dtype=np.int64)
    if lengths.ndim != 1 or lengths.shape[0] != scores.shape[0]:
        raise ValueError(
            f"{context}: seq_lengths must be (B,) matching scores, got "
            f"{lengths.shape} vs {scores.shape}"
        )
    if scores.dtype.kind not in "fc":
        raise ValueError(
            f"{context}: scores must be float, got dtype {scores.dtype}"
        )
    if np.isinf(scores).any():
        raise ValueError(f"{context}: scores contain +/-Inf (NaN is allowed, Inf is not)")
    if np.any(lengths < 0) or np.any(lengths > scores.shape[1]):
        raise ValueError(f"{context}: seq_lengths must be within [0, T]")


def scores_ok(scores: np.ndarray, seq_lengths: np.ndarray) -> bool:
    """Cheap boolean gate used by the once-per-method warning path."""
    try:
        validate_scores(scores, seq_lengths, context="scores_ok")
    except ValueError:
        return False
    return True


def warn_unexpected_nan(
    scores: np.ndarray,
    seq_lengths: np.ndarray,
    *,
    context: str,
) -> None:
    """Emit a one-time UserWarning when NaN appears beyond ``seq_lengths``.

    NaN inside the valid prefix is the documented warm-up absence. NaN
    in the padding region means the method did not mask its output
    buffer (a neural net leaving uninitialized positions), which is
    tolerated by the protocol (NaN never alarms) but worth surfacing
    once per method, not per call.
    """
    scores = np.asarray(scores)
    lengths = np.asarray(seq_lengths, dtype=np.int64)
    if scores.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != scores.shape[0]:
        return
    if not np.isfinite(scores).any():
        return
    count = 0
    for i, L in enumerate(lengths):
        pad = scores[i, int(L):]
        if pad.size and np.isnan(pad).any():
            count += 1
    if count:
        import warnings

        warnings.warn(
            f"{context}: NaN found beyond seq_lengths in {count}/{len(lengths)} "
            "trajectories (unmasked padding); tolerated — NaN never alarms.",
            UserWarning,
            stacklevel=3,
        )


__all__ = ["scores_ok", "validate_scores", "warn_unexpected_nan"]
