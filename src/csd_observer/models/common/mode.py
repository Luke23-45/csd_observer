"""Shared preprocessing for the score pipeline (mode extraction).

Only the scalar dominant-mode extraction survives the removal of the
spectral-drift observer; it is required by ``models/common/indicators.py``
before any raw indicator runs.
"""

from __future__ import annotations

import numpy as np

from csd_observer.models.common.systems import SUPPORTED_SYSTEMS


def extract_mode(features: np.ndarray, system: str) -> np.ndarray:
    """Extract the scalar dominant-mode sequence for a system.

    * ``fold`` / ``logistic``: the single observed channel.
    * ``hopf``: the radial mode ``r = sqrt(x1^2 + x2^2)`` of the Hopf
      normal form ``dr = (mu r - r^3) dt`` -- i.e. the fold-type stability
      coordinate of the Hopf system.
    * ``subcritical_hopf``: two channels -> radial mode as for ``hopf``
      (synthetic); one channel -> the pre-extracted amplitude envelope
      (TAC per §5.4, where the radial mode is observed directly, e.g.
      as a dominant-mode Hilbert envelope).
    * ``transcritical``: the single observed channel (per-replicate
      population counts, DaphniaExt per §4).

    Args:
        features: ``(B, T, C)`` float array of observed features.
        system: one of ``SUPPORTED_SYSTEMS`` (``fold``, ``hopf``,
            ``logistic``, ``subcritical_hopf``, ``transcritical``).

    Returns:
        ``(B, T)`` float32 array.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be (B, T, C), got shape {features.shape}")
    if system not in SUPPORTED_SYSTEMS:
        raise ValueError(
            f"Unknown system: {system!r}. Options: {', '.join(SUPPORTED_SYSTEMS)}"
        )
    if features.shape[-1] < 1:
        raise ValueError(f"system {system!r} requires at least 1 channel")
    if features.shape[-1] >= 2 and system in ("hopf", "subcritical_hopf"):
        return np.sqrt(features[..., 0] ** 2 + features[..., 1] ** 2).astype(np.float32)
    if system == "hopf":
        raise ValueError(
            f"hopf requires at least 2 channels, got {features.shape[-1]}"
        )
    return np.asarray(features[..., 0], dtype=np.float32)


__all__ = ["extract_mode"]
