"""Shared preprocessing for the score pipeline (mode extraction).

Only the scalar dominant-mode extraction survives the removal of the
spectral-drift observer; it is required by ``models/common/indicators.py``
before any raw indicator runs.

Since R2.2 the mode is **declared by the dataset**, not inferred from
the system name: ``DatasetConfig.feature_mode`` (``radial``,
``channel_0``, ``envelope``, or ``None`` = auto). The ``system``-based
fallback below is kept only for ``None`` (backward compatibility with
hand-built configs), and raises on shape/system combinations it cannot
resolve — the dataset declaration is the contract.
"""

from __future__ import annotations

import numpy as np

from csd_observer.models.common.systems import SUPPORTED_SYSTEMS

#: Registered feature-mode names (see DatasetConfig.feature_mode).
VALID_FEATURE_MODES = ("radial", "channel_0", "envelope")


def extract_mode(
    features: np.ndarray,
    system: str,
    mode: str | None = None,
) -> np.ndarray:
    """Extract the scalar dominant-mode sequence for a dataset.

    With an explicit ``mode``:

    * ``radial``: the radial coordinate ``r = sqrt(x1^2 + x2^2)`` of the
      first two channels (Hopf-family stability coordinate; requires
      ``C >= 2``).
    * ``channel_0``: the first channel (fold/logistic state, DaphniaExt
      counts, single-channel traces).
    * ``envelope``: the first channel, which is already the pre-extracted
      amplitude envelope (TAC per §5.4).

    With ``mode=None`` (auto, legacy): the system-name inference below.

    Args:
        features: ``(B, T, C)`` float array of observed features.
        system: one of ``SUPPORTED_SYSTEMS`` (``fold``, ``hopf``,
            ``logistic``, ``subcritical_hopf``, ``transcritical``).
        mode: dataset-declared mode name, or ``None`` for auto.

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
    if mode is not None:
        if mode not in VALID_FEATURE_MODES:
            raise ValueError(
                f"Unknown feature mode: {mode!r}. Options: {', '.join(VALID_FEATURE_MODES)}"
            )
        if mode == "radial":
            if features.shape[-1] < 2:
                raise ValueError(
                    f"feature_mode=radial requires at least 2 channels, got {features.shape[-1]}"
                )
            return np.sqrt(features[..., 0] ** 2 + features[..., 1] ** 2).astype(np.float32)
        return np.asarray(features[..., 0], dtype=np.float32)
    if features.shape[-1] >= 2 and system in ("hopf", "subcritical_hopf"):
        return np.sqrt(features[..., 0] ** 2 + features[..., 1] ** 2).astype(np.float32)
    if system == "hopf":
        raise ValueError(
            f"hopf requires at least 2 channels, got {features.shape[-1]}"
        )
    return np.asarray(features[..., 0], dtype=np.float32)


__all__ = ["VALID_FEATURE_MODES", "extract_mode"]
