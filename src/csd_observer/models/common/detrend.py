"""Shared detrending helper for the CSD indicator implementations.

Relocated verbatim from ``csd_observer.utils.metrics`` (phase 2 of the
plan): the indicator modules under ``models/indicators/`` consume
``_linear_detrend`` through this module so no implementation imports the
legacy ``utils`` package.
"""

from __future__ import annotations

import numpy as np


def _linear_detrend(seg: np.ndarray) -> np.ndarray:
    """Linearly detrend a 1-D segment via least squares (legacy definition).

    Fits ``x -> a*x + b`` to the segment by OLS (``numpy.linalg.lstsq``
    with a column of ones) and returns the residuals. The residual has
    zero mean by construction, which the indicator wrappers rely on.

    Args:
        seg: ``(n,)`` numeric segment.

    Returns:
        ``(n,)`` detrended segment with the same dtype as ``seg``.
    """
    x = np.arange(len(seg), dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, seg.astype(np.float64), rcond=None)
    return (seg.astype(np.float64) - A @ coeffs).astype(seg.dtype)


__all__ = ["_linear_detrend"]
