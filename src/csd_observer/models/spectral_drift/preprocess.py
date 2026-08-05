"""Preprocessing for the spectral-drift observer (mode extraction and causal centring)."""

from __future__ import annotations

import numpy as np


def extract_mode(features: np.ndarray, system: str) -> np.ndarray:
    """Extract the scalar dominant-mode sequence for a system.

    * ``fold`` / ``logistic``: the single observed channel.
    * ``hopf``: the radial mode ``r = sqrt(x1^2 + x2^2)`` of the Hopf
      normal form ``dr = (mu r - r^3) dt`` -- i.e. the fold-type stability
      coordinate of the Hopf system.

    Args:
        features: ``(B, T, C)`` float array of observed features.
        system: one of ``"fold"``, ``"hopf"``, ``"logistic"``.

    Returns:
        ``(B, T)`` float32 array.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be (B, T, C), got shape {features.shape}")
    if system not in ("fold", "hopf", "logistic"):
        raise ValueError(f"Unknown system: {system!r}. Options: fold, hopf, logistic")
    if system == "hopf":
        if features.shape[-1] < 2:
            raise ValueError(
                f"hopf requires at least 2 channels, got {features.shape[-1]}"
            )
        return np.sqrt(features[..., 0] ** 2 + features[..., 1] ** 2).astype(np.float32)
    if features.shape[-1] < 1:
        raise ValueError(f"system {system!r} requires at least 1 channel")
    return np.asarray(features[..., 0], dtype=np.float32)


def running_mean_center(
    seq: np.ndarray,
    seq_lengths: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """Subtract a causal running mean (centring) from each trajectory.

    Implements the preprocessing step of the formal definition: the OU
    transition is written for a centred mode (``bar_u = 0``), so the
    equilibrium is removed by a running-mean subtraction. The window is
    causal (uses only past samples) so the observer remains online. No
    ground-truth parameter values are used.

    Args:
        seq: ``(B, T)`` float array of mode sequences.
        seq_lengths: ``(B,)`` integer array of valid prefix lengths.
        window: running-mean window size (effective window is
            ``min(window, t+1)`` at each step).

    Returns:
        ``(B, T)`` float32 array, centred per trajectory.
    """
    seq = np.asarray(seq, dtype=np.float32)
    B, T = seq.shape
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
    if seq_lengths.shape[0] != B:
        raise ValueError(
            f"seq_lengths length {seq_lengths.shape[0]} != B {B}"
        )
    window = max(1, int(window))
    centered = np.zeros_like(seq)
    cumsum = np.cumsum(seq, axis=1)
    for b in range(B):
        L = int(seq_lengths[b])
        if L <= 0:
            continue
        L = min(L, T)
        W = min(window, L)
        mean = np.zeros(T, dtype=np.float32)
        for t in range(L):
            lo = max(0, t - W + 1)
            hi = t + 1
            mean[t] = (
                cumsum[b, hi - 1] - (cumsum[b, lo - 1] if lo > 0 else 0.0)
            ) / (hi - lo)
        centered[b, :L] = seq[b, :L] - mean[:L]
    return centered
