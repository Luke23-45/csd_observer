"""Alarm-label construction for neural baselines.

Every neural baseline trains with the same label contract (plan A6.4
fairness: identical input pipeline and splits as the indicators):

* per-trajectory target ``target[b, t] = 1`` iff the trajectory is
  positive and ``t`` lies in the late-window ``[tau - label_window, tau)``
  of the bifurcation time ``tau``;
* null trajectories get an all-zero target stream;
* steps at or beyond ``seq_lengths[b]`` are ignored by the loss
  (padding must never contribute gradients).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_alarm_labels(
    bundle: dict[str, Any],
    *,
    label_window: int = 60,
) -> np.ndarray:
    """Float targets ``(B, T)`` for a dataset-registry bundle.

    Args:
        bundle: registry bundle with ``features``, ``seq_lengths``,
            ``bifurcation_times``, ``is_positive``.
        label_window: number of pre-transition steps marked positive
            (default 60, matching the legacy early-warning window).

    Returns:
        ``(B, T)`` float32 array of targets (``1.0`` in the late window,
        ``0.0`` elsewhere).
    """
    features = np.asarray(bundle["features"])
    seq_lengths = np.asarray(bundle["seq_lengths"], dtype=np.int64)
    bifurcation_times = np.asarray(bundle["bifurcation_times"], dtype=np.float64)
    is_positive = np.asarray(bundle["is_positive"], dtype=bool)
    B, T = features.shape[0], features.shape[1]
    label_window = max(1, int(label_window))
    targets = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        L = int(seq_lengths[b])
        if L <= 0 or not is_positive[b]:
            continue
        tau = float(bifurcation_times[b])
        if not np.isfinite(tau) or tau <= 0:
            continue
        t_end = min(L, int(np.floor(tau)))
        t_start = max(0, t_end - label_window)
        if t_end > t_start:
            targets[b, t_start:t_end] = 1.0
    return targets


def valid_mask(seq_lengths: np.ndarray, T: int) -> np.ndarray:
    """Boolean ``(B, T)`` mask: True inside each trajectory's prefix."""
    lengths = np.asarray(seq_lengths, dtype=np.int64)
    B = lengths.shape[0]
    mask = np.zeros((B, T), dtype=bool)
    for b in range(B):
        mask[b, : max(0, min(int(lengths[b]), T))] = True
    return mask


def causal_running_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """Causal running mean of ``x`` along the time axis (window ``k``).

    ``out[..., t] = mean(x[..., max(0, t-k+1):t+1])``; the window is
    causal so the smoothed stream remains an online quantity.

    Note: the persistence-aware *loss* lives in
    ``training/common/losses.py`` (plan A7) and is injected into the
    neural baselines by the orchestration runner (the one-way import
    rule forbids ``models`` from importing ``training``); this helper is
    shared by both sides.
    """
    k = max(1, int(k))
    cum = torch.cumsum(x, dim=-1)
    n = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
    denom = torch.minimum(n, torch.as_tensor(k, dtype=x.dtype, device=x.device))
    prev = torch.zeros_like(cum)
    prev[..., k:] = cum[..., :-k]
    return (cum - prev) / denom


def valid_mask_tensor(seq_lengths: torch.Tensor, T: int) -> torch.Tensor:
    """Boolean ``(B, T)`` mask tensor: True inside each prefix."""
    lens = seq_lengths.to(torch.int64)
    idx = torch.arange(T, dtype=torch.int64, device=lens.device)
    return idx[None, :] < lens[:, None]


__all__ = ["build_alarm_labels", "causal_running_mean", "valid_mask", "valid_mask_tensor"]
