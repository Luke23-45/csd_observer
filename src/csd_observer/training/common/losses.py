"""Losses for causal per-step alarm networks."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from csd_observer.models.neural.base.labels import (
    causal_running_mean,  # noqa: F401  (re-exported: one-way-safe shared helper)
)


def alarm_bce(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, *,
              k_persist: int = 5, alpha: float = .5, eps: float = 1e-7) -> tuple[torch.Tensor, dict[str, Any]]:
    if probs.shape != targets.shape or probs.shape != mask.shape:
        raise ValueError("probs, targets, and mask must have identical shapes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    p = probs.float().clamp(eps, 1.0 - eps)
    y = targets.float()
    valid = mask.to(p.dtype)
    raw_terms = F.binary_cross_entropy(p, y, reduction="none")
    smooth_p = causal_running_mean(p, int(k_persist)).clamp(eps, 1.0 - eps)
    smooth_terms = F.binary_cross_entropy(smooth_p, y, reduction="none")
    denominator = valid.sum().clamp_min(1.0)
    raw = (raw_terms * valid).sum() / denominator
    smooth = (smooth_terms * valid).sum() / denominator
    return alpha * raw + (1.0 - alpha) * smooth, {"bce_raw": raw.detach(), "bce_smoothed": smooth.detach()}


__all__ = ["alarm_bce", "causal_running_mean"]
