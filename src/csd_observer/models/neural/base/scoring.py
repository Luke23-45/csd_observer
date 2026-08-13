"""Batched neural scoring for the alarm baselines (R2.5).

The neural methods evaluate the full ``(B, T, C)`` feature array by
forwarding **batches of whole trajectories** — never time-chunking.
This is deliberate: the TAC length mismatch is resolved at the data
level (``processing.analysis_window`` crops the real datasets before
the bundle reaches the models), and the networks are causal and
per-timestep, so trajectory-wise batching is byte-identical to a
single full-array forward while keeping memory bounded on the long
real datasets.

Padding semantics: cells beyond ``seq_lengths`` are forced to ``NaN``
so downstream evaluation never reads them as alarms.
"""

from __future__ import annotations

import numpy as np
import torch


def score_batched(
    net: torch.nn.Module,
    features: np.ndarray,
    seq_lengths: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Alarm probabilities ``(B, T)`` float32 for a full feature array.

    Args:
        net: the fitted alarm network (``(B, T, C) -> (B, T)`` logits).
        features: ``(B, T, C)`` float array.
        seq_lengths: ``(B,)`` integer valid-prefix lengths.
        batch_size: maximum trajectories per forward pass (>= 1).
        device: compute device.

    Returns:
        ``(B, T)`` float32 probabilities; ``NaN`` beyond ``seq_lengths``.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    features = np.asarray(features, dtype=np.float32)
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
    if features.ndim != 3 or seq_lengths.ndim != 1 or seq_lengths.shape[0] != features.shape[0]:
        raise ValueError(
            f"features must be (B, T, C) with seq_lengths (B,), got "
            f"{features.shape} vs {seq_lengths.shape}"
        )
    B, T = features.shape[0], features.shape[1]
    out = np.full((B, T), np.nan, dtype=np.float32)
    net = net.to(device).eval()
    with torch.no_grad():
        for start in range(0, B, batch_size):
            stop = min(start + batch_size, B)
            feats = torch.as_tensor(features[start:stop], device=device)
            probs = torch.sigmoid(net(feats)).cpu().numpy().astype(np.float32)
            for i, row in enumerate(range(start, stop)):
                n = int(seq_lengths[row])
                out[row, :n] = probs[i, :n]
    return out


__all__ = ["score_batched"]
