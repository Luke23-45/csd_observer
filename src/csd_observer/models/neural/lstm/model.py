"""LSTM alarm-network baseline (plan A6.4, family: recurrent).

A single-layer-stack LSTM encoder over the scalar dominant mode
(``(B, T, C)`` observed features) followed by a per-step MLP alarm head
producing logits. This is the classical deep-learning CSD baseline
(recurrent encoder + alarm head, BCE on late-window positives).

References:

* Bury et al. (2021), *PNAS* 118(39): e2106140118 -- deep recurrent
  early-warning indicators (LSTM) for critical slowing down.
* Scheffer et al. (2009), *Nature* 461: 53-59 -- generic CSD indicators
  the network is trained to emulate.

The network is deliberately small (capacity budget, plan A6.4): one
hidden recurrent layer, no bidirectional stacking (causality is
required for online early warning).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LstmAlarmNet(nn.Module):
    """Causal LSTM encoder + per-step alarm head.

    Args:
        in_channels: number of feature channels (1 = scalar mode).
        hidden_size: LSTM hidden width.
        num_layers: number of stacked LSTM layers (>= 1).
        dropout: dropout probability inside the head and (for
            ``num_layers > 1``) between LSTM layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.input_proj = nn.Linear(self.in_channels, self.hidden_size)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=(float(dropout) if self.num_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alarm logits.

        Args:
            x: ``(B, T, C)`` feature batch (already padded).

        Returns:
            ``(B, T)`` logits.
        """
        h = torch.relu(self.input_proj(x))
        out, _ = self.lstm(h)
        return self.head(out).squeeze(-1)


__all__ = ["LstmAlarmNet"]
