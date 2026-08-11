"""Causal dilated temporal-convolution alarm network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TcnAlarmNet(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_size: int = 32, kernel_size: int = 5, levels: int = 3) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0 or levels < 1:
            raise ValueError("kernel_size must be positive odd and levels must be >= 1")
        blocks: list[nn.Module] = []
        current = int(in_channels)
        for level in range(int(levels)):
            dilation = 2 ** level
            blocks.extend([nn.Conv1d(current, hidden_size, kernel_size, dilation=dilation), nn.ReLU()])
            current = hidden_size
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv1d(current, 1, 1)
        self.kernel_size = int(kernel_size)
        self.levels = int(levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for i, layer in enumerate(self.blocks):
            if isinstance(layer, nn.Conv1d):
                dilation = 2 ** (i // 2)
                x = F.pad(x, ((self.kernel_size - 1) * dilation, 0))
            x = layer(x)
        return self.head(x).squeeze(1)


__all__ = ["TcnAlarmNet"]
