"""PatchTST-style transformer alarm baseline (plan §6.4, family b3).

The network follows the patch-tokenisation scheme of Nie et al., "A
Time Series is Worth 64 Words: Long-term Forecasting with
Transformers", ICLR 2023 (PatchTST): the ``(B, T, C)`` series is cut
into fixed-length patches, each patch is linearly embedded into a
``d_model`` token, and a transformer encoder processes the token
sequence.

Two adaptations make the alarm head *strictly causal* (the persistence
protocol rejects future leakage):

* the self-attention over patches is causally masked (patch ``i``
  attends only to patches ``j <= i``);
* a score at step ``t`` is emitted only from the representation of the
  *latest patch completed at ``t``* (its window ends at or before
  ``t``). Steps before the first patch completes emit a neutral logit
  of 0 (``p = 0.5``). The alarm signal is therefore piecewise-constant
  between patch completions and carries an architectural latency of at
  most one patch window -- the honest cost of the patch representation,
  documented for cross-method comparison.

The positional encoding is sinusoidal (computed for the actual number
of patches), so sequence length is not capped by a learned position
table. Capacity (defaults): ``d_model=64``, ``n_heads=2``, ``n_layers=2``
-- deliberately in the same budget as the LSTM/TCN baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _sinusoidal_positions(n_patches: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Absolute sinusoidal positional encoding over patch indices."""
    pe = torch.zeros(n_patches, d_model, device=device)
    half = d_model // 2
    if half == 0:
        return pe
    pos = torch.arange(n_patches, dtype=torch.float32, device=device).unsqueeze(1)
    theta = pos / (
        10000.0 ** (torch.arange(half, dtype=torch.float32, device=device) / d_model)
    )
    pe[:, 0:half] = torch.sin(theta)
    pe[:, half : 2 * half] = torch.cos(theta)
    return pe


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(
            torch.full((n, n), float("-inf"), device=x.device), diagonal=1
        )
        att = self.drop(torch.softmax(att + causal, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(b, n, -1)
        return self.proj(y)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchTstAlarmNet(nn.Module):
    """Causal patch-transformer alarm network.

    Args:
        in_channels: number of feature channels (1 = scalar mode).
        patch_len: patch window length (steps).
        stride: step between patch starts; ``None`` = ``patch_len``
            (non-overlapping patches, PatchTST default).
        d_model: token width (must be divisible by ``n_heads``).
        n_heads: attention heads per layer.
        n_layers: transformer layer count.
        mlp_ratio: MLP hidden width as a multiple of ``d_model``.
        dropout: dropout probability inside attention/MLP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_len: int = 16,
        stride: int | None = None,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if patch_len < 1:
            raise ValueError(f"patch_len must be >= 1, got {patch_len}")
        stride = patch_len if stride is None else int(stride)
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if d_model < 1 or d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be a positive multiple of n_heads={n_heads}")
        if n_heads < 1 or n_layers < 1:
            raise ValueError("n_heads and n_layers must be >= 1")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio}")
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.patch_embed = nn.Linear(self.patch_len * int(in_channels), self.d_model)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(self.d_model, int(n_heads), float(mlp_ratio), float(dropout))
             for _ in range(int(n_layers))]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, 1)

    def _n_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_len) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alarm logits.

        Args:
            x: ``(B, T, C)`` feature batch (already padded).

        Returns:
            ``(B, T)`` logits, piecewise-constant over completed
            patches; neutral (0.0) before the first patch completes.
        """
        b, t, c = x.shape
        if t < self.patch_len:
            raise ValueError(
                f"sequence length {t} is shorter than patch_len={self.patch_len}"
            )
        n_patches = self._n_patches(t)
        starts = torch.arange(n_patches, device=x.device) * self.stride
        idx = starts[:, None] + torch.arange(self.patch_len, device=x.device)[None, :]
        windows = x[:, idx.clamp(max=t - 1), :]
        windows[:, idx >= t] = 0.0
        tok = self.patch_embed(windows.reshape(b, n_patches, self.patch_len * c))
        tok = tok + _sinusoidal_positions(n_patches, self.d_model, x.device)
        for block in self.blocks:
            tok = block(tok)
        logits = self.head(self.norm(tok)).squeeze(-1)  # (B, N)

        steps = torch.arange(t, device=x.device)
        idx_latest = (steps + 1 - self.patch_len) // self.stride
        valid = idx_latest >= 0
        idx_latest = idx_latest.clamp(min=0, max=n_patches - 1)
        emitted = logits[:, idx_latest]  # (B, T)
        return torch.where(valid, emitted, torch.zeros_like(emitted))


__all__ = ["PatchTstAlarmNet"]
