from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CSDKalmanObserver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        lstm_head: bool = False,
        lstm_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lstm_head = lstm_head
        self.lstm_dim = lstm_dim

        A_init = torch.eye(latent_dim) * 0.95
        self.A = nn.Parameter(A_init)
        self.C = nn.Parameter(torch.randn(input_dim, latent_dim).mul(0.1))
        self.K = nn.Parameter(torch.randn(latent_dim, input_dim).mul(0.1))

        if lstm_head:
            self.lstm_cell = nn.LSTMCell(latent_dim, lstm_dim)
            self.out_head = nn.Sequential(
                nn.LayerNorm(lstm_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(lstm_dim, lstm_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(lstm_dim // 2, 1),
            )
        else:
            h_dim = max(latent_dim // 2, 2)
            self.head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(latent_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(h_dim, 1),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.lstm_head:
            for n, p in self.lstm_cell.named_parameters():
                if "weight" in n:
                    nn.init.orthogonal_(p)
            for m in self.out_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        d = self.latent_dim
        device = x.device

        A = self.A
        C = self.C
        K = self.K

        z = torch.zeros(B, d, device=device)

        if self.lstm_head:
            hx = torch.zeros(B, self.lstm_dim, device=device)
            cx = torch.zeros(B, self.lstm_dim, device=device)
            eye_mat = torch.eye(d, device=device)

        logits = []
        zs = []

        for t in range(T):
            obs = x[:, t, :]
            if mask is not None:
                obs = obs * mask[:, t, :]

            if self.lstm_head:
                hx, cx = self.lstm_cell(z, (hx, cx))
                logits_t = self.out_head(hx)
                alpha_t = torch.sigmoid(logits_t)
                A_t = (1.0 - alpha_t.unsqueeze(-1)) * A.unsqueeze(0) + alpha_t.unsqueeze(-1) * eye_mat.unsqueeze(0)
                z_pred = torch.bmm(A_t, z.unsqueeze(-1)).squeeze(-1)
                logits.append(logits_t)
            else:
                z_pred = z @ A.T
                logits.append(self.head(z))

            y_pred = z_pred @ C.T
            residual = obs - y_pred
            z = z_pred + residual @ K.T

            zs.append(z)

        logits = torch.stack(logits, dim=1).squeeze(-1)
        zs = torch.stack(zs, dim=1)

        return logits, zs, A, K, C

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CSDKalmanObserver(input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"lstm_head={self.lstm_head}, "
            f"params={self.count_parameters():,})"
        )
