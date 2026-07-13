from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CSDKalmanObserver(nn.Module):
    """Adaptive Criticality Kalman Observer.

    The Kalman filter produces a latent state z[t] from observations.
    Two head options:
      - MLP head: processes z[t] independently per timestep with static physics (default).
      - Physics-Informed LSTM head: processes z[t-1] through a causal LSTM cell to 
        estimate a Criticality Index (alpha_t in [0, 1]). This index directly
        modulates the Kalman Filter's transition matrix, driving the restoring
        force to zero (Identity matrix) as the system approaches a bifurcation.

    Parameters
    ----------
    input_dim : int
    latent_dim : int
        Latent state dimension (default 4).
    lstm_head : bool
        If True, use Physics-Informed LSTM cell.
    lstm_dim : int
        LSTM hidden dimension (default 8, only used when lstm_head=True).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        lstm_head: bool = False,
        lstm_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lstm_head = lstm_head
        self.lstm_dim = lstm_dim

        self.obs_proj = nn.Linear(input_dim, 1)

        A_init = torch.eye(latent_dim) * 0.95
        self.A = nn.Parameter(A_init)
        self.C = nn.Parameter(torch.randn(1, latent_dim).mul(0.1))
        self.K = nn.Parameter(torch.randn(latent_dim, 1).mul(0.1))

        if lstm_head:
            # Physics-Informed Criticality RNN
            self.lstm_cell = nn.LSTMCell(latent_dim, lstm_dim)
            self.out_head = nn.Sequential(
                nn.LayerNorm(lstm_dim),
                nn.Linear(lstm_dim, lstm_dim // 2),
                nn.GELU(),
                nn.Linear(lstm_dim // 2, 1),
            )
        else:
            h_dim = max(latent_dim // 2, 2)
            self.head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, h_dim),
                nn.GELU(),
                nn.Linear(h_dim, 1),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.obs_proj.weight)
        nn.init.zeros_(self.obs_proj.bias)
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
        """Forward pass.

        Returns
        -------
        logits : [B, T] risk logits (pre-sigmoid criticality).
        zs : [B, T, d] latent states.
        A, K, C : [d, d], [d, 1], [1, d] matrices (A is the stable base matrix).
        """
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
            I = torch.eye(d, device=device)
            
        logits = []
        zs = []

        for t in range(T):
            x_t = x[:, t, :]
            if mask is not None:
                x_t = x_t * mask[:, t, :]

            obs = self.obs_proj(x_t)
            
            if self.lstm_head:
                # 1. Update Criticality based on previous state
                hx, cx = self.lstm_cell(z, (hx, cx))
                logits_t = self.out_head(hx)
                alpha_t = torch.sigmoid(logits_t)
                
                # 2. Modulate physics: A_t approaches Identity as alpha_t approaches 1
                A_t = (1.0 - alpha_t.unsqueeze(-1)) * A.unsqueeze(0) + alpha_t.unsqueeze(-1) * I.unsqueeze(0)
                
                # 3. Time-varying Kalman prediction
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
