from __future__ import annotations

from typing import Optional, Tuple, Union

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
        aux_head: bool = False,
        aux_dim: int = 2,
        parity_aware: bool = False,
        parity_channel: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lstm_head = lstm_head
        self.lstm_dim = lstm_dim
        self.aux_head = aux_head
        self.aux_dim = aux_dim
        self.parity_aware = parity_aware

        if parity_aware:
            if latent_dim % 2 != 0:
                raise ValueError(
                    f"parity_aware requires latent_dim even, got {latent_dim}"
                )
            if parity_channel is None:
                parity_channel = input_dim - 1
            self.parity_channel = parity_channel

            A_init = torch.eye(latent_dim) * 0.95
            self.A_e = nn.Parameter(A_init.clone())
            self.A_o = nn.Parameter(A_init.clone())
            self.C_e = nn.Parameter(torch.randn(input_dim, latent_dim).mul(0.1))
            self.C_o = nn.Parameter(torch.randn(input_dim, latent_dim).mul(0.1))
            self.K_e = nn.Parameter(torch.randn(latent_dim, input_dim).mul(0.1))
            self.K_o = nn.Parameter(torch.randn(latent_dim, input_dim).mul(0.1))
            self.alternans_head = nn.Linear(1, 1)
            h_dim = max(latent_dim // 2, 2)
            self.head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(latent_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(h_dim, 1),
            )
        else:
            self.A = nn.Parameter(torch.eye(latent_dim) * 0.95)
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
                if aux_head:
                    self.aux_out = nn.Sequential(
                        nn.LayerNorm(lstm_dim),
                        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                        nn.Linear(lstm_dim, max(lstm_dim // 2, aux_dim)),
                        nn.GELU(),
                        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                        nn.Linear(max(lstm_dim // 2, aux_dim), aux_dim),
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
        if self.parity_aware:
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.xavier_uniform_(self.alternans_head.weight)
            nn.init.zeros_(self.alternans_head.bias)
        elif self.lstm_head:
            for n, p in self.lstm_cell.named_parameters():
                if "weight" in n:
                    nn.init.orthogonal_(p)
            for m in self.out_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            if self.aux_head:
                for m in self.aux_out.modules():
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
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None],
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None],
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None],
        Optional[torch.Tensor],
    ]:
        B, T, _ = x.shape
        d = self.latent_dim
        device = x.device

        if self.parity_aware:
            half = d // 2
            z = torch.zeros(B, d, device=device)
            logits = []
            alt_logits = []
            zs = []

            for t in range(T):
                obs = x[:, t, :]
                if mask is not None:
                    obs = obs * mask[:, t, :]

                p = x[:, t, self.parity_channel]
                is_even = (p >= 0).float().view(B, 1, 1)

                A_t = is_even * self.A_e.unsqueeze(0) + (1 - is_even) * self.A_o.unsqueeze(0)
                C_t = is_even * self.C_e.unsqueeze(0) + (1 - is_even) * self.C_o.unsqueeze(0)
                K_t = is_even * self.K_e.unsqueeze(0) + (1 - is_even) * self.K_o.unsqueeze(0)

                z_pred = torch.bmm(A_t, z.unsqueeze(-1)).squeeze(-1)
                y_pred = torch.bmm(z_pred.unsqueeze(1), C_t.transpose(-1, -2)).squeeze(1)
                residual = obs - y_pred
                z = z_pred + torch.bmm(residual.unsqueeze(1), K_t.transpose(-1, -2)).squeeze(1)

                logits.append(self.head(z))

                e_part = z[:, :half]
                o_part = z[:, half:]
                a_t = torch.norm(e_part - o_part, dim=-1, keepdim=True)
                alt_logits.append(self.alternans_head(a_t))

                zs.append(z)

            logits = torch.stack(logits, dim=1).squeeze(-1)
            zs = torch.stack(zs, dim=1)
            alt = torch.stack(alt_logits, dim=1)
            return logits, zs, None, None, None, alt

        A = self.A
        C = self.C
        K = self.K

        z = torch.zeros(B, d, device=device)

        if self.lstm_head:
            hx = torch.zeros(B, self.lstm_dim, device=device)
            cx = torch.zeros(B, self.lstm_dim, device=device)
            eye_mat = torch.eye(d, device=device)

        logits = []
        aux_logits = []
        zs = []

        for t in range(T):
            obs = x[:, t, :]
            if mask is not None:
                obs = obs * mask[:, t, :]

            if self.lstm_head:
                hx, cx = self.lstm_cell(z, (hx, cx))
                logits_t = self.out_head(hx)
                if self.aux_head:
                    aux_logits.append(self.aux_out(hx))
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
        aux = torch.stack(aux_logits, dim=1) if aux_logits else None

        return logits, zs, A, K, C, aux

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        parts = [
            f"input_dim={self.input_dim}",
            f"latent_dim={self.latent_dim}",
            f"params={self.count_parameters():,}",
        ]
        if self.parity_aware:
            parts.append("parity_aware=True")
        if self.lstm_head:
            parts.append(f"lstm_head=True, lstm_dim={self.lstm_dim}")
        if self.aux_head:
            parts.append("aux_head=True")
        return f"CSDKalmanObserver({', '.join(parts)})"
