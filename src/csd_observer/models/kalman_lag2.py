from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class ClassicalKalmanLag2(nn.Module):
    """Classical Kalman filter tracking lag-2 autocorrelation dynamics.

    State: z_t = [mu_t, delta_t]^T  (lag-2 value + discrete derivative)
    Transition: F = [[1, 1], [0, 1]]  (constant-velocity / CWNA model)
    Observation: H = [1, 0]  (observe smoothed lag-2 only)

    All matrices are fixed (buffers, not Parameters).
    The only tunable parameter is q (process noise intensity), set at construction.
    """

    def __init__(self, q: float = 1e-3, r: float = 1.0) -> None:
        super().__init__()
        self.r = r

        self.register_buffer("F", torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
        self.register_buffer("H", torch.tensor([[1.0, 0.0]]))
        self.register_buffer(
            "Q",
            q * torch.tensor([[1.0 / 3.0, 1.0 / 2.0], [1.0 / 2.0, 1.0]]),
        )
        self.register_buffer(
            "P_init",
            torch.tensor([[r, 0.0], [0.0, r / 10.0]]),
        )

    def forward(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T = y.shape
        device = y.device

        mu_hat = torch.zeros(B, T, device=device)
        delta_hat = torch.zeros(B, T, device=device)
        innovation = torch.zeros(B, T, device=device)

        z = torch.zeros(B, 2, device=device)
        P = self.P_init.unsqueeze(0).expand(B, -1, -1).clone()
        eye = torch.eye(2, device=device)

        for t in range(T):
            z_pred = z @ self.F.T
            P_pred = self.F @ P @ self.F.T + self.Q

            y_pred = z_pred[:, 0]
            innov = y[:, t] - y_pred
            innovation[:, t] = innov

            S = P_pred[:, 0, 0] + self.r
            K = P_pred[:, :, 0] / S.unsqueeze(-1)

            z = z_pred + K * innov.unsqueeze(-1)
            L = eye.unsqueeze(0) - K.unsqueeze(-1) @ self.H.unsqueeze(0)
            P = L @ P_pred @ L.transpose(-1, -2) + self.r * K.unsqueeze(-1) @ K.unsqueeze(-2)
            P = (P + P.transpose(-1, -2)) / 2.0

            mu_hat[:, t] = z[:, 0]
            delta_hat[:, t] = z[:, 1]

        return {
            "mu_hat": mu_hat,
            "delta_hat": delta_hat,
            "innovation": innovation,
            "y": y,
        }


def grid_search_q(
    y_val: np.ndarray,
    bifs_val: np.ndarray,
    is_pos_val: np.ndarray,
    seq_lens_val: np.ndarray,
    q_grid: List[float] | None = None,
    device: torch.device | None = None,
) -> float:
    if q_grid is None:
        q_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from csd_observer.utils.metrics import compute_early_warning_auc

    best_q = q_grid[0]
    best_auc = -1.0

    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    for q in q_grid:
        kalman = ClassicalKalmanLag2(q=q, r=1.0).to(device)
        kalman.eval()
        with torch.no_grad():
            out = kalman(y_val_t)
        scores = out["mu_hat"].cpu().numpy()

        scores_sig = scores[is_pos_val]
        scores_null = scores[~is_pos_val]
        bifs_sig = bifs_val[is_pos_val]
        is_pos_sig = is_pos_val[is_pos_val]
        lens_sig = seq_lens_val[is_pos_val]
        lens_null = seq_lens_val[~is_pos_val]

        auc = compute_early_warning_auc(
            scores_sig, bifs_sig, is_pos_sig, lens_sig,
            scores_null, lens_null,
        )
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_q = q

    return best_q


class KalmanLag2MLPHead(nn.Module):
    """Tiny per-step MLP on [mu_hat, delta_hat, innovation, y] features.

    Architecture: LayerNorm -> Linear(4->4) -> GELU -> Linear(4->1)
    Total: 25 trainable parameters.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class KalmanLag2Net(nn.Module):
    """Fixed classical Kalman smoother + learned MLP head.

    The Kalman filter matrices are fixed (no gradient flow).
    Only the 25-parameter MLP head is trained via BCE loss.
    """

    def __init__(self, q: float = 1e-3, r: float = 1.0) -> None:
        super().__init__()
        self.kalman = ClassicalKalmanLag2(q=q, r=r)
        self.head = KalmanLag2MLPHead(input_dim=4, hidden_dim=4)

    def forward(self, lag2_scores: torch.Tensor) -> torch.Tensor:
        kalman_out = self.kalman(lag2_scores)
        features = torch.stack(
            [
                kalman_out["mu_hat"],
                kalman_out["delta_hat"],
                kalman_out["innovation"],
                kalman_out["y"],
            ],
            dim=-1,
        )
        return self.head(features)
