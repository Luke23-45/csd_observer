from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SpectralRadiusLoss(nn.Module):
    """Penalizes the spectral radius of the observer's error dynamics.

    Constrains ρ((I - KC)A) < threshold, keeping the observer stable
    while allowing near-instability dynamics to emerge naturally.
    """

    def __init__(
        self,
        n_power_iter: int = 10,
        threshold: float = 0.95,
        weight: float = 0.1,
    ):
        super().__init__()
        self.n_power_iter = n_power_iter
        self.threshold = threshold
        self.weight = weight

    @staticmethod
    def _power_iteration(M: torch.Tensor, n_iter: int) -> torch.Tensor:
        d = M.shape[0]
        v = torch.randn(d, device=M.device)
        v = v / (v.norm() + 1e-8)
        for _ in range(n_iter):
            v = M @ v
            v_norm = v.norm()
            if v_norm > 1e-8:
                v = v / v_norm
        rho_hat = v @ (M @ v)
        return rho_hat

    def forward(
        self,
        A: torch.Tensor,
        K: torch.Tensor,
        C: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        d = A.shape[0]
        eye = torch.eye(d, device=A.device, dtype=A.dtype)
        M = (eye - K @ C) @ A
        rho = self._power_iteration(M, self.n_power_iter)
        penalty = torch.clamp(rho - self.threshold, min=0.0)
        loss = self.weight * penalty
        return {
            "loss": loss,
            "spectral_radius": rho.detach(),
        }
