"""The spectral-drift observer: Rao-Blackwellised particle filter over the spectral gap."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SpectralDriftObserver(nn.Module):
    """Rao-Blackwellised particle filter over the spectral gap ``c_k``.

    Discrete state-space model (formal definition, section 2):

    * ``c_{k+1} = max(c_min, c_k + eta_k)``, ``eta_k ~ N(0, Q_drift)``;
    * ``u_{k+1} = exp(-c_k dt) u_k + eps_k``,
      ``Var(eps_k) = sigma_u^2 / (2 c_k) (1 - exp(-2 c_k dt))``;
    * ``y_k = u_k + nu_k``, ``nu_k ~ N(0, r)``.

    The filter propagates ``n_particles`` particles over ``c_k``; for each
    particle the *conditional* Kalman filter over ``u_k`` is exact because
    the model is linear-Gaussian given ``c_k`` (Rao-Blackwellisation).
    Resampling is systematic, triggered when the effective sample size
    falls below ``n_particles / 2``.

    All parameters are buffers; the observer has no trainable parameters.
    """

    def __init__(
        self,
        *,
        sigma_u: float,
        r: float,
        q_drift: float,
        n_particles: int = 500,
        c_min: float = 1e-3,
        c_init: float = 0.1,
        c_init_std: float = 0.05,
        delta: float = 0.05,
        dt: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        if sigma_u <= 0:
            raise ValueError(f"sigma_u must be > 0, got {sigma_u}")
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")
        if q_drift <= 0:
            raise ValueError(f"q_drift must be > 0, got {q_drift}")
        if c_min <= 0:
            raise ValueError(f"c_min must be > 0, got {c_min}")
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if c_init <= c_min:
            raise ValueError(
                f"c_init must be > c_min, got c_init={c_init}, c_min={c_min}"
            )

        self.n_particles = n_particles
        self.c_min = float(c_min)
        self.delta = float(delta)
        self.dt = float(dt)
        self.seed = int(seed)

        for name, value in {
            "sigma_u": float(sigma_u),
            "r": float(r),
            "q_drift": float(q_drift),
            "c_init": float(c_init),
            "c_init_std": float(c_init_std),
        }.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # filter core
    # ------------------------------------------------------------------ #
    def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the filter over a batch of sequences.

        Args:
            y: ``(B, T)`` float tensor of centred mode observations.

        Returns:
            dict with keys

            * ``collapse_prob`` ``(B, T)``: posterior ``Pr(c_t < delta | y_{0:t})``;
            * ``c_hat`` ``(B, T)``: posterior mean of the spectral gap;
            * ``c_std`` ``(B, T)``: posterior std of the spectral gap;
            * ``log_lik`` ``(B,)``: predictive log-likelihood (diagnostics).
        """
        if y.ndim != 2:
            raise ValueError(f"y must be (B, T), got shape {tuple(y.shape)}")
        y = y.to(torch.float32)
        B, T = y.shape
        device = y.device
        N = self.n_particles

        generator = torch.Generator(device=device).manual_seed(self.seed)

        c_min_t = torch.as_tensor(self.c_min, device=device, dtype=torch.float32)
        delta_t = torch.as_tensor(self.delta, device=device, dtype=torch.float32)
        dt_t = torch.as_tensor(self.dt, device=device, dtype=torch.float32)
        sigma_u2 = self.sigma_u * self.sigma_u
        r_t = self.r
        q_sqrt = torch.sqrt(self.q_drift)
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=torch.float32))

        # --- initialisation ---------------------------------------------------
        c = self._truncated_normal(
            (B, N),
            mean=self.c_init,
            std=self.c_init_std,
            lower=self.c_min,
            generator=generator,
        )
        u = torch.zeros(B, N, device=device, dtype=torch.float32)
        # u_0 is drawn from the stationary distribution of the OU at the
        # particle's own initial gap c_0: p_0 = sigma_u^2 / (2 c_0).
        p_u = sigma_u2 / (2.0 * c)
        log_w = torch.zeros(B, N, device=device, dtype=torch.float32)
        weights = torch.full(
            (B, N), 1.0 / N, device=device, dtype=torch.float32
        )

        collapse_prob = torch.zeros(B, T, device=device, dtype=torch.float32)
        c_hat = torch.zeros(B, T, device=device, dtype=torch.float32)
        c_std = torch.zeros(B, T, device=device, dtype=torch.float32)
        log_lik = torch.zeros(B, device=device, dtype=torch.float32)

        for t in range(T):
            # 1. predict (skip on the first step: y_0 conditions the prior
            #    state (c_1, u_1) directly -- formal definition, section 2) --
            if t > 0:
                eta = torch.randn(B, N, generator=generator, device=device, dtype=torch.float32)
                c_pred = torch.clamp(c + q_sqrt * eta, min=c_min_t)

                # 2. conditional Kalman prediction over u. The OU transition
                #    u_{k+1} = phi(c_k) u_k uses the gap at the START of the
                #    interval, i.e. the pre-prediction particle gap c.
                phi = torch.exp(-c * dt_t)
                phi2 = phi * phi
                u_pred = phi * u
                p_pred = phi2 * p_u + sigma_u2 / (2.0 * c) * (1.0 - phi2)
                p_pred = torch.clamp(p_pred, min=1e-12)
            else:
                c_pred = c
                u_pred = u
                p_pred = p_u

            # 3. weight update (predictive likelihood) --------------------------
            y_t = y[:, t].unsqueeze(-1)  # (B, 1)
            innov = y_t - u_pred
            s = p_pred + r_t
            llh = -0.5 * (innov * innov / s + torch.log(s) + log_2pi)
            log_lik = log_lik + torch.logsumexp(log_w + llh, dim=1)

            log_w = log_w + llh
            max_log_w = torch.max(log_w, dim=1, keepdim=True).values
            weights = torch.exp(log_w - max_log_w)
            w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
            weights = weights / w_sum
            log_w = (log_w - max_log_w) - torch.log(w_sum)

            # 4. conditional Kalman update over u -------------------------------
            k_gain = p_pred / s
            u_corr = u_pred + k_gain * innov
            p_corr = (1.0 - k_gain) * p_pred

            # 5. systematic resampling if ESS is low ----------------------------
            ess = 1.0 / (weights * weights).sum(dim=1).clamp(min=1e-12)
            do_resample = ess < (self.n_particles / 2.0)
            if do_resample.any():
                idx = torch.arange(
                    N, device=device
                ).unsqueeze(0).expand(B, N).clone()
                flagged = do_resample.nonzero(as_tuple=True)[0]
                idx[flagged] = self._systematic_resample(
                    weights[flagged], generator
                )
                c_pred = torch.gather(c_pred, 1, idx)
                u_corr = torch.gather(u_corr, 1, idx)
                p_corr = torch.gather(p_corr, 1, idx)
                # only the resampled rows lose their posterior weights; the
                # particles (and weights) of non-flagged rows are untouched.
                log_w[flagged] = -np.log(N)
                weights[flagged] = 1.0 / N

            c = c_pred
            u = u_corr
            p_u = p_corr

            # 6. record posterior quantities ------------------------------------
            # collapse_prob[:, t] is Pr(c_t < delta | y_{0:t}), i.e. the
            # posterior probability that the spectral gap at index ``t``
            # has crossed the collapse threshold given all observations
            # up to and including ``y_t``. The per-index state ``c`` is
            # the *updated* (post-Kalman, post-resample) gap, so the
            # index matches.
            collapse_prob[:, t] = (
                weights * (c < delta_t).to(torch.float32)
            ).sum(dim=1)
            c_hat[:, t] = (weights * c).sum(dim=1)
            c_std[:, t] = torch.sqrt(
                (weights * (c - c_hat[:, t].unsqueeze(-1)) ** 2)
                .sum(dim=1)
                .clamp(min=0.0)
            )

        return {
            "collapse_prob": collapse_prob,
            "c_hat": c_hat,
            "c_std": c_std,
            "log_lik": log_lik,
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _truncated_normal(
        shape: tuple,
        *,
        mean: float,
        std: float,
        lower: float | torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Draw a normal, zeros clipped to a lower bound (rejection-free)."""
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        device = generator.device
        if isinstance(lower, torch.Tensor):
            lower = lower.to(device)
        else:
            lower = torch.as_tensor(lower, device=device, dtype=torch.float32)
        x = mean + std * torch.randn(
            shape, generator=generator, dtype=torch.float32, device=device
        )
        return torch.clamp(x, min=lower)

    @staticmethod
    def _systematic_resample(
        weights: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Systematic resampling over the particle dimension.

        Args:
            weights: ``(B, N)`` normalised particle weights.
            generator: RNG for the uniform offset.

        Returns:
            ``(B, N)`` int64 tensor of ancestor indices.
        """
        B, N = weights.shape
        device = weights.device
        u0 = torch.rand(B, 1, generator=generator, device=device, dtype=torch.float32) / N
        offsets = (
            torch.arange(N, device=device, dtype=torch.float32) / N
        ).unsqueeze(0)
        u = u0 + offsets  # (B, N) evenly spaced offsets in (0, 1]
        cumw = torch.cumsum(weights, dim=1).clamp(max=1.0 - 1e-12)
        idx = torch.searchsorted(cumw, u)
        return torch.clamp(idx, max=N - 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SpectralDriftObserver(n_particles={self.n_particles}, "
            f"c_min={self.c_min}, delta={self.delta}, dt={self.dt}, "
            f"params={self.count_parameters()})"
        )
