"""Spectral-drift Bayesian early warning observer (fold-type).

Implements the formal definition in ``docs/z2/formal_defininition.md``:

* the scalar OU reduction of the dominant mode, ``du = -c(t) u dt + sigma_u dW``;
* the exact discrete-time transition
  ``u_{k+1} = exp(-c_k dt) u_k + eps_k`` with
  ``Var(eps_k) = sigma_u^2 / (2 c_k) (1 - exp(-2 c_k dt))``;
* a positive state process ``c_{k+1} = max(c_min, c_k + eta_k)`` with
  ``eta_k ~ N(0, Q_drift)`` (c_min > 0 keeps the transition variance
  well defined);
* a Rao-Blackwellised particle filter: particles over the spectral gap
  ``c_k``, exact conditional scalar Kalman filter over the mode ``u_k``;
* a Shiryaev-type alarm: the collapse probability
  ``p_collapse,k = Pr(c_k < delta | y_{1:k})``.

The reduction is theoretically grounded for fold (saddle-node) bifurcations
only. On Hopf and period-doubling (logistic) systems the observer is applied
empirically; results there should carry that scope caveat.

The observer has no trainable parameters (buffers only), mirroring
``ClassicalKalmanLag2`` in ``csd_observer.models.kalman_lag2``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def extract_mode(features: np.ndarray, system: str) -> np.ndarray:
    """Extract the scalar dominant-mode sequence for a system.

    * ``fold`` / ``logistic``: the single observed channel.
    * ``hopf``: the radial mode ``r = sqrt(x1^2 + x2^2)`` of the Hopf
      normal form ``dr = (mu r - r^3) dt`` -- i.e. the fold-type stability
      coordinate of the Hopf system.

    Args:
        features: ``(B, T, C)`` float array of observed features.
        system: one of ``"fold"``, ``"hopf"``, ``"logistic"``.

    Returns:
        ``(B, T)`` float32 array.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be (B, T, C), got shape {features.shape}")
    if system not in ("fold", "hopf", "logistic"):
        raise ValueError(f"Unknown system: {system!r}. Options: fold, hopf, logistic")
    if system == "hopf":
        if features.shape[-1] < 2:
            raise ValueError(
                f"hopf requires at least 2 channels, got {features.shape[-1]}"
            )
        return np.sqrt(features[..., 0] ** 2 + features[..., 1] ** 2).astype(np.float32)
    if features.shape[-1] < 1:
        raise ValueError(f"system {system!r} requires at least 1 channel")
    return np.asarray(features[..., 0], dtype=np.float32)


def running_mean_center(
    seq: np.ndarray,
    seq_lengths: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """Subtract a causal running mean (centring) from each trajectory.

    Implements the preprocessing step of the formal definition: the OU
    transition is written for a centred mode (``bar_u = 0``), so the
    equilibrium is removed by a running-mean subtraction. The window is
    causal (uses only past samples) so the observer remains online. No
    ground-truth parameter values are used.

    Args:
        seq: ``(B, T)`` float array of mode sequences.
        seq_lengths: ``(B,)`` integer array of valid prefix lengths.
        window: running-mean window size (effective window is
            ``min(window, t+1)`` at each step).

    Returns:
        ``(B, T)`` float32 array, centred per trajectory.
    """
    seq = np.asarray(seq, dtype=np.float32)
    B, T = seq.shape
    seq_lengths = np.asarray(seq_lengths, dtype=np.int64)
    if seq_lengths.shape[0] != B:
        raise ValueError(
            f"seq_lengths length {seq_lengths.shape[0]} != B {B}"
        )
    window = max(1, int(window))
    centered = np.zeros_like(seq)
    cumsum = np.cumsum(seq, axis=1)
    for b in range(B):
        L = int(seq_lengths[b])
        if L <= 0:
            continue
        L = min(L, T)
        W = min(window, L)
        mean = np.zeros(T, dtype=np.float32)
        for t in range(L):
            lo = max(0, t - W + 1)
            hi = t + 1
            mean[t] = (
                cumsum[b, hi - 1] - (cumsum[b, lo - 1] if lo > 0 else 0.0)
            ) / (hi - lo)
        centered[b, :L] = seq[b, :L] - mean[:L]
    return centered


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
    def forward(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the filter over a batch of sequences.

        Args:
            y: ``(B, T)`` float tensor of centred mode observations.

        Returns:
            dict with keys

            * ``collapse_prob`` ``(B, T)``: posterior ``Pr(c_k < delta)``;
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
                if len(flagged) > 0:
                    idx[flagged] = self._systematic_resample(
                        weights[flagged], generator
                    )
                c_pred = torch.gather(c_pred, 1, idx)
                u_corr = torch.gather(u_corr, 1, idx)
                p_corr = torch.gather(p_corr, 1, idx)
                log_w = torch.full(
                    (B, N), -np.log(N), device=device, dtype=torch.float32
                )
                weights = torch.full(
                    (B, N), 1.0 / N, device=device, dtype=torch.float32
                )

            c = c_pred
            u = u_corr
            p_u = p_corr

            # 6. record posterior quantities ------------------------------------
            # collapse_prob[:, t] = Pr(c_{t+1} < delta | y_{0:t})
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


def grid_search_q_drift(
    y_val_signal: np.ndarray,
    y_val_null: np.ndarray,
    bifs_val: np.ndarray,
    lens_signal_val: np.ndarray,
    lens_null_val: np.ndarray,
    *,
    sigma_u: float,
    r: float,
    q_grid: Optional[List[float]] = None,
    n_particles: int = 500,
    c_min: float = 1e-3,
    delta: float = 0.05,
    device: Optional[torch.device] = None,
) -> float:
    """Select ``Q_drift`` by validation early-warning AUC.

    Mirrors ``grid_search_q`` in ``csd_observer.models.kalman_lag2``: run
    the observer for each candidate ``q`` on the validation split and keep
    the one with the highest ``compute_early_warning_auc`` over the collapse
    probabilities.

    Args:
        y_val_signal: ``(B_sig, T)`` centred mode sequences (signal).
        y_val_null: ``(B_null, T)`` centred mode sequences (null).
        bifs_val: ``(B_sig,)`` bifurcation times of the signal validation set.
        lens_signal_val: ``(B_sig,)`` signal sequence lengths.
        lens_null_val: ``(B_null,)`` null sequence lengths.
        sigma_u: process noise scale of the mode.
        r: observation noise variance.
        q_grid: candidate drift-noise values (default: six orders of
            magnitude from ``1e-6`` up to ``1e-1``).
        n_particles, c_min, delta: observer hyperparameters.
        device: torch device.

    Returns:
        The ``q`` value achieving the highest validation EW-AUC.
    """
    _, best_q = grid_search_sigma_u_q_drift(
        y_val_signal,
        y_val_null,
        bifs_val,
        lens_signal_val,
        lens_null_val,
        sigma_u_grid=[sigma_u],
        r=r,
        q_grid=q_grid,
        n_particles=n_particles,
        c_min=c_min,
        delta=delta,
        device=device,
    )
    return best_q


def grid_search_sigma_u_q_drift(
    y_val_signal: np.ndarray,
    y_val_null: np.ndarray,
    bifs_val: np.ndarray,
    lens_signal_val: np.ndarray,
    lens_null_val: np.ndarray,
    *,
    sigma_u_grid: Optional[List[float]] = None,
    r: float,
    q_grid: Optional[List[float]] = None,
    n_particles: int = 500,
    c_min: float = 1e-3,
    delta: float = 0.05,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Select ``(sigma_u, Q_drift)`` jointly by validation early-warning AUC.

    The model's per-step noise scale ``sigma_u`` is a free hyperparameter
    (diagnostics: ``docs/notes/spectral_drift_diagnosis.md``): fixing it to
    ``noise_scale`` mis-specifies the innovation variance by up to ~6x on
    some systems (fold, logistic), which pushes ``c_hat`` down on nulls and
    inflates the fixed-FPR threshold. Running the observer over a grid of
    ``sigma_u`` values (each with the full ``q_grid``) and keeping the pair
    with the highest validation EW-AUC lets the data pick its own noise scale,
    exactly like ``grid_search_q_drift`` does for ``Q_drift``. The observer
    remains non-learned (hyperparameter selection, no training).

    Args:
        y_val_signal: ``(B_sig, T)`` centred mode sequences (signal).
        y_val_null: ``(B_null, T)`` centred mode sequences (null).
        bifs_val: ``(B_sig,)`` bifurcation times of the signal validation set.
        lens_signal_val: ``(B_sig,)`` signal sequence lengths.
        lens_null_val: ``(B_null,)`` null sequence lengths.
        sigma_u_grid: candidate process-noise scales (default
            ``[0.15, 0.3, 0.6, 1.0]``). All entries must be positive.
        r: observation noise variance.
        q_grid: candidate drift-noise values (default: six orders of
            magnitude from ``1e-6`` up to ``1e-1``).
        n_particles, c_min, delta: observer hyperparameters.
        device: torch device.

    Returns:
        The ``(sigma_u, q)`` pair achieving the highest validation EW-AUC.
    """
    if sigma_u_grid is None:
        sigma_u_grid = [0.15, 0.3, 0.6, 1.0]
    if q_grid is None:
        q_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sigma_u_grid = [float(s) for s in sigma_u_grid]
    q_grid = [float(q) for q in q_grid]
    if not sigma_u_grid:
        raise ValueError("sigma_u_grid must contain at least one value")
    if any(s <= 0 for s in sigma_u_grid):
        raise ValueError(f"sigma_u_grid entries must be > 0, got {sigma_u_grid}")
    if not q_grid:
        raise ValueError("q_grid must contain at least one value")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from csd_observer.utils.metrics import compute_early_warning_auc

    y_sig_t = torch.from_numpy(
        np.asarray(y_val_signal, dtype=np.float32)
    ).to(device)
    y_null_t = torch.from_numpy(
        np.asarray(y_val_null, dtype=np.float32)
    ).to(device)
    is_pos_sig = np.ones(y_val_signal.shape[0], dtype=bool)

    best_sigma_u = sigma_u_grid[0]
    best_q = q_grid[0]
    best_auc = -1.0

    for sigma_u in sigma_u_grid:
        for q in q_grid:
            observer = SpectralDriftObserver(
                sigma_u=sigma_u,
                r=r,
                q_drift=q,
                n_particles=n_particles,
                c_min=c_min,
                delta=delta,
            ).to(device)
            observer.eval()
            with torch.no_grad():
                probs_sig = observer(y_sig_t)["collapse_prob"].cpu().numpy()
                probs_null = observer(y_null_t)["collapse_prob"].cpu().numpy()

            auc = compute_early_warning_auc(
                probs_sig,
                bifs_val,
                is_pos_sig,
                lens_signal_val,
                probs_null,
                lens_null_val,
            )
            if np.isfinite(auc) and auc > best_auc:
                best_auc = auc
                best_sigma_u = sigma_u
                best_q = q

    return best_sigma_u, best_q
