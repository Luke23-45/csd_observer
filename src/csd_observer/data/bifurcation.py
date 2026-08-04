from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

_DEFAULT_DRIVE_NOISE = {"fold": 0.30, "hopf": 0.05, "logistic": 0.02}
_DEFAULT_OBS_NOISE = {"fold": 0.10, "hopf": 0.15, "logistic": 0.05}

# Ramp targets used by the "hard" generator tier.
_HARD_PARAM_END = {"fold": -1.0, "hopf": 0.5, "logistic": 3.15}

# Per-system drive/colour bounds for the hard tier.  Fold noise must stay
# below the saddle-node barrier except near r -> 0: with sigma_eff too large
# the trajectory flips across the unstable branch while r >> 0 and pins on
# the integration clip, which destroys the CSD signature (an over-driven
# oscillator, not a bifurcation crash).
_HARD_DRIVE_MULT = {"fold": (0.3, 1.0), "hopf": (0.3, 3.0), "logistic": (0.3, 3.0)}
_HARD_COLOR_RANGE = {"fold": (0.0, 0.3), "hopf": (0.0, 0.8), "logistic": (0.0, 0.8)}
_HARD_CROSS_AT = {"fold": 0.0, "hopf": 0.0, "logistic": 3.0}
_HARD_PLATEAU = {"fold": 0.15, "hopf": -0.02, "logistic": 2.95}
_HARD_START = {"fold": (0.3, 3.0), "hopf": (-2.0, -0.4), "logistic": (2.0, 2.9)}


def _split_indices(
    n: int,
    *,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_frac <= 0.0 or val_frac <= 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("Invalid split fractions.")
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _build_return_dict(
    features: np.ndarray,
    true_states: np.ndarray,
    *,
    n_trajectories: int,
    max_length: int,
    bifurcation_time: float | np.ndarray,
    is_null: bool,
    param_name: str,
    param_values: np.ndarray,
    split_seed: int,
) -> Dict[str, np.ndarray]:
    N = n_trajectories
    T = max_length
    bifs = np.asarray(bifurcation_time, dtype=np.float32)
    if bifs.ndim == 0:
        bifs = np.full(N, float(bifs), dtype=np.float32)
    if bifs.shape != (N,):
        raise ValueError(f"bifurcation_time must be a scalar or shape {(N,)}; got {bifs.shape}")
    train_idx, val_idx, test_idx = _split_indices(N, seed=split_seed)
    return {
        "features": features,
        "true_states": true_states,
        "bifurcation_times": bifs,
        "is_positive": np.full(N, not is_null, dtype=np.bool_),
        "seq_lengths": np.full(N, T, dtype=np.int64),
        param_name: param_values,
        "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
    }


class FoldBifurcationDataset:
    def __init__(
        self,
        n_trajectories: int = 500,
        max_length: int = 200,
        r_start: float = 2.0,
        r_end: float = -1.0,
        noise_scale: float = 0.30,
        obs_noise_scale: float = 0.10,
        seed: int = 42,
        null: bool = False,
    ) -> None:
        self.n_trajectories = n_trajectories
        self.max_length = max_length
        self.r_start = r_start
        self.r_end = r_end if not null else r_start
        self.noise_scale = noise_scale
        self.obs_noise_scale = obs_noise_scale
        self.seed = seed
        self.null = null

    @property
    def bifurcation_time(self) -> float:
        if self.null:
            return float(self.max_length + 1)
        return float(self.max_length * (0.0 - self.r_start) / (self.r_end - self.r_start))

    def generate(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        N = self.n_trajectories
        T = self.max_length
        r = np.linspace(self.r_start, self.r_end, T).astype(np.float32)

        x = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            tau_i = self.bifurcation_time
            for _attempt in range(30):
                x_i = rng.normal(0.0, 0.5)
                crash_t: Optional[int] = None
                for t in range(T):
                    # Explicit Euler with dt=1 is unstable on the fold for r>1
                    # (linearized multiplier |1 - 2*sqrt(r)| > 1), so substep the
                    # deterministic drift and inject the noise once per step.
                    for _ in range(10):
                        x_i = np.clip(x_i + (r[t] - x_i ** 2) * 0.1, -5.0, 5.0)
                    x_i = np.clip(x_i + self.noise_scale * rng.normal(0.0, 1.0), -5.0, 5.0)
                    x[i, t] = x_i
                    if crash_t is None and x_i < -1.5:
                        crash_t = t
                if crash_t is None:
                    break
                if not self.null and float(crash_t) >= tau_i - 10.0:
                    break

        y = x + self.obs_noise_scale * rng.normal(0.0, 1.0, size=(N, T)).astype(np.float32)
        y = y[..., None]

        return _build_return_dict(
            y, x[..., None],
            n_trajectories=N, max_length=T,
            bifurcation_time=self.bifurcation_time, is_null=self.null,
            param_name="r_values", param_values=np.tile(r, (N, 1)),
            split_seed=self.seed + 1000,
        )


class HopfBifurcationDataset:
    def __init__(
        self,
        n_trajectories: int = 500,
        max_length: int = 200,
        mu_start: float = -0.5,
        mu_end: float = 0.5,
        omega: float = 0.1,
        noise_scale: float = 0.05,
        obs_noise_scale: float = 0.15,
        seed: int = 42,
        null: bool = False,
    ) -> None:
        self.n_trajectories = n_trajectories
        self.max_length = max_length
        self.mu_start = mu_start
        self.mu_end = mu_end if not null else mu_start
        self.omega = omega
        self.noise_scale = noise_scale
        self.obs_noise_scale = obs_noise_scale
        self.seed = seed
        self.null = null

    @property
    def bifurcation_time(self) -> float:
        if self.null:
            return float(self.max_length + 1)
        return float(self.max_length * (0.0 - self.mu_start) / (self.mu_end - self.mu_start))

    def generate(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        N = self.n_trajectories
        T = self.max_length
        mu = np.linspace(self.mu_start, self.mu_end, T).astype(np.float32)

        r = np.zeros((N, T), dtype=np.float32)
        theta = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            r_i = rng.uniform(0.5, 1.5)
            theta_i = rng.uniform(0.0, 2 * np.pi)
            for t in range(T):
                r_i = r_i + (mu[t] * r_i - r_i ** 3) + self.noise_scale * rng.normal(0.0, 1.0)
                r_i = max(r_i, 0.01)
                theta_i = theta_i + self.omega + self.noise_scale * rng.normal(0.0, 1.0)
                r[i, t] = r_i
                theta[i, t] = theta_i

        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        obs = np.stack([x1, x2], axis=-1).astype(np.float32)
        obs += self.obs_noise_scale * rng.normal(0.0, 1.0, size=obs.shape).astype(np.float32)

        return _build_return_dict(
            obs, np.stack([r, theta], axis=-1),
            n_trajectories=N, max_length=T,
            bifurcation_time=self.bifurcation_time, is_null=self.null,
            param_name="mu_values", param_values=np.tile(mu, (N, 1)),
            split_seed=self.seed + 1000,
        )


class LogisticMapDataset:
    def __init__(
        self,
        n_trajectories: int = 500,
        max_length: int = 200,
        mu_start: float = 2.5,
        mu_end: float = 4.0,
        noise_scale: float = 0.02,
        obs_noise_scale: float = 0.05,
        seed: int = 42,
        null: bool = False,
    ) -> None:
        self.n_trajectories = n_trajectories
        self.max_length = max_length
        self.mu_start = mu_start
        self.mu_end = mu_end if not null else 2.8
        self.noise_scale = noise_scale
        self.obs_noise_scale = obs_noise_scale
        self.seed = seed
        self.null = null

    @property
    def bifurcation_time(self) -> float:
        if self.null:
            return float(self.max_length + 1)
        mu_bif = 3.0
        return float(self.max_length * (mu_bif - self.mu_start) / (self.mu_end - self.mu_start))

    def generate(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        N = self.n_trajectories
        T = self.max_length
        mu = np.linspace(self.mu_start, self.mu_end, T).astype(np.float32)

        x = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            x_i = rng.uniform(0.1, 0.9)
            for t in range(T):
                x_i = mu[t] * x_i * (1.0 - x_i) + self.noise_scale * rng.normal(0.0, 1.0)
                x_i = np.clip(x_i, 0.0, 1.0)
                x[i, t] = x_i

        y = x[..., None] + self.obs_noise_scale * rng.normal(0.0, 1.0, size=(N, T, 1)).astype(np.float32)

        return _build_return_dict(
            y, x[..., None],
            n_trajectories=N, max_length=T,
            bifurcation_time=self.bifurcation_time, is_null=self.null,
            param_name="mu_values", param_values=np.tile(mu, (N, 1)),
            split_seed=self.seed + 1000,
        )


def _ar1_noise(
    rng: np.random.Generator,
    n: int,
    phi: float,
    sigma: float,
) -> np.ndarray:
    """AR(1) red-noise sequence with marginal standard deviation ``sigma``."""
    xi = rng.normal(0.0, sigma * np.sqrt(max(0.0, 1.0 - phi * phi)), size=n)
    e = np.empty(n, dtype=np.float64)
    e[0] = xi[0]
    for t in range(1, n):
        e[t] = phi * e[t - 1] + xi[t]
    return e.astype(np.float32)


class RandomizedBifurcationDataset:
    """Bury-style randomized normal-form dataset (Bury et al., 2023, Nat. Commun.).

    Follows the randomized-model recipe used to train deep early-warning
    signals at scale. Unlike :class:`FoldBifurcationDataset` and friends,
    every trajectory is drawn from a different model and noise regime:

    - ``bifurcation_time`` sampled per trajectory within ``tau_frac_range``,
      so runs start both near and far from the critical point while sharing
      the same ramp end value;
    - the normal form is corrupted by a random polynomial perturbation of
      degree ``1..10`` (coefficients ~ ``N(0, 1)`` decayed geometrically at
      rate ``perturbation_scale``);
    - the driving noise amplitude is drawn log-uniformly per trajectory
      around ``noise_scale``, and the noise is coloured (AR(1), per-trajectory
      persistence from ``color_range``);
    - the observation noise is drawn log-uniformly per trajectory around
      ``obs_noise_scale``.
    """

    def __init__(
        self,
        system: str,
        n_trajectories: int = 500,
        max_length: int = 200,
        noise_scale: float | None = None,
        obs_noise_scale: float | None = None,
        perturbation_scale: float = 0.3,
        tau_frac_range: Tuple[float, float] | None = None,
        noise_mult_range: Tuple[float, float] = (0.5, 2.0),
        color_range: Tuple[float, float] = (0.0, 0.6),
        logistic_mu_end: float = 3.2,
        difficulty: str = "standard",
        seed: int = 42,
        null: bool = False,
    ) -> None:
        if system not in _DEFAULT_DRIVE_NOISE:
            raise ValueError(f"Unknown system: {system!r}. Valid options: {sorted(_DEFAULT_DRIVE_NOISE)}")
        if difficulty not in {"standard", "hard"}:
            raise ValueError(f"Unknown difficulty: {difficulty!r}. Valid options: ['standard', 'hard']")
        self.system = system
        self.n_trajectories = n_trajectories
        self.max_length = max_length
        self.noise_scale = _DEFAULT_DRIVE_NOISE[system] if noise_scale is None else noise_scale
        self.obs_noise_scale = _DEFAULT_OBS_NOISE[system] if obs_noise_scale is None else obs_noise_scale
        self.perturbation_scale = perturbation_scale
        if tau_frac_range is None:
            tau_frac_range = (0.2, 0.5) if system == "logistic" else (0.2, 0.75)
        self.tau_frac_range = tau_frac_range
        self.noise_mult_range = noise_mult_range
        self.color_range = color_range
        self.logistic_mu_end = logistic_mu_end
        self.difficulty = difficulty
        self.seed = seed
        self.null = null

    @staticmethod
    def _polynomial(
        x: float,
        scaled_c: np.ndarray,
        k_min: int,
    ) -> float:
        p = 0.0
        for c, k in zip(scaled_c, range(k_min, k_min + len(scaled_c))):
            p += c * (x ** k)
        return p

    def generate(self) -> Dict[str, np.ndarray]:
        if self.difficulty == "hard":
            return self._generate_hard()
        return self._generate_standard()

    def _generate_standard(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        N = self.n_trajectories
        T = self.max_length
        system = self.system
        tfrac = np.arange(T, dtype=np.float64) / float(T - 1) if T > 1 else np.zeros(1)

        if self.null:
            tau = np.full(N, float(T + 1), dtype=np.float32)
        else:
            tau = (rng.uniform(*self.tau_frac_range, size=N) * (T - 1)).astype(np.float32)
        tau_frac = tau / float(T - 1)

        if system == "fold":
            noise_mult = np.exp(rng.uniform(np.log(0.5), np.log(0.9), size=N))
            color = rng.uniform(0.0, 0.3, size=N)
        else:
            noise_mult = np.exp(rng.uniform(*np.log(self.noise_mult_range), size=N))
            color = rng.uniform(*self.color_range, size=N)
        obs_mult = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=N))
        degree = rng.integers(2, 11, size=N)
        pscale = self.perturbation_scale * np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=N))

        n_channels = 2 if system == "hopf" else 1
        features = np.zeros((N, T, n_channels), dtype=np.float32)
        true_states = np.zeros((N, T, n_channels), dtype=np.float32)
        param_values = np.zeros((N, T), dtype=np.float32)
        omega = 0.1

        for i in range(N):
            if self.null:
                tf_i = float(rng.uniform(*self.tau_frac_range))
            else:
                tf_i = float(tau_frac[i])
            if system == "fold":
                start = tf_i / (1.0 - tf_i)
                ramp = (start + (-1.0 - start) * tfrac).astype(np.float32) if not self.null else np.full(T, start, dtype=np.float32)
            elif system == "hopf":
                start = -0.8 * tf_i / (1.0 - tf_i)
                ramp = (start + (0.8 - start) * tfrac).astype(np.float32) if not self.null else np.full(T, start, dtype=np.float32)
            else:
                mu_end = self.logistic_mu_end
                start = (3.0 - mu_end * tf_i) / (1.0 - tf_i)
                ramp = (start + (mu_end - start) * tfrac).astype(np.float32) if not self.null else np.full(T, start, dtype=np.float32)

            d = int(degree[i])
            coeffs = rng.normal(0.0, 1.0, size=d)
            s = float(pscale[i])
            k_min = 1
            scaled_c = coeffs * (s ** np.arange(k_min, k_min + d))

            sig = self.noise_scale * float(noise_mult[i])
            phi = float(color[i])
            e = _ar1_noise(rng, T, phi, sig)
            wn = rng.normal(0.0, 1.0, size=T)
            obs = self.obs_noise_scale * float(obs_mult[i])

            if system == "fold":
                # Reject trajectories whose noise-driven crash precedes the
                # labelled crossing: a fold that collapses while r >> 0 is a
                # mislabelled early bifurcation, not a saddle-node transition.
                tau_i = float(T + 1) if self.null else float(tau[i])
                for _attempt in range(30):
                    coeffs = rng.normal(0.0, 1.0, size=d)
                    scaled_c = coeffs * (s ** np.arange(k_min, k_min + d))
                    e = _ar1_noise(rng, T, phi, sig)
                    wn = rng.normal(0.0, 1.0, size=T)
                    x = float(np.sqrt(max(start, 1e-3)) * max(0.0, 1.0 + 0.2 * float(rng.normal(0.0, 1.0))))
                    crash_t: Optional[int] = None
                    for t in range(T):
                        # Substep the deterministic drift: explicit Euler with
                        # dt=1 is unstable for the fold when r>1.
                        for _ in range(10):
                            x = float(np.clip(
                                x + (ramp[t] - x * x + self._polynomial(x, scaled_c, k_min)) * 0.1,
                                -5.0, 5.0,
                            ))
                        x = float(np.clip(x + e[t], -5.0, 5.0))
                        true_states[i, t, 0] = x
                        features[i, t, 0] = x + obs * float(wn[t])
                        if crash_t is None and x < -1.5:
                            crash_t = t
                    if crash_t is None:
                        break
                    if not self.null and float(crash_t) >= tau_i - 10.0:
                        break
            elif system == "hopf":
                r = float(rng.uniform(0.5, 1.5))
                th = float(rng.uniform(0.0, 2.0 * np.pi))
                for t in range(T):
                    drift = ramp[t] * r - r ** 3 + self._polynomial(r, scaled_c, k_min)
                    r = float(np.clip(r + drift + e[t], 0.01, 3.0))
                    th = th + omega + sig * float(wn[t])
                    true_states[i, t, 0] = r
                    true_states[i, t, 1] = th
                    features[i, t, 0] = r * np.cos(th) + obs * rng.normal(0.0, 1.0)
                    features[i, t, 1] = r * np.sin(th) + obs * rng.normal(0.0, 1.0)
            else:
                x = float(rng.uniform(0.1, 0.9))
                for t in range(T):
                    x = float(np.clip(ramp[t] * x * (1.0 - x) + self._polynomial(x, scaled_c, k_min) + e[t], 0.0, 1.0))
                    true_states[i, t, 0] = x
                    features[i, t, 0] = x + obs * float(wn[t])

            param_values[i] = ramp

        param_name = "r_values" if system == "fold" else "mu_values"

        return _build_return_dict(
            features, true_states,
            n_trajectories=N, max_length=T,
            bifurcation_time=tau,
            is_null=self.null,
            param_name=param_name, param_values=param_values,
            split_seed=self.seed + 1000,
        )

    @staticmethod
    def _find_cross(ramp: np.ndarray, threshold: float) -> float:
        """First threshold crossing time of a monotonic ramp (linear interp).
        Returns ``T + 1`` when the ramp never crosses."""
        T = len(ramp)
        for t in range(T - 1):
            a = float(ramp[t])
            b = float(ramp[t + 1])
            if (a - threshold) * (b - threshold) <= 0.0:
                if abs(b - a) < 1e-12:
                    return float(t + 1)
                return float(t + (threshold - a) / (b - a))
        return float(T + 1)

    def _hard_ramp(
        self,
        system: str,
        kind: str,
        tfrac: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        """Parameter ramp for the hard tier.

        ``kind``:
          - ``"stat"``: constant parameter (stationary null),
          - ``"co"``:   slow approach to a plateau below the critical value
                        (co-moving null; never crosses),
          - ``"bif"``:  linear ramp across the critical (standard crossing),
          - ``"rate"``: sigmoidal ramp concentrated near the end (rate-type
                        approach; crosses but with little advance structure).

        Returns ``(ramp, tau)`` with ``tau = T + 1`` when no crossing occurs.
        """
        p0_min, p0_max = _HARD_START[system]
        start = float(rng.uniform(p0_min, p0_max))
        end = _HARD_PARAM_END[system]
        if kind == "stat":
            ramp = np.full(tfrac.shape, start, dtype=np.float64)
            return ramp.astype(np.float32), float(self.max_length + 1)
        if kind == "co":
            plateau = _HARD_PLATEAU[system]
            tau_c = 0.45
            ramp = start + (plateau - start) * (1.0 - np.exp(-tfrac / max(tau_c, 1e-6)))
            return ramp.astype(np.float32), float(self.max_length + 1)
        if kind == "rate":
            steep = float(rng.uniform(15.0, 30.0))
            t0 = float(rng.uniform(0.55, 0.75))
            sig = 1.0 / (1.0 + np.exp(-steep * (tfrac - t0)))
            ramp = start + (end - start) * sig
        else:
            ramp = start + (end - start) * tfrac
        tau_i = self._find_cross(ramp, _HARD_CROSS_AT[system])
        return ramp.astype(np.float32), tau_i

    def _generate_hard(self) -> Dict[str, np.ndarray]:
        """Hard tier: co-moving nulls (never cross), rate-concentrated
        crossings, and a wider noise/SNR range per trajectory."""
        rng = np.random.default_rng(self.seed)
        N = self.n_trajectories
        T = self.max_length
        system = self.system
        tfrac = np.arange(T, dtype=np.float64) / float(T - 1) if T > 1 else np.zeros(1)

        drive_mult = np.exp(rng.uniform(*np.log(_HARD_DRIVE_MULT[system]), size=N))
        color = rng.uniform(*_HARD_COLOR_RANGE[system], size=N)
        obs_mult = np.exp(rng.uniform(np.log(0.2), np.log(2.5), size=N))
        degree = rng.integers(2, 11, size=N)
        pscale = self.perturbation_scale * np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=N))

        n_channels = 2 if system == "hopf" else 1
        features = np.zeros((N, T, n_channels), dtype=np.float32)
        true_states = np.zeros((N, T, n_channels), dtype=np.float32)
        param_values = np.zeros((N, T), dtype=np.float32)
        tau = np.zeros(N, dtype=np.float32)
        omega = 0.1

        co_null_frac = 0.75
        rate_frac = 0.5

        for i in range(N):
            d = int(degree[i])
            coeffs = rng.normal(0.0, 1.0, size=d)
            s = float(pscale[i])
            scaled_c = coeffs * (s ** np.arange(1, d + 1))

            sig = self.noise_scale * float(drive_mult[i])
            phi = float(color[i])
            e = _ar1_noise(rng, T, phi, sig)
            wn = rng.normal(0.0, 1.0, size=T)
            obs = self.obs_noise_scale * float(obs_mult[i])

            if self.null:
                kind = "co" if float(rng.uniform(0.0, 1.0)) < co_null_frac else "stat"
                ramp, _ = self._hard_ramp(system, kind, tfrac, rng)
                tau[i] = float(T + 1)
            else:
                kind = "rate" if float(rng.uniform(0.0, 1.0)) < rate_frac else "bif"
                ramp, tau_i = self._hard_ramp(system, kind, tfrac, rng)
                tau[i] = float(tau_i)

            if system == "fold":
                # Reject noise-driven crashes that precede the labelled
                # crossing (see _generate_standard for the rationale).
                for _attempt in range(30):
                    coeffs = rng.normal(0.0, 1.0, size=d)
                    scaled_c = coeffs * (s ** np.arange(1, d + 1))
                    e = _ar1_noise(rng, T, phi, sig)
                    wn = rng.normal(0.0, 1.0, size=T)
                    x = float(np.sqrt(max(float(ramp[0]), 1e-3)) * max(0.0, 1.0 + 0.2 * float(rng.normal(0.0, 1.0))))
                    crash_t: Optional[int] = None
                    for t in range(T):
                        for _ in range(10):
                            x = float(np.clip(
                                x + (ramp[t] - x * x + self._polynomial(x, scaled_c, 1)) * 0.1,
                                -5.0, 5.0,
                            ))
                        x = float(np.clip(x + e[t], -5.0, 5.0))
                        true_states[i, t, 0] = x
                        features[i, t, 0] = x + obs * float(wn[t])
                        if crash_t is None and x < -1.5:
                            crash_t = t
                    if crash_t is None:
                        break
                    if not self.null and float(crash_t) >= float(tau[i]) - 10.0:
                        break
            elif system == "hopf":
                r = float(rng.uniform(0.5, 1.5))
                th = float(rng.uniform(0.0, 2.0 * np.pi))
                for t in range(T):
                    drift = ramp[t] * r - r ** 3 + self._polynomial(r, scaled_c, 1)
                    r = float(np.clip(r + drift + e[t], 0.01, 3.0))
                    th = th + omega + sig * float(wn[t])
                    true_states[i, t, 0] = r
                    true_states[i, t, 1] = th
                    features[i, t, 0] = r * np.cos(th) + obs * rng.normal(0.0, 1.0)
                    features[i, t, 1] = r * np.sin(th) + obs * rng.normal(0.0, 1.0)
            else:
                x = float(rng.uniform(0.1, 0.9))
                for t in range(T):
                    x = float(np.clip(ramp[t] * x * (1.0 - x) + self._polynomial(x, scaled_c, 1) + e[t], 0.0, 1.0))
                    true_states[i, t, 0] = x
                    features[i, t, 0] = x + obs * float(wn[t])

            param_values[i] = ramp

        param_name = "r_values" if system == "fold" else "mu_values"

        return _build_return_dict(
            features, true_states,
            n_trajectories=N, max_length=T,
            bifurcation_time=tau,
            is_null=self.null,
            param_name=param_name, param_values=param_values,
            split_seed=self.seed + 1000,
        )


def build_dataset(
    system: str,
    *,
    n_trajectories: int = 500,
    max_length: int = 200,
    noise_scale: float = 0.30,
    obs_noise_scale: float | None = None,
    seed: int = 42,
    null: bool = False,
    generator: str = "classic",
    difficulty: str = "standard",
) -> Dict[str, np.ndarray]:
    _VALID_SYSTEMS = {"fold", "hopf", "logistic"}
    if system not in _VALID_SYSTEMS:
        raise ValueError(
            f"Unknown system: {system!r}. Valid options: {sorted(_VALID_SYSTEMS)}"
        )
    kwargs = dict(n_trajectories=n_trajectories, max_length=max_length, seed=seed, null=null)
    if obs_noise_scale is not None:
        kwargs["obs_noise_scale"] = obs_noise_scale
    if generator == "classic":
        if system == "fold":
            return FoldBifurcationDataset(noise_scale=noise_scale, **kwargs).generate()
        elif system == "hopf":
            return HopfBifurcationDataset(noise_scale=noise_scale, **kwargs).generate()
        return LogisticMapDataset(noise_scale=noise_scale, **kwargs).generate()
    elif generator == "bury":
        return RandomizedBifurcationDataset(
            system, noise_scale=noise_scale, difficulty=difficulty, **kwargs,
        ).generate()
    raise ValueError(f"Unknown generator: {generator!r}. Valid options: ['classic', 'bury']")
