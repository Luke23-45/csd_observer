from __future__ import annotations

from typing import Dict

import numpy as np


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


class FoldBifurcationDataset:
    """Dynamical system nearing a fold bifurcation: dx/dt = r(t) - x^2 + noise.

    The parameter r decreases linearly from r_start (>0) to r_end (<0),
    crossing the bifurcation point r=0 at a known time.

    Observation: y = x + observation_noise.

    Config
    ------
    n_trajectories : int
        Number of trajectories (default 500).
    max_length : int
        Maximum sequence length (default 200).
    r_start : float
        Initial bifurcation parameter (default 2.0).
    r_end : float
        Final bifurcation parameter (default -1.0).
    noise_scale : float
        Diffusion noise standard deviation (default 0.30).
    obs_noise_scale : float
        Observation noise standard deviation (default 0.10).
    seed : int
        Random seed (default 42).
    null : bool
        If True, r stays constant and no bifurcation occurs (default False).
    """

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
        """Time index at which r(t) = 0 (the fold bifurcation)."""
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
            x_i = rng.normal(0.0, 0.5)
            for t in range(T):
                x_i = x_i + (r[t] - x_i ** 2) + self.noise_scale * rng.normal(0.0, 1.0)
                x_i = np.clip(x_i, -5.0, 5.0)
                x[i, t] = x_i

        y = x + self.obs_noise_scale * rng.normal(0.0, 1.0, size=(N, T)).astype(np.float32)
        y = y[..., None]

        bifurcation_times = np.full(N, self.bifurcation_time, dtype=np.float32)
        seq_lengths = np.full(N, T, dtype=np.int64)
        is_positive = np.full(N, not self.null, dtype=np.bool_)

        split_seed = self.seed + 1000
        train_idx, val_idx, test_idx = _split_indices(N, seed=split_seed)

        return {
            "features": y,
            "true_states": x[..., None],
            "bifurcation_times": bifurcation_times,
            "is_positive": is_positive,
            "seq_lengths": seq_lengths,
            "r_values": np.tile(r, (N, 1)),
            "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
        }


class HopfBifurcationDataset:
    """Dynamical system nearing a Hopf bifurcation.

    In polar coordinates:
        dr/dt = mu(t) * r - r^3 + noise_r
        dtheta/dt = omega + noise_theta

    mu increases from mu_start (<0) to mu_end (>0), crossing 0 at a known time.

    Observation: [r cos(theta), r sin(theta)] + observation_noise.

    Config
    ------
    n_trajectories : int
    max_length : int
    mu_start : float
        Initial bifurcation parameter (default -0.5).
    mu_end : float
        Final bifurcation parameter (default 0.5).
    omega : float
        Angular frequency (default 0.1).
    noise_scale : float
        Diffusion noise standard deviation (default 0.05).
    obs_noise_scale : float
        Observation noise standard deviation (default 0.15).
    seed : int
    null : bool
        If True, mu stays negative and no bifurcation occurs (default False).
    """

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

        bifurcation_times = np.full(N, self.bifurcation_time, dtype=np.float32)
        seq_lengths = np.full(N, T, dtype=np.int64)
        is_positive = np.full(N, not self.null, dtype=np.bool_)

        split_seed = self.seed + 1000
        train_idx, val_idx, test_idx = _split_indices(N, seed=split_seed)

        return {
            "features": obs,
            "true_states": np.stack([r, theta], axis=-1),
            "bifurcation_times": bifurcation_times,
            "is_positive": is_positive,
            "seq_lengths": seq_lengths,
            "mu_values": np.tile(mu, (N, 1)),
            "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
        }


class LogisticMapDataset:
    """Period-doubling route to chaos via the logistic map.

        x_{t+1} = mu(t) * x_t * (1 - x_t) + noise

    mu increases from mu_start (~2.5) to mu_end (~4.0).
    First period-doubling at mu=3.0 (this is the labeled bifurcation point).

    Config
    ------
    n_trajectories : int
    max_length : int
    mu_start : float
    mu_end : float
    noise_scale : float
        Additive noise on each map iteration (default 0.02).
    obs_noise_scale : float
        Observation noise (default 0.05).
    seed : int
    null : bool
        If True, mu stays below 3.0 and no bifurcation occurs (default False).
    """

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

        bifurcation_times = np.full(N, self.bifurcation_time, dtype=np.float32)
        seq_lengths = np.full(N, T, dtype=np.int64)
        is_positive = np.full(N, not self.null, dtype=np.bool_)

        split_seed = self.seed + 1000
        train_idx, val_idx, test_idx = _split_indices(N, seed=split_seed)

        return {
            "features": y,
            "true_states": x[..., None],
            "bifurcation_times": bifurcation_times,
            "is_positive": is_positive,
            "seq_lengths": seq_lengths,
            "mu_values": np.tile(mu, (N, 1)),
            "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
        }


def build_dataset(
    system: str,
    *,
    n_trajectories: int = 500,
    max_length: int = 200,
    noise_scale: float = 0.30,
    seed: int = 42,
    null: bool = False,
) -> Dict[str, np.ndarray]:
    """Factory: build a bifurcation dataset by system name."""
    kwargs = dict(n_trajectories=n_trajectories, max_length=max_length, seed=seed, null=null)
    if system == "fold":
        return FoldBifurcationDataset(noise_scale=noise_scale, **kwargs).generate()
    elif system == "hopf":
        return HopfBifurcationDataset(noise_scale=noise_scale, **kwargs).generate()
    elif system == "logistic":
        return LogisticMapDataset(noise_scale=noise_scale, **kwargs).generate()
    else:
        raise ValueError(f"Unknown system: {system!r}")
