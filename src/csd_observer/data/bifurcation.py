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


def _build_return_dict(
    features: np.ndarray,
    true_states: np.ndarray,
    *,
    n_trajectories: int,
    max_length: int,
    bifurcation_time: float,
    is_null: bool,
    param_name: str,
    param_values: np.ndarray,
    split_seed: int,
) -> Dict[str, np.ndarray]:
    N = n_trajectories
    T = max_length
    train_idx, val_idx, test_idx = _split_indices(N, seed=split_seed)
    return {
        "features": features,
        "true_states": true_states,
        "bifurcation_times": np.full(N, bifurcation_time, dtype=np.float32),
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
            x_i = rng.normal(0.0, 0.5)
            for t in range(T):
                x_i = x_i + (r[t] - x_i ** 2) + self.noise_scale * rng.normal(0.0, 1.0)
                x_i = np.clip(x_i, -5.0, 5.0)
                x[i, t] = x_i

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


def build_dataset(
    system: str,
    *,
    n_trajectories: int = 500,
    max_length: int = 200,
    noise_scale: float = 0.30,
    obs_noise_scale: float | None = None,
    seed: int = 42,
    null: bool = False,
) -> Dict[str, np.ndarray]:
    _VALID_SYSTEMS = {"fold", "hopf", "logistic"}
    if system not in _VALID_SYSTEMS:
        raise ValueError(
            f"Unknown system: {system!r}. Valid options: {sorted(_VALID_SYSTEMS)}"
        )
    kwargs = dict(n_trajectories=n_trajectories, max_length=max_length, seed=seed, null=null)
    if obs_noise_scale is not None:
        kwargs["obs_noise_scale"] = obs_noise_scale
    if system == "fold":
        return FoldBifurcationDataset(noise_scale=noise_scale, **kwargs).generate()
    elif system == "hopf":
        return HopfBifurcationDataset(noise_scale=noise_scale, **kwargs).generate()
    else:
        return LogisticMapDataset(noise_scale=noise_scale, **kwargs).generate()
