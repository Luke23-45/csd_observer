"""Deterministic replicate-level split utilities."""
import numpy as np


def replicate_split(n: int, *, seed: int, train_frac: float = .6, val_frac: float = .2) -> dict[str, np.ndarray]:
    if n < 3 or train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("replicate split requires n >= 3 and valid fractions")
    order = np.arange(n)
    np.random.default_rng(seed).shuffle(order)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_val = n - n_train - 1
        if n_val < 1:
            n_train = n - 2
            n_val = 1
    return {"train": order[:n_train], "val": order[n_train:n_train+n_val], "test": order[n_train+n_val:]}


__all__ = ["replicate_split"]
