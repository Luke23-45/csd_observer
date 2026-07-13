from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _stratified_split(
    rng: np.random.Generator,
    is_positive: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Dict[str, np.ndarray]:
    pos_idx = np.where(is_positive)[0]
    neg_idx = np.where(~is_positive)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def _split_group(idx: np.ndarray) -> Tuple[int, int]:
        n_tot = len(idx)
        n_train = max(1, int(round(n_tot * train_frac)))
        n_val = max(1, int(round(n_tot * val_frac)))
        n_val = min(n_val, n_tot - n_train - 1)
        return n_train, n_val

    n_train_p, n_val_p = _split_group(pos_idx)
    n_train_n, n_val_n = _split_group(neg_idx)

    train_idx = np.concatenate([pos_idx[:n_train_p], neg_idx[:n_train_n]])
    val_idx = np.concatenate([pos_idx[n_train_p:n_train_p + n_val_p], neg_idx[n_train_n:n_train_n + n_val_n]])
    test_idx = np.concatenate([pos_idx[n_train_p + n_val_p:], neg_idx[n_train_n + n_val_n:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def prepare(
    df: pd.DataFrame,
    transitions: Dict[Tuple[int, str], int],
    *,
    max_length: int = 861,
    split_seed: int = 1042,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    out_dir: str | Path,
) -> Dict[str, np.ndarray]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys_sorted = sorted(transitions.keys(), key=lambda k: (k[1], k[0]))
    N = len(keys_sorted)

    features = np.zeros((N, max_length, 1), dtype=np.float32)
    seq_lengths = np.zeros(N, dtype=np.int64)
    bifurcation_times = np.full(N, -1.0, dtype=np.float32)
    is_positive = np.zeros(N, dtype=np.bool_)

    for i, (tsid, typ) in enumerate(keys_sorted):
        mask = (df["tsid"] == tsid) & (df["type"] == typ)
        group = df[mask].sort_values("Beat number")
        ibi = group["IBI (s)"].values.astype(np.float32)
        L = len(ibi)
        mu, std = ibi.mean(), ibi.std()
        ibi_norm = (ibi - mu) / (std + 1e-8) if std > 1e-8 else np.zeros_like(ibi)

        features[i, :L, 0] = ibi_norm
        seq_lengths[i] = L
        is_positive[i] = (typ == "pd")

        transition = transitions[(tsid, typ)]
        if transition >= 0:
            bifurcation_times[i] = float(transition)
        elif typ == "pd":
            bifurcation_times[i] = float(L)
        else:
            bifurcation_times[i] = -1.0

    rng = np.random.default_rng(split_seed)
    split_indices = _stratified_split(rng, is_positive, train_frac, val_frac)

    np.savez_compressed(
        str(out_dir / "dataset.npz"),
        features=features,
        seq_lengths=seq_lengths,
        bifurcation_times=bifurcation_times,
        is_positive=is_positive,
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
    )

    return {
        "features": features,
        "seq_lengths": seq_lengths,
        "bifurcation_times": bifurcation_times,
        "is_positive": is_positive,
        "split_indices": split_indices,
    }
