from typing import Dict, Tuple

import numpy as np

from .load import build_chick_heart_dataset


def build_benchmark_datasets(
    force_download: bool = False,
    force_transitions: bool = False,
    force_prepare: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    combined = build_chick_heart_dataset(
        force_download=force_download,
        force_transitions=force_transitions,
        force_prepare=force_prepare,
    )
    is_pos = combined["is_positive"]
    neg_idx = np.where(~is_pos)[0]

    def _subset(global_idx: np.ndarray) -> Dict[str, np.ndarray]:
        old_splits = combined["split_indices"]
        new_splits: Dict[str, np.ndarray] = {}
        for name in ("train", "val", "test"):
            mask = np.isin(global_idx, old_splits[name])
            new_splits[name] = np.where(mask)[0]
        return {
            "features": combined["features"][global_idx],
            "seq_lengths": combined["seq_lengths"][global_idx],
            "bifurcation_times": combined["bifurcation_times"][global_idx],
            "is_positive": combined["is_positive"][global_idx],
            "split_indices": new_splits,
        }

    arrays_signal = {
        "features": combined["features"],
        "seq_lengths": combined["seq_lengths"],
        "bifurcation_times": combined["bifurcation_times"],
        "is_positive": combined["is_positive"],
        "split_indices": combined["split_indices"],
    }
    arrays_null = _subset(neg_idx)
    return arrays_signal, arrays_null


__all__ = ["build_chick_heart_dataset", "build_benchmark_datasets"]
