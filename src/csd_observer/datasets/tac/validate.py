"""TAC processing gates independent of the parser implementation."""
from __future__ import annotations

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode


def validate_bundle(bundle: dict, *, min_length: int = 100) -> None:
    features = np.asarray(bundle.get("features"))
    lengths = np.asarray(bundle.get("seq_lengths"), dtype=np.int64)
    if features.ndim != 3 or lengths.shape != (features.shape[0],):
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, "TAC bundle must contain features (B,T,C) and seq_lengths (B,)")
    if not np.isfinite(features).all():
        raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, "TAC features contain non-finite values")
    if np.any(lengths < min_length):
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, f"TAC trajectory shorter than {min_length} samples")


__all__ = ["validate_bundle"]
