"""DaphniaExt processing gates."""
from __future__ import annotations

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode


def validate_bundle(bundle: dict, *, min_length: int = 100) -> None:
    features = np.asarray(bundle.get("features"))
    lengths = np.asarray(bundle.get("seq_lengths"), dtype=np.int64)
    positive = np.asarray(bundle.get("is_positive"))
    if features.ndim != 3 or lengths.shape != (features.shape[0],) or positive.shape != (features.shape[0],):
        raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, "invalid DaphniaExt bundle shapes")
    if not np.isfinite(features).all():
        raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, "DaphniaExt features contain non-finite values")
    if np.any(lengths < min_length):
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, f"trajectory shorter than {min_length} samples")
    if not positive.any() or positive.all():
        raise DatasetError(DatasetErrorCode.SPLIT_IMBALANCE, "DaphniaExt bundle needs both signal and null replicates")


__all__ = ["validate_bundle"]
