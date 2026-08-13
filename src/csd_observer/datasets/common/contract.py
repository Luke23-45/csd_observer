"""Bundle-shape contract checked on every dataset load path (R2.1).

Both the synthetic fast path and the processed-real load path must
return bundles that satisfy one identical shape contract, so that a
method can never silently receive a malformed array (wrong dtype,
imbalanced split, corrupted manifest) in a production run. Validation
is cheap (single pass, no copies) and runs once per seed loop at
prepare time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode

_MIN_SPLIT_PER_CLASS = 1


def validate_bundle(
    bundle: dict[str, Any],
    context: str = "dataset",
) -> None:
    """Validate the shape contract of a signal-or-null subset bundle.

    Contract (documented in the plan §5.4, enforced here verbatim):

    * ``features``: ``(B, T, C)`` float array, finite, ``C >= 1``,
      ``T >= 1``;
    * ``seq_lengths``: ``(B,)`` integer array with ``1 <= l <= T``;
    * ``bifurcation_times``: ``(B,)`` finite float array;
    * ``is_positive``: ``(B,)`` bool array;
    * ``split_indices``: dict with exactly ``train``/``val``/``test``
      integer arrays, disjoint, covering ``[0, B)`` (or a single
      ``"all"`` key for the legacy full-bundle path).

    Raises :class:`DatasetError` with code ``MANIFEST_CORRUPT`` or
    ``SPLIT_IMBALANCE``; never returns.
    """
    features = bundle.get("features")
    if features is None or not isinstance(features, np.ndarray):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: features missing or not ndarray")
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[2] < 1:
        raise DatasetError(
            DatasetErrorCode.MANIFEST_CORRUPT,
            f"{context}: features must be (B, T, C) with B>=1, C>=1, got {features.shape}",
        )
    if features.dtype.kind not in "fc":
        raise DatasetError(
            DatasetErrorCode.MANIFEST_CORRUPT,
            f"{context}: features must be float/complex, got dtype {features.dtype}",
        )
    if not np.isfinite(features).all():
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: features contain NaN/Inf")

    B, T, C = features.shape
    seq_lengths = bundle.get("seq_lengths")
    if seq_lengths is None or np.asarray(seq_lengths).ndim != 1 or len(seq_lengths) != B:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: seq_lengths must be (B,)")
    lengths = np.asarray(seq_lengths)
    if lengths.dtype.kind not in "iu" or not np.all((lengths >= 1) & (lengths <= T)):
        raise DatasetError(
            DatasetErrorCode.MANIFEST_CORRUPT,
            f"{context}: seq_lengths must be ints with 1 <= l <= T={T}",
        )

    bifurcation_times = bundle.get("bifurcation_times")
    if bifurcation_times is None or np.asarray(bifurcation_times).ndim != 1 or len(bifurcation_times) != B:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: bifurcation_times must be (B,)")
    if not np.isfinite(np.asarray(bifurcation_times, dtype=np.float64)).all():
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: bifurcation_times contain NaN/Inf")

    is_positive = bundle.get("is_positive")
    if is_positive is None or np.asarray(is_positive).ndim != 1 or len(is_positive) != B:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: is_positive must be (B,) bool")
    if np.asarray(is_positive).dtype.kind != "b":
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: is_positive must be bool")

    _validate_split(bundle.get("split_indices"), B, context)


def _validate_split(split_indices: Any, n: int, context: str) -> None:
    """Split indices must be disjoint and cover every trajectory."""
    if not isinstance(split_indices, dict):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: split_indices must be a dict")
    keys = set(split_indices)
    if keys == {"all"}:
        arr = np.asarray(split_indices["all"])
        if arr.ndim == 1 and len(arr) == n and np.array_equal(np.sort(arr), np.arange(n)):
            return
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: split_indices['all'] must cover [0, n)")
    if keys != {"train", "val", "test"}:
        raise DatasetError(
            DatasetErrorCode.MANIFEST_CORRUPT,
            f"{context}: split_indices must have exactly train/val/test keys, got {sorted(keys)}",
        )
    seen = np.zeros(n, dtype=bool)
    for part in ("train", "val", "test"):
        arr = np.asarray(split_indices[part], dtype=np.int64)
        if arr.ndim != 1 or np.any(arr < 0) or np.any(arr >= n) or np.any(seen[arr]):
            raise DatasetError(
                DatasetErrorCode.MANIFEST_CORRUPT,
                f"{context}: split_indices[{part!r}] invalid or overlapping",
            )
        seen[arr] = True
    if not seen.all():
        raise DatasetError(
            DatasetErrorCode.SPLIT_IMBALANCE,
            f"{context}: split_indices do not cover all {n} trajectories",
        )


__all__ = ["validate_bundle"]
