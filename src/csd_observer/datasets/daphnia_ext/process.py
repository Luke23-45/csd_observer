"""DaphniaExt table processing with explicit column/treatment mappings."""
from __future__ import annotations

from typing import Any

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode


def process_table(rows: list[dict[str, Any]], *, replicate_column: str, time_column: str,
                  count_column: str, positive_column: str, min_length: int = 100) -> dict[str, np.ndarray]:
    """Convert already parsed rows; no treatment labels are inferred."""
    required = {replicate_column, time_column, count_column, positive_column}
    if not rows or not required.issubset(rows[0]):
        raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, f"missing explicit columns: {sorted(required)}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[replicate_column]), []).append(row)
    features: list[np.ndarray] = []
    positive: list[bool] = []
    for replicate, group in sorted(groups.items()):
        group = sorted(group, key=lambda r: float(r[time_column]))
        if len(group) < min_length:
            raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, f"replicate {replicate} has {len(group)} rows")
        try:
            values = np.asarray([float(r[count_column]) for r in group], dtype=np.float32)
            label_values = {str(r[positive_column]).strip().lower() for r in group}
        except (TypeError, ValueError) as exc:
            raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, str(exc)) from exc
        if len(label_values) != 1 or not label_values <= {"true", "false", "1", "0", "yes", "no"}:
            raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, f"replicate {replicate} has inconsistent treatment labels")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"replicate {replicate} has invalid counts")
        features.append(values[:, None])
        positive.append(label_values.pop() in {"true", "1", "yes"})
    length = {len(x) for x in features}
    if len(length) != 1:
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, "replicates must have equal aligned lengths")
    return {"features": np.stack(features), "seq_lengths": np.full(len(features), length.pop(), dtype=np.int64),
            "is_positive": np.asarray(positive, dtype=bool)}


__all__ = ["process_table"]
