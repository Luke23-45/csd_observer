"""Canonical row schema for run results.

Every entry in ``results.jsonl`` (and every aggregator input) must match
the row dataclass below. Validators reject missing, wrong-typed, or
unknown keys so downstream summarizers never see a malformed row.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

_REQUIRED_KEYS = (
    "run_id",
    "timestamp",
    "run_name",
    "dataset",
    "bif_type",
    "system",
    "replicate",
    "method",
    "family",
    "is_learned",
    "k_persist",
    "fpr_target",
    "detection_rate",
    "detection_time_mean",
    "detection_time_median",
    "detection_time_std",
    "ew_auc",
    "fpr",
    "persistent_fpr",
    "threshold",
    "params",
    "config_hash",
    "git_sha",
    "seed",
)

_NUMERIC_FIELDS = (
    "k_persist",
    "fpr_target",
    "detection_rate",
    "detection_time_mean",
    "detection_time_median",
    "detection_time_std",
    "ew_auc",
    "fpr",
    "persistent_fpr",
    "threshold",
)

_INT_FIELDS = ("k_persist", "seed")
_STR_FIELDS = (
    "run_id",
    "timestamp",
    "run_name",
    "dataset",
    "bif_type",
    "system",
    "replicate",
    "method",
    "family",
    "config_hash",
    "git_sha",
)
_BOOL_FIELDS = ("is_learned",)
_DICT_FIELDS = ("params",)


@dataclass
class ResultRow:
    """One result row = one (system/replicate, method, seed) evaluation."""

    run_id: str
    timestamp: str
    run_name: str
    dataset: str
    bif_type: str
    system: str
    replicate: str
    method: str
    family: str
    is_learned: bool
    seed: int
    k_persist: int
    fpr_target: float
    detection_rate: float
    detection_time_mean: float
    detection_time_median: float
    detection_time_std: float
    ew_auc: float
    fpr: float
    persistent_fpr: float
    threshold: float
    params: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    git_sha: str = ""
    detection_time_95_low: float = float("nan")
    detection_time_95_high: float = float("nan")
    achieved_fpr_target: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out


class SchemaError(ValueError):
    """Raised when a row fails schema validation."""


def validate_row(row: dict[str, Any]) -> None:
    """Strict validator. Raises SchemaError on any deviation.

    Rules:
      * all required keys present
      * known types per field
      * numeric fields finite-or-nan (no infs except where explicitly allowed)
      * unknown keys rejected (extra allowed only inside ``params`` / ``extra``)
    """
    if not isinstance(row, dict):
        raise SchemaError(f"Row must be a dict, got {type(row).__name__}")
    missing = [k for k in _REQUIRED_KEYS if k not in row]
    if missing:
        raise SchemaError(f"Row missing required keys: {missing}")
    known = set(_REQUIRED_KEYS) | {"detection_time_95_low", "detection_time_95_high", "achieved_fpr_target", "extra"}
    unknown = set(row) - known
    if unknown:
        raise SchemaError(f"Row has unknown keys: {sorted(unknown)}")
    for k in _INT_FIELDS:
        v = row[k]
        if not isinstance(v, int) or isinstance(v, bool):
            raise SchemaError(f"Row[{k!r}] must be int, got {type(v).__name__}")
    for k in _STR_FIELDS:
        if not isinstance(row[k], str):
            raise SchemaError(f"Row[{k!r}] must be str, got {type(row[k]).__name__}")
    for k in _BOOL_FIELDS:
        if not isinstance(row[k], bool):
            raise SchemaError(f"Row[{k!r}] must be bool, got {type(row[k]).__name__}")
    for k in _DICT_FIELDS:
        if not isinstance(row[k], dict):
            raise SchemaError(f"Row[{k!r}] must be dict, got {type(row[k]).__name__}")
    for k in _NUMERIC_FIELDS:
        v = row[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SchemaError(f"Row[{k!r}] must be numeric, got {type(v).__name__}")
        if isinstance(v, float) and math.isinf(v):
            raise SchemaError(f"Row[{k!r}] must be finite or NaN, got inf")
    for k in ("detection_time_95_low", "detection_time_95_high", "achieved_fpr_target"):
        v = row.get(k, float("nan"))
        if not isinstance(v, (int, float)):
            raise SchemaError(f"Row[{k!r}] must be numeric, got {type(v).__name__}")
    if row["k_persist"] < 1:
        raise SchemaError(f"Row['k_persist'] must be >= 1, got {row['k_persist']}")
    if not (0.0 < row["fpr_target"] < 1.0):
        raise SchemaError(f"Row['fpr_target'] must be in (0,1), got {row['fpr_target']}")


__all__ = ["ResultRow", "SchemaError", "validate_row"]
