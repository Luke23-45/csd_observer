"""Common interface for every benchmark method (§6.1 of the plan).

Every method — statistical indicator or neural baseline — exposes the
same contract so the governance driver
(``evaluation/persistence/governance.py``) treats them identically.

    fit(train_arrays, val_arrays, cfg) -> None   # no-op for non-learned
    score(features, seq_lengths, cfg) -> (B,T) float32
    meta                                   -> MethodMeta

Arrays follow the dataset-registry bundle contract:
``{"features": (B,T,C), "seq_lengths": (B,), "bifurcation_times": (B,),
"is_positive": (B,), "split_indices": {...}}``.

``evaluation`` imports ``models`` nowhere: governance only sees objects
satisfying this protocol structurally (duck-typed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MethodMeta:
    """Static description of one benchmark method."""

    name: str
    family: str  # "indicator" | "neural"
    is_learned: bool
    scope_caveat: str = ""  # e.g. "theoretically grounded for fold; empirical elsewhere"
    bif_types_supported: list[str] = field(default_factory=list)  # empty = any
    default_params: dict[str, Any] = field(default_factory=dict)
    config_path: tuple[str, ...] = ()  # nested key path into config["model"]


@runtime_checkable
class MethodInterface(Protocol):
    """Structural protocol satisfied by every method implementation."""

    @property
    def meta(self) -> MethodMeta: ...

    def fit(
        self,
        train_arrays: dict[str, Any],
        val_arrays: dict[str, Any],
        cfg: dict[str, Any],
    ) -> None:
        """Train (learned methods).

        Non-learned methods: no-op.
        """
        ...

    def score(
        self,
        features: Any,
        seq_lengths: Any,
        cfg: dict[str, Any],
    ) -> Any:
        """Return ``(B, T)`` float32 alarm scores (0..1 range expected)."""
        ...


__all__ = ["MethodMeta", "MethodInterface"]
