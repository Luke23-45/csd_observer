"""Synthetic fold system wrapper (registry contract §5.6)."""

from __future__ import annotations

from typing import Any

from csd_observer.datasets.synthetic.common import build_bundle

_SYSTEM = "fold"


def build_bundle_fold(**overrides: Any) -> dict[str, Any]:
    """Synthetic saddle-node (fold) bundle; see ``synthetic.common.build_bundle``."""
    return build_bundle(_SYSTEM, **overrides)


def build_signal_null(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Signal + null bundle pair plus ``meta`` (registry ``get_dataset`` shape).

    §13 seed schedule: the caller may pin ``null_seed`` explicitly
    (``seed_offset + s*1000 + 202``); otherwise null uses ``seed + 1``.
    """
    overrides = dict(overrides or {})
    seed = int(overrides.get("seed", 42))
    null_seed = int(overrides.pop("null_seed", seed + 1))
    rest = {k: v for k, v in overrides.items() if k != "seed"}
    signal = build_bundle_fold(seed=seed, null=False, **rest)
    null = build_bundle_fold(seed=null_seed, null=True, **rest)
    return {"signal": signal, "null": null, "meta": signal["meta"]}


__all__ = ["build_bundle_fold", "build_signal_null", "_SYSTEM"]
