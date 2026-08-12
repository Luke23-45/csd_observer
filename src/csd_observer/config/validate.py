"""Fail-fast validation for the composed protocol configuration (§10.3).

Called by the CLI right after Hydra composition, before any I/O or
training work: unknown registry names, disabled training for learned
methods, out-of-range evaluation knobs, non-replicate-based splits on
real data, and off-schedule seeds all raise here.

Method names are resolved through the model registry (the single source
of truth per §6.1); dataset names through the dataset registry.

Minimum dataset-dimension gates (``n_trajectories >= 3``,
``max_length >= 100``) are bypassed by setting the environment variable
``CSD_OBSERVER_SKIP_MIN_LENGTH_GATES=1``; this is intended for CI/smoke
runs only and is documented in AGENTS.md. Production runs should leave
the gates enabled.
"""

from __future__ import annotations

import os
from typing import Any

from csd_observer.models.common.registry import list_families, validate_names
from csd_observer.models.common.systems import SUPPORTED_SYSTEMS

# §10.2: dataset overrides are whitelisted keys only (registry kwargs
# that change generation or provenance; ``data_root`` switches where
# processed real datasets are read from).
DATASET_OVERRIDE_WHITELIST = {
    "n_trajectories",
    "max_length",
    "noise_scale",
    "obs_noise_scale",
    "seed",
    "null_seed",
    "generator",
    "difficulty",
    "data_root",
}

_LEARNED_FAMILIES = ("neural",)

_SYNTHETIC = {"synthetic_fold", "synthetic_hopf", "synthetic_logistic"}


def _resolve_methods(config: dict[str, Any]) -> list[str]:
    models = config.get("models", [])
    if isinstance(models, str):
        models = [models]
    return [str(m) for m in (models or [])]


def validate_config(config: dict[str, Any]) -> None:
    """Enforce every §10.3 invariant; raises ``ValueError`` on the first hit."""
    dataset = config.get("dataset", {})
    if isinstance(dataset, dict):
        dataset_name = str(dataset.get("name", ""))
    else:
        dataset_name = str(dataset)
    if not dataset_name:
        raise ValueError("dataset is required")

    # ---- §10.3: dataset in registry; split policy for real data ----
    from csd_observer.datasets.registry import list_datasets

    available = list_datasets()
    if dataset_name not in available:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; valid: {sorted(available)}"
        )
    split = config.get("split", {}) or {}
    if isinstance(dataset, dict):
        split = dataset.get("split", split)
    if dataset_name not in _SYNTHETIC and not split.get("replicate_based", False):
        raise ValueError("real datasets require split.replicate_based=true")

    # ---- §5.6: dataset bif_type must be in the method-side system set ----
    bif_type = dataset.get("bif_type") if isinstance(dataset, dict) else None
    if bif_type is not None and str(bif_type) not in SUPPORTED_SYSTEMS:
        raise ValueError(
            f"dataset {dataset_name!r} declares unsupported bif_type "
            f"{bif_type!r}; supported: {', '.join(SUPPORTED_SYSTEMS)}"
        )

    # ---- §10.3: every model in registry ----
    methods = _resolve_methods(config)
    if not methods:
        raise ValueError("models must list at least one method")
    validate_names(methods)

    # ---- §10.3: model.is_learned ⇒ training != none ----
    families = list_families()
    training = config.get("training", {})
    if isinstance(training, str):
        training_enabled = training != "none"
    else:
        training_enabled = bool((training or {}).get("enabled", True))
    learned = any(families.get(m, "") in _LEARNED_FAMILIES for m in methods)
    if learned and not training_enabled:
        raise ValueError(
            f"learned method(s) {[m for m in methods if families.get(m) in _LEARNED_FAMILIES]} "
            "require a training configuration (training.enabled=true)"
        )

    # ---- §10.3: evaluation knobs ----
    evaluation = config.get("evaluation", {})
    if isinstance(evaluation, str):
        evaluation = {"name": evaluation}
    if int(evaluation.get("k_persist", 5)) < 1:
        raise ValueError("evaluation.k_persist must be >= 1")
    fpr = float(evaluation.get("fpr_target", 0.05))
    if not 0.0 < fpr < 1.0:
        raise ValueError("evaluation.fpr_target must be in (0, 1)")

    # ---- §10.3: training epochs ----
    if training_enabled:
        epochs = int((training or {}).get("epochs", 50))
        if epochs < 1:
            raise ValueError("training.epochs must be >= 1")

    # ---- §10.2: dataset overrides whitelist ----
    overrides = config.get("dataset_overrides", {}) or {}
    unknown = sorted(set(overrides) - DATASET_OVERRIDE_WHITELIST)
    if unknown:
        raise ValueError(
            f"dataset_overrides keys not in whitelist {sorted(DATASET_OVERRIDE_WHITELIST)}: {unknown}"
        )

    # ---- §13: seed schedule ----
    seed_offset = int(config.get("seed_offset", 0))
    n_seeds = int(config.get("n_seeds", 1))
    if seed_offset < 0 or n_seeds < 1:
        raise ValueError("seed_offset must be >= 0 and n_seeds >= 1")
    # ``seed_schedule`` checks int32 overflow per (s, split) itself.
    for s in range(n_seeds):
        seed_schedule(seed_offset, s)

    # ---- §5.4 minimum dataset dimensions ----
    # replicate_split requires n >= 3, DFA requires max_length >= 100.
    # The MIN_LENGTH gate is bypassed by setting
    # ``CSD_OBSERVER_SKIP_MIN_LENGTH_GATES=1`` so CI/smoke runs can use
    # tiny synthetic data; the warning makes the bypass explicit.
    skip_min_gates = os.environ.get("CSD_OBSERVER_SKIP_MIN_LENGTH_GATES", "").lower() in {"1", "true", "yes"}
    overrides = config.get("dataset_overrides", {}) or {}
    n_traj = overrides.get("n_trajectories", config.get("n_trajectories"))
    if n_traj is not None and not skip_min_gates:
        try:
            n_traj_i = int(n_traj)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"n_trajectories must be an integer, got {n_traj!r}") from exc
        if n_traj_i < 3:
            raise ValueError(
                f"n_trajectories must be >= 3 (replicate_split requires at least train+val+test), "
                f"got {n_traj_i}"
            )
    max_len = overrides.get("max_length", config.get("max_length"))
    if max_len is not None and not skip_min_gates:
        try:
            max_len_i = int(max_len)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_length must be an integer, got {max_len!r}") from exc
        if max_len_i < 100:
            raise ValueError(
                f"max_length must be >= 100 (DFA gate §5.4 MIN_LENGTH), got {max_len_i}"
            )


def seed_schedule(
    seed_offset: int,
    s: int,
) -> dict[str, int]:
    """§13 per-seed schedule: ``seed_offset + s*1000 + 101/202``.

    ``s`` is the 0-based seed index within a run's ``n_seeds`` loop.
    Raises :class:`ValueError` if the resulting run seed overflows
    signed int32 (numpy's default integer dtype is 32-bit on Windows;
    a 64-bit int silently down-casts and breaks per-seed reproducibility).
    """
    base = int(seed_offset) + int(s) * 1000
    out: dict[str, int] = {}
    _max_run_seed = 2**31 - 1
    for split_name, delta in (("signal", 101), ("null", 202)):
        run_seed = base + delta
        if run_seed < 0:
            raise ValueError(
                f"seed schedule produces negative run seed for seed s={s} ({split_name})"
            )
        if run_seed > _max_run_seed:
            raise ValueError(
                f"seed schedule overflows int32 for seed s={s} ({split_name}): "
                f"run_seed={run_seed} > {_max_run_seed}; reduce seed_offset or n_seeds"
            )
        out[split_name] = run_seed
    return out


__all__ = [
    "DATASET_OVERRIDE_WHITELIST",
    "seed_schedule",
    "validate_config",
]
