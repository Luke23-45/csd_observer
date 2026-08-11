"""Fail-fast validation for the composed protocol configuration (§10.3).

Called by the CLI right after Hydra composition, before any I/O or
training work: unknown registry names, disabled training for learned
methods, out-of-range evaluation knobs, non-replicate-based splits on
real data, and off-schedule seeds all raise here.

Method names are resolved through the model registry (the single source
of truth per §6.1); dataset names through the dataset registry.
"""

from __future__ import annotations

from typing import Any

from csd_observer.models.common.registry import list_families, validate_names

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
    for s in range(n_seeds):
        base = seed_offset + s * 1000
        for split_name, delta in (("signal", 101), ("null", 202)):
            run_seed = base + delta
            if run_seed < 0:
                raise ValueError(
                    f"seed schedule produces negative run seed for seed s={s} ({split_name})"
                )


def seed_schedule(
    seed_offset: int,
    s: int,
) -> dict[str, int]:
    """§13 per-seed schedule: ``seed_offset + s*1000 + 101/202``.

    ``s`` is the 0-based seed index within a run's ``n_seeds`` loop.
    """
    base = int(seed_offset) + int(s) * 1000
    return {"signal": base + 101, "null": base + 202}


__all__ = [
    "DATASET_OVERRIDE_WHITELIST",
    "seed_schedule",
    "validate_config",
]
