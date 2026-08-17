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

from csd_observer.config.store import (
    EvaluationConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    dataset_group_default,
    dataset_node,
)
from csd_observer.models.common.registry import list_families, validate_names
from csd_observer.models.common.systems import SUPPORTED_SYSTEMS

# §10.2: dataset overrides are whitelisted keys only (registry kwargs
# that change generation or provenance; ``data_root`` switches where
# processed real datasets are read from). ``null_seed`` was removed
# (R0.5): the runner injects the null-generator seed internally per the
# §13 schedule, so an override could silently desynchronize it.
DATASET_OVERRIDE_WHITELIST = {
    "n_trajectories",
    "max_length",
    "noise_scale",
    "obs_noise_scale",
    "seed",
    "generator",
    "difficulty",
    "data_root",
}

#: Generator knobs that must not be set on the ``dataset`` group itself
#: (R0.1): setting e.g. ``dataset.n_trajectories=16`` composes fine but
#: the synthetic generator never sees it — the only channel into the
#: generator is ``+dataset_overrides.*``. Comparing against the
#: registered node defaults catches the silent divergence at validate
#: time instead of producing 500-trajectory runs.
_GENERATOR_KNOBS = (
    "n_trajectories",
    "max_length",
    "noise_scale",
    "obs_noise_scale",
    "seed",
    "generator",
    "difficulty",
)

_LEARNED_FAMILIES = ("neural",)

_SYNTHETIC = {"synthetic_fold", "synthetic_hopf", "synthetic_logistic"}


def _resolve_methods(config: dict[str, Any]) -> list[str]:
    models = config.get("models", [])
    if isinstance(models, str):
        models = [models]
    return [str(m) for m in (models or [])]


def _skip_min_gates() -> bool:
    return os.environ.get("CSD_OBSERVER_SKIP_MIN_LENGTH_GATES", "").lower() in {"1", "true", "yes"}


def _effective_min_length(
    dataset_name: str,
    dataset: dict[str, Any],
    overrides: dict[str, Any],
    processing: dict[str, Any] | None,
) -> int | None:
    """§5.4 gate-2 floor: ``max_length`` for synthetic data (overridable),
    ``processing.min_length`` for processed real data."""
    if dataset_name in _SYNTHETIC:
        value = overrides.get("max_length", dataset.get("max_length"))
        return None if value is None else int(value)
    return None if processing is None else int(processing.get("min_length", 100))


def _check_block_schema(block: str, value: dict[str, Any], node: Any) -> None:
    """Fail-fast conformance of a composed group block to its schema node.

    The group values live in ``configs/<group>/<name>.yaml`` (pinned to
    the store.py nodes by ``tests/config/test_yaml_alignment.py``);
    merging the block into the structured node rejects unknown keys
    (``ConfigKeyError``), wrong types (``ValidationError``) and unknown
    nested keys (structured children) — the strictness the ConfigStore
    used to provide at compose time, which Hydra no longer applies to
    plain yaml configs beyond unknown top-level keys.
    """
    from omegaconf import OmegaConf
    from omegaconf.errors import (
        ConfigKeyError,
        InterpolationToMissingValueError,
        UnsupportedValueType,
        ValidationError,
    )

    try:
        OmegaConf.merge(OmegaConf.structured(node), OmegaConf.create(value))
    except (
        ConfigKeyError,
        InterpolationToMissingValueError,
        UnsupportedValueType,
        ValidationError,
    ) as exc:
        raise ValueError(
            f"{block} group block does not conform to its "
            f"{type(node).__name__} schema: {exc}"
        ) from exc


def validate_config(config: dict[str, Any]) -> None:
    """Enforce every §10.3 invariant; raises ``ValueError`` on the first hit."""
    dataset = config.get("dataset", {})
    if isinstance(dataset, dict):
        dataset_name = str(dataset.get("name", ""))
    else:
        dataset_name = str(dataset)
    if not dataset_name:
        raise ValueError("dataset is required")

    # ---- group blocks conform to the store.py schema ----
    # Registered datasets are checked against their exact node; the other
    # groups against their (all-optional) schema node.
    if isinstance(dataset, dict):
        schema = dataset_node(dataset_name)
        if schema is not None:
            _check_block_schema("dataset", dataset, schema)
    model_block = config.get("model")
    if isinstance(model_block, dict):
        _check_block_schema("model", model_block, ModelConfig())
    training_block = config.get("training")
    if isinstance(training_block, dict):
        _check_block_schema("training", training_block, TrainingConfig())
    evaluation_block = config.get("evaluation")
    if isinstance(evaluation_block, dict):
        _check_block_schema("evaluation", evaluation_block, EvaluationConfig())
    output_block = config.get("output")
    if isinstance(output_block, dict):
        _check_block_schema("output", output_block, OutputConfig())

    # ---- §10.3: dataset in registry; split policy for real data ----
    from csd_observer.datasets.registry import list_resolvable

    available = list_resolvable()
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

    # ---- R0.1: generator knobs on the dataset group are a silent no-op ----
    # The check compares only keys that are *present* in the composed
    # block: yaml-composed configs carry the node's non-None fields,
    # while hand-built minimal dicts (tests) omit the knobs entirely.
    # Unregistered (materialized) datasets have no node defaults, so the
    # comparison is skipped there.
    if isinstance(dataset, dict):
        registered = dataset_group_default(dataset_name)
        if registered:
            misplaced = [
                k for k in _GENERATOR_KNOBS
                if k in dataset and dataset.get(k) != registered.get(k)
            ]
            if misplaced:
                raise ValueError(
                    f"generator knob(s) {sorted(misplaced)} set on the dataset group "
                    f"are ignored by the synthetic generator; set them via "
                    f"+dataset_overrides.{misplaced[0]} instead"
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
    learned_methods = [m for m in methods if families.get(m, "") in _LEARNED_FAMILIES]
    if learned_methods and not training_enabled:
        model_block = config.get("model", {})
        missing_ckpts = []
        for m in learned_methods:
            from csd_observer.models.common.registry import get_method
            method_obj = get_method(m)
            key = method_obj.meta.config_path[0]
            if not isinstance(model_block, dict) or not model_block.get(key, {}).get("checkpoint"):
                missing_ckpts.append(m)
        if missing_ckpts:
            raise ValueError(
                f"learned method(s) {missing_ckpts} require a training configuration "
                "(training.enabled=true) or a pre-trained checkpoint (model.<key>.checkpoint=<path>)"
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
    early_start_delta = float(evaluation.get("early_start_delta", 50.0))
    early_end_delta = float(evaluation.get("early_end_delta", 5.0))
    if early_start_delta <= 0.0 or early_end_delta < 0.0:
        raise ValueError("evaluation.early_start_delta must be > 0 and early_end_delta >= 0")

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
    skip_min_gates = _skip_min_gates()
    dataset_block = dataset if isinstance(dataset, dict) else {}
    processing = dataset_block.get("processing")
    if dataset_name in _SYNTHETIC:
        n_traj = overrides.get("n_trajectories", dataset_block.get("n_trajectories"))
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
    effective_min_length = _effective_min_length(dataset_name, dataset_block, overrides, processing)

    # ---- R2.4: label_window vs the effective length (learned only) ----
    # ``label_window`` marks the last ``label_window`` steps before tau as
    # the alarm target; a window longer than the shortest trajectory marks
    # the whole prefix positive and saturates the supervision signal. This
    # is a supervision invariant, so it fires even when the MIN_LENGTH gate
    # is bypassed: real datasets with ``min_length < 100`` (daphnia) would
    # otherwise silently train with a saturated label window. Indicators
    # never consume ``label_window``, so only learned-method runs check it.
    if learned_methods and training_enabled:
        label_window = (training or {}).get("label_window") if isinstance(training, dict) else None
        if label_window is not None and effective_min_length is not None and int(label_window) > effective_min_length:
            raise ValueError(
                f"training.label_window {label_window} exceeds min dataset length "
                f"{effective_min_length} (label_window longer than the trajectory "
                f"saturates the alarm signal); set a dataset-appropriate "
                f"training.label_window (e.g. <= {effective_min_length})"
            )

    # ---- R2.4: TAC aligned ramp windows must host the early-window ----
    # With chunking enabled each ramp trace is one window aligned to its
    # detected onset at ``ramp_pre_samples``; the evaluation early-window
    # ``[tau - early_start_delta, tau - early_end_delta)`` must therefore
    # fit inside it. Defaults (50/5 vs 2048/2048) pass; oversized deltas
    # would silently clip every early-warning window to nothing.
    if dataset_name == "tac" and isinstance(dataset_block, dict):
        proc = dataset_block.get("processing") or {}
        if proc.get("analysis_window") is not None:
            pre = proc.get("ramp_pre_samples")
            post = proc.get("ramp_post_samples")
            if pre is not None and (early_start_delta > float(pre) or early_end_delta > float(post or 0)):
                raise ValueError(
                    f"TAC aligned ramp windows (ramp_pre_samples={pre}, "
                    f"ramp_post_samples={post}) cannot host the early-warning "
                    f"window (early_start_delta={early_start_delta}, "
                    f"early_end_delta={early_end_delta}); reduce the deltas or "
                    f"raise ramp_pre_samples/ramp_post_samples"
                )

    if effective_min_length is not None and not skip_min_gates:
        if effective_min_length < 100:
            raise ValueError(
                f"min dataset length must be >= 100 (DFA gate §5.4 MIN_LENGTH), "
                f"got {effective_min_length}"
            )

        # ---- R2.4: window/floor invariants against the effective length ----
        model_block = config.get("model", {})
        if isinstance(model_block, dict):
            dfa_window = model_block.get("dfa_csd", {}).get("window_size")
            if dfa_window is not None and int(dfa_window) > effective_min_length:
                raise ValueError(
                    f"model.dfa_csd.window_size {dfa_window} exceeds min dataset length "
                    f"{effective_min_length} (DFA needs window <= series length)"
                )
            patch_len = model_block.get("patchtst", {}).get("patch_len")
            if patch_len is not None and int(patch_len) > effective_min_length:
                raise ValueError(
                    f"model.patchtst.patch_len {patch_len} exceeds min dataset length "
                    f"{effective_min_length} (patch_len must fit inside a trajectory)"
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
