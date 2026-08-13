"""Thin orchestration driver for one governed benchmark run (§11, R1).

The runner contains the pipeline order only — prepare → execute →
finalize — plus the failure lifecycle. Everything else is delegated:

* ``phase_prepare.prepare`` — provision, output tree, provenance;
* ``phase_execute.execute`` — (seed, method) loop, training, scoring;
* ``phase_finalize.finalize`` — summarize + completion markers;
* dataset generation / method training / scoring live in their own
  layers (datasets, training, models, evaluation) and are never
  reimplemented here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from csd_observer.models.common.registry import list_families
from csd_observer.orchestration.phase_execute import execute
from csd_observer.orchestration.phase_finalize import finalize
from csd_observer.orchestration.phase_prepare import prepare
from csd_observer.orchestration.run_context import RunContext
from csd_observer.outputs.writer import OutputWriter


def run_benchmark(
    config: dict[str, Any],
    *,
    cli_overrides: Sequence[str] = (),
) -> OutputWriter:
    """Run every (seed, method) pair of a composed, validated config.

    Args:
        config: full composed config dict (see ``config/store.py``):
            ``dataset`` (group block), ``models`` (registry display
            names), ``seed_offset``/``n_seeds`` (§13 schedule),
            ``evaluation``, ``training``, ``output``,
            ``dataset_overrides`` (whitelisted keys only).
        cli_overrides: raw Hydra override strings, recorded verbatim.

    Returns:
        the run's :class:`OutputWriter` (output tree already finalized).
    """
    ctx = _build_context(config, cli_overrides)
    prepare(ctx)
    try:
        execute(ctx)
    except BaseException as exc:
        # Partial rows written before the failure are intentional: a
        # multi-method run that succeeds for VAR-CSD and fails on
        # LSTM-AlarmNet leaves VAR-CSD's row on disk so the partial
        # result is still analysable. Downstream consumers must filter
        # by ``status`` in the ledger / lifecycle marker.
        writer = ctx.writer
        if writer is not None:
            writer.mark_failed(exc)
        if ctx.ledger is not None:
            ctx.ledger.update_status(ctx.run_id, "failed")
        raise
    finalize(ctx)
    writer = ctx.writer
    assert writer is not None
    return writer


def _build_context(
    config: dict[str, Any],
    cli_overrides: Sequence[str],
) -> RunContext:
    """Resolve the composed config into a :class:`RunContext`."""
    dataset_cfg = dict(config.get("dataset", {}) or {})
    dataset = str(dataset_cfg.get("name", ""))
    if not dataset:
        raise ValueError("config.dataset.name is required")
    methods = [str(m) for m in config.get("models", []) or []]
    if not methods:
        raise ValueError("config.models must list at least one method")

    evaluation = dict(config.get("evaluation", {}) or {})
    training = config.get("training", {}) or {}
    # ``training="none"`` is the CLI shortcut for the ``training=none``
    # group; the validator treats the string as ``enabled=false``. The
    # runner must respect the same semantic so a string here is
    # normalised to a dict with ``enabled=False`` rather than coerced
    # via ``dict(str)``.
    if isinstance(training, str):
        training = {"enabled": training != "none"}
    training = dict(training or {})
    output = dict(config.get("output", {}) or {})
    base_dir = str(output.get("base_dir", "outputs"))
    evaluation_name = str(evaluation.get("name", "persistenceaware"))
    store_trajectories = bool(output.get("store_trajectories", False))
    seed_offset = int(config.get("seed_offset", 0))
    n_seeds = int(config.get("n_seeds", 1))
    overrides = dict(config.get("dataset_overrides", {}) or {})

    families = list_families()
    learned = [m for m in methods if families.get(m) == "neural"]

    return RunContext(
        config=config,
        cli_overrides=list(cli_overrides),
        dataset_cfg=dataset_cfg,
        dataset=dataset,
        methods=methods,
        evaluation=evaluation,
        evaluation_name=evaluation_name,
        k_persist=int(evaluation.get("k_persist", 5)),
        fpr_target=float(evaluation.get("fpr_target", 0.05)),
        early_start_delta=float(evaluation.get("early_start_delta", 50.0)),
        early_end_delta=float(evaluation.get("early_end_delta", 5.0)),
        training=training,
        base_dir=base_dir,
        store_trajectories=store_trajectories,
        overrides=overrides,
        seed_offset=seed_offset,
        n_seeds=n_seeds,
        families=families,
        learned=learned,
    )


__all__ = ["run_benchmark"]
