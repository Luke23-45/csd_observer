"""Phase EXECUTE (R1): the (seed, method) evaluation loop.

The per-seed bundle is loaded with the §13 schedule seeds, trained
when the method is learned (checkpoint wired back into the config),
and run through the persistence-aware governance driver. Nothing here
touches lifecycle markers — failure handling belongs to the driver.
"""

from __future__ import annotations

from csd_observer.config.validate import seed_schedule
from csd_observer.evaluation.persistence.governance import evaluate_method
from csd_observer.models.common.registry import get_method
from csd_observer.orchestration.run_context import RunContext, load_bundle
from csd_observer.outputs.metadata import collect_environment
from csd_observer.training.common.train import train_method


def execute(ctx: RunContext) -> None:
    """Run every (seed, method) pair; raise on the first failure."""
    writer = ctx.writer
    if writer is None:
        raise RuntimeError("execute() called before prepare()")

    for s in range(ctx.n_seeds):
        per_seed_overrides = dict(ctx.overrides)
        schedule = seed_schedule(ctx.seed_offset, s)
        per_seed_overrides["seed"] = schedule["signal"]
        per_seed_overrides["null_seed"] = schedule["null"]
        bundle = load_bundle(ctx.dataset, per_seed_overrides, schedule["signal"])

        # B3: the dataset's bifurcation type is a property of the dataset,
        # not of the seed — a per-seed flip means the generators are no
        # longer reproducible and the run is invalid.
        if bundle["meta"].get("bif_type") != ctx.meta.get("bif_type"):
            raise RuntimeError(
                f"dataset {ctx.dataset!r} meta.bif_type changed across seeds: "
                f"pre-loop {ctx.meta.get('bif_type')!r} vs seed {s} "
                f"{bundle['meta'].get('bif_type')!r}"
            )

        for name in ctx.methods:
            method = get_method(name, ctx.meta["bif_type"])
            per_method_cfg = train_method(
                method,
                bundle,
                ctx.config,
                ctx.training,
                writer,
                seed_index=s,
                run_seed=schedule["signal"],
            )
            evaluate_method(
                method, bundle["signal"], bundle["null"],
                system=ctx.meta["bif_type"], bif_type=ctx.meta["bif_type"],
                dataset=ctx.dataset, run_name=ctx.dataset, run_id=ctx.run_id,
                timestamp=writer.timestamp, seed=s,
                replicate=f"s{s}",
                config=per_method_cfg, writer=writer,
                k_persist=ctx.k_persist, fpr_target=ctx.fpr_target,
                early_start_delta=ctx.early_start_delta,
                early_end_delta=ctx.early_end_delta,
                evaluation=ctx.evaluation_name,
                git_sha=collect_environment()["git_sha"],
                config_hash=ctx.config_hash,
                store_trajectories=ctx.store_trajectories,
            )


__all__ = ["execute"]
