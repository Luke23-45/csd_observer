"""Phase PREPARE (R1): provision, mint the output tree, record provenance.

Runs before any scoring work and before any output bytes exist:

1. provision real datasets (ingest → process → gates → manifest);
   synthetic datasets short-circuit (in-memory fast path);
2. mint the run's :class:`OutputWriter` (one timestamp);
3. write resolved config, CLI overrides, environment (with the
   dataset-manifest hashes, R4);
4. open the run log and register it with the ingest layer so manual
   download instructions land in ``logs/run.log``;
5. load the pre-loop bundle (s=0 signal seed) for the dataset ``meta``
   (``bif_type``) and record the ledger row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from csd_observer.datasets.common.ingest import set_run_log
from csd_observer.orchestration.run_context import RunContext, load_bundle, preloop_seed
from csd_observer.outputs.ledger import LedgerRow, RunLedger
from csd_observer.outputs.metadata import collect_environment, hash_config
from csd_observer.outputs.writer import OutputWriter


def prepare(ctx: RunContext) -> None:
    """Populate every prepare-time field of ``ctx`` (idempotent per run)."""
    if ctx.writer is not None:
        raise RuntimeError("prepare() already ran for this context")

    # ---- real datasets must be provisioned before anything else ----
    if str(ctx.dataset_cfg.get("source", "synthetic")) != "synthetic":
        from csd_observer.datasets.provision import provision_dataset

        provision_dataset(
            ctx.dataset,
            ctx.dataset_cfg,
            root=ctx.overrides.get("data_root", "datasets"),
        )

    # ---- pre-loop bundle load: dataset meta + manifest hashes ----
    bundle = load_bundle(ctx.dataset, ctx.overrides, preloop_seed(ctx.seed_offset))
    ctx.meta = bundle["meta"]
    ctx.dataset_manifest_hashes = _manifest_hashes(
        ctx.dataset, ctx.dataset_cfg, ctx.overrides
    )

    # ---- output tree + provenance ----
    writer = OutputWriter(ctx.dataset, ctx.base_dir)
    ctx.writer = writer
    resolved = dict(ctx.config)
    resolved.setdefault("dataset_overrides", {})
    ctx.config_hash = hash_config(resolved)
    writer.write_resolved_config(resolved)
    writer.write_cli_overrides(list(ctx.cli_overrides))
    writer.write_environment(
        collect_environment(
            config_hash=ctx.config_hash,
            dataset_manifest_hashes=dict(ctx.dataset_manifest_hashes),
        )
    )

    # ---- run log shared by ingest + training callbacks ----
    set_run_log(writer.open_log())

    # ---- ledger ----
    ctx.ledger = RunLedger(Path(ctx.base_dir) / "_ledger")
    ctx.run_id = f"{ctx.dataset}-{writer.timestamp}"
    ctx.ledger.append(
        LedgerRow(
            run_id=ctx.run_id, timestamp=writer.timestamp, run_name=ctx.dataset,
            dataset=ctx.dataset, methods=list(ctx.methods), k_persist=ctx.k_persist,
            config_hash=ctx.config_hash, git_sha=collect_environment()["git_sha"],
            status="pending", path=str(writer.root),
            bif_types=[ctx.meta["bif_type"]],
            is_learned_methods=list(ctx.learned),
        )
    )


def _manifest_hashes(
    dataset: str,
    dataset_cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, str]:
    """Dataset provenance hashes for the run environment (R4).

    Real datasets: the processed manifest's ``content_hash`` (the
    canonical binding of array bytes + params). Synthetic: a hash over
    the seed-independent generator provenance (the dataset group block
    keys; per-seed schedule values are deliberately excluded so the
    hash identifies the dataset, not the seed).
    """
    from csd_observer.datasets.common.pipeline import data_dir

    folder = data_dir(dataset)
    manifest_path = (
        Path(str(overrides.get("data_root", "datasets")))
        / "processed" / folder / "manifest.json"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "dataset": dataset,
            "kind": "content_hash",
            "value": str(manifest.get("content_hash", "")),
            "schema_version": str(manifest.get("schema_version", "")),
        }
    provenance = {
        key: dataset_cfg.get(key)
        for key in ("n_trajectories", "max_length", "noise_scale",
                    "obs_noise_scale", "generator", "difficulty")
        if key in dataset_cfg
    }
    return {
        "dataset": dataset,
        "kind": "provenance_hash",
        "value": hash_config(provenance),
    }


__all__ = ["prepare"]
