"""RunContext — the immutable per-run context shared by the phases (R1).

The monolith runner is decomposed into prepare / execute / finalize
phases; every value a phase needs is materialized exactly once here,
so the phases stay pure functions of the context and the failure
lifecycle lives in a thin driver (``runner.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from csd_observer.config.validate import seed_schedule
from csd_observer.datasets.registry import get_dataset
from csd_observer.outputs.ledger import RunLedger
from csd_observer.outputs.writer import OutputWriter


@dataclass
class RunContext:
    """Everything one run needs, resolved from the composed config.

    ``writer``/``ledger``/``run_id``/``meta``/``config_hash`` are
    filled by :func:`csd_observer.orchestration.phase_prepare.prepare`
    before ``execute`` runs; ``dataset_manifest_hashes`` is computed
    there from the dataset's manifest (real) or provenance (synthetic).
    """

    config: dict[str, Any]
    cli_overrides: list[str] = field(default_factory=list)

    # ---- resolved run knobs (§10.2 / §13) ----
    dataset_cfg: dict[str, Any] = field(default_factory=dict)
    dataset: str = ""
    methods: list[str] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    evaluation_name: str = "persistenceaware"
    k_persist: int = 5
    fpr_target: float = 0.05
    early_start_delta: float = 50.0
    early_end_delta: float = 5.0
    training: dict[str, Any] = field(default_factory=dict)
    base_dir: str = "outputs"
    store_trajectories: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)
    seed_offset: int = 0
    n_seeds: int = 1
    families: dict[str, str] = field(default_factory=dict)
    learned: list[str] = field(default_factory=list)

    # ---- filled by prepare() ----
    config_hash: str = ""
    dataset_manifest_hashes: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    writer: OutputWriter | None = None
    ledger: RunLedger | None = None
    run_id: str = ""


def load_bundle(dataset: str, overrides: dict[str, Any], seed_base: int) -> dict[str, Any]:
    """Load a dataset bundle with the §13 seed defaulted in.

    ``seed_base`` is the run-seed base for the current seed index; the
    caller adds it when it must be explicit (per-seed loop), and the
    s=0 signal seed for the pre-loop provenance load.
    """
    effective = dict(overrides)
    effective.setdefault("seed", seed_base)
    bundle = get_dataset(dataset, effective)
    meta = bundle.get("meta", {})
    if not meta.get("bif_type"):
        raise ValueError(f"dataset {dataset!r} returned no meta.bif_type")
    return bundle


def preloop_seed(seed_offset: int) -> int:
    """Signal run seed of seed index 0 — used for the pre-loop meta load."""
    return int(seed_schedule(seed_offset, 0)["signal"])


__all__ = ["RunContext", "load_bundle", "preloop_seed"]
