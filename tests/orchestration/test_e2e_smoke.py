"""L7.5: end-to-end smoke test for one governed benchmark run.

Composes the real Hydra run config (same path as the CLI), runs the
orchestration pipeline on a tiny synthetic dataset, and asserts the
canonical output tree, lifecycle marker, ledger row, and schema-valid
result rows — with no network access (synthetic fast path only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from csd_observer.config.store import register_configs
from csd_observer.config.validate import validate_config
from csd_observer.outputs.ledger import RunLedger
from csd_observer.outputs.schema import validate_row

_REPO = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = os.path.relpath(_REPO / "configs", Path(__file__).resolve().parent)

_CANONICAL_SUBDIRS = (
    "resolved_config",
    "metadata",
    "metrics",
    "results",
    "artifacts",
    "logs",
    "times",
    "tables",
)


@pytest.fixture()
def smoke_config(tmp_path: Path) -> dict:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    register_configs()
    with initialize(version_base=None, config_path=_CONFIG_PATH):
        cfg = compose(
            config_name="run",
            overrides=[
                "+dataset_overrides.n_trajectories=16",
                "+dataset_overrides.max_length=64",
            ],
        )
    config = OmegaConf.to_container(cfg, resolve=True)
    config["output"]["base_dir"] = str(tmp_path)
    validate_config(config)
    return config


def test_e2e_smoke(smoke_config: dict, tmp_path: Path) -> None:
    from csd_observer.orchestration.runner import run_benchmark

    writer = run_benchmark(smoke_config)

    # -- canonical tree ------------------------------------------------------
    assert writer.root.parent == tmp_path / "synthetic_fold"
    for sub in _CANONICAL_SUBDIRS:
        assert (writer.root / sub).is_dir(), f"missing canonical dir {sub}"
    assert (writer.root / "resolved_config" / "resolved.yaml").is_file()
    assert (writer.root / "resolved_config" / "cli_overrides.yaml").is_file()
    assert (writer.root / "metadata" / "environment.json").is_file()

    # -- lifecycle marker ------------------------------------------------------
    marker = writer.root.with_suffix(writer.root.suffix + ".completed")
    assert marker.is_file(), f"missing .completed marker at {marker}"
    assert writer.status == "completed"
    state = json.loads(marker.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["run_name"] == "synthetic_fold"

    # -- result rows are schema-valid -----------------------------------------
    results = writer.root / "results" / "results.jsonl"
    assert results.is_file()
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "results.jsonl is empty"
    for row in rows:
        validate_row(row)  # raises SchemaError on malformed rows
        assert row["dataset"] == "synthetic_fold"
        assert row["method"] == "VAR-CSD"
        assert row["bif_type"] == "fold"

    # -- protocol checks + tables written -------------------------------------
    assert (writer.root / "metrics" / "protocol_checks.json").is_file()
    assert (writer.root / "tables" / "aggregates.csv").is_file()

    # -- ledger row -------------------------------------------------------------
    ledger = RunLedger(tmp_path / "_ledger")
    entries = ledger.read_all()
    assert len(entries) == 1
    row = entries[0]
    assert row.status == "completed"
    assert row.run_id == f"synthetic_fold-{writer.timestamp}"
    assert row.dataset == "synthetic_fold"
    assert row.methods == ["VAR-CSD"]

    # -- no network: synthetic provenance recorded -----------------------------
    env = json.loads((writer.root / "metadata" / "environment.json").read_text(encoding="utf-8"))
    assert env.get("config_hash")
    resolved = (writer.root / "resolved_config" / "resolved.yaml").read_text(encoding="utf-8")
    assert "source: synthetic" in resolved


def test_e2e_smoke_two_seeds(smoke_config: dict, tmp_path: Path) -> None:
    from csd_observer.orchestration.runner import run_benchmark

    smoke_config["n_seeds"] = 2
    smoke_config["seed_offset"] = 0
    writer = run_benchmark(smoke_config)

    marker = writer.root.with_suffix(writer.root.suffix + ".completed")
    assert marker.is_file()
    rows = [
        json.loads(line)
        for line in (writer.root / "results" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    seeds = sorted({row["seed"] for row in rows})
    assert seeds == [0, 1]
