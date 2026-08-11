"""Outputs package: self-contained run artifact system.

Public API:

* :class:`OutputWriter` — one timestamp, one run directory, atomic writes,
  schema validation, lifecycle markers.
* :class:`RunLedger` — append-only ``runs.jsonl`` + ``index.json``.
* :class:`ResultRow` / :func:`validate_row` — canonical row schema.
* :func:`collect_environment` — ``metadata/environment.json`` fingerprint.
* :func:`summarize_run` — rebuild paper-ready ``tables/`` from ``results.jsonl``.
"""

from csd_observer.outputs.ledger import LedgerRow, RunLedger
from csd_observer.outputs.metadata import collect_environment, hash_config, hash_file
from csd_observer.outputs.schema import ResultRow, SchemaError, validate_row
from csd_observer.outputs.summarize import read_results_jsonl, summarize_run
from csd_observer.outputs.tables import (
    Aggregate,
    aggregate_rows,
    bootstrap_ci,
    bootstrap_ci_per_row,
    paired_wilcoxon,
    write_aggregates_csv,
    write_bootstrap_csv,
    write_paired_wilcoxon_csv,
)
from csd_observer.outputs.writer import OutputWriter, RunLog, RunPaths

__all__ = [
    "OutputWriter",
    "RunLog",
    "RunPaths",
    "RunLedger",
    "LedgerRow",
    "ResultRow",
    "SchemaError",
    "validate_row",
    "collect_environment",
    "hash_config",
    "hash_file",
    "Aggregate",
    "aggregate_rows",
    "bootstrap_ci",
    "bootstrap_ci_per_row",
    "paired_wilcoxon",
    "write_aggregates_csv",
    "write_bootstrap_csv",
    "write_paired_wilcoxon_csv",
    "read_results_jsonl",
    "summarize_run",
]
