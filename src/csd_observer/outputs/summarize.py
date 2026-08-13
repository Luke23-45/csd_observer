"""Top-level summarize() entry point.

Reads ``results.jsonl`` for a run, rebuilds aggregates/bootstrap/Wilcoxon
CSVs in ``tables/`` and the run-level ``metrics.json`` summary
(R4: ``metrics/metrics.json`` aggregates across seeds per method).
Idempotent; safe to call multiple times.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from csd_observer.outputs.schema import ResultRow, validate_row
from csd_observer.outputs.tables import (
    write_aggregates_csv,
    write_bootstrap_csv,
    write_paired_wilcoxon_csv,
)

_AGGREGATE_METRICS = (
    "detection_rate",
    "detection_time_mean",
    "ew_auc",
    "fpr",
    "persistent_fpr",
)


def read_results_jsonl(path: str | Path) -> list[ResultRow]:
    """Read and schema-validate a results.jsonl file."""
    rows: list[ResultRow] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            d = json.loads(line)
            validate_row(d)
            # filter out the optional extra stats for dataclass construction
            for k in ("detection_time_95_low", "detection_time_95_high", "achieved_fpr_target", "extra"):
                d.setdefault(k, None if k in ("detection_time_95_low", "detection_time_95_high", "achieved_fpr_target") else {})
            rows.append(ResultRow(**d))
    return rows


def build_metrics_summary(rows: list[ResultRow]) -> dict[str, Any]:
    """Aggregate per-method metrics across seeds for ``metrics.json`` (R4).

    Output shape: ``{"summary": {method: {metric: mean}}, "n_seeds": ...,
    "rows": n}``. A metric aggregates over finite values only (NaN seeds
    are skipped, their count reported in ``n_nan`` per metric).
    """
    methods: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        methods.setdefault(row.method, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for method, method_rows in methods.items():
        entry: dict[str, Any] = {}
        for metric in _AGGREGATE_METRICS:
            values = [
                float(getattr(r, metric))
                for r in method_rows
                if math.isfinite(float(getattr(r, metric)))
            ]
            entry[metric] = float(sum(values) / len(values)) if values else None
            entry[f"n_nan_{metric}"] = len(method_rows) - len(values)
        summary[method] = entry
    return {
        "summary": summary,
        "n_seeds": len({(r.seed, r.replicate) for r in rows}),
        "rows": len(rows),
    }


def _protocol_checks_csv(run_dir: Path) -> Path | None:
    """Write ``tables/protocol_checks.csv`` from the metrics snapshot."""
    metrics_dir = run_dir / "metrics" / "protocol_checks.json"
    if not metrics_dir.exists():
        return None
    checks = json.loads(metrics_dir.read_text(encoding="utf-8"))
    if not isinstance(checks, list) or not checks:
        return None
    import csv

    path = run_dir / "tables" / "protocol_checks.csv"
    keys = list(checks[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in checks:
            writer.writerow({k: row.get(k) for k in keys})
    return path


def summarize_run(run_dir: str | Path) -> dict[str, Path]:
    """Rebuild tables/ + metrics.json from results.jsonl under <run_dir>/."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results" / "results.jsonl"
    if not results_path.exists():
        return {}
    rows = read_results_jsonl(results_path)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregates": write_aggregates_csv(rows, tables_dir / "aggregates.csv"),
        "bootstrap": write_bootstrap_csv(rows, tables_dir / "bootstrap_ci.csv"),
        "wilcoxon": write_paired_wilcoxon_csv(rows, tables_dir / "paired_wilcoxon.csv"),
        "protocol_checks": _protocol_checks_csv(run_dir),
    }
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(build_metrics_summary(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["metrics"] = metrics_path
    return paths


__all__ = ["build_metrics_summary", "read_results_jsonl", "summarize_run"]
