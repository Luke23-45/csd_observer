"""Top-level summarize() entry point.

Reads ``results.jsonl`` for a run, rebuilds aggregates/bootstrap/Wilcoxon
CSVs in ``tables/``. Idempotent; safe to call multiple times.
"""

from __future__ import annotations

from pathlib import Path

from csd_observer.outputs.schema import ResultRow, validate_row
from csd_observer.outputs.tables import (
    write_aggregates_csv,
    write_bootstrap_csv,
    write_paired_wilcoxon_csv,
)


def read_results_jsonl(path: str | Path) -> list[ResultRow]:
    """Read and schema-validate a results.jsonl file."""
    rows: list[ResultRow] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json

            d = json.loads(line)
            validate_row(d)
            # filter out the optional extra stats for dataclass construction
            for k in ("detection_time_95_low", "detection_time_95_high", "achieved_fpr_target", "extra"):
                d.setdefault(k, None if k in ("detection_time_95_low", "detection_time_95_high", "achieved_fpr_target") else {})
            rows.append(ResultRow(**d))
    return rows


def summarize_run(run_dir: str | Path) -> dict[str, Path]:
    """Rebuild tables/ from results.jsonl under <run_dir>/."""
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
    }
    return paths


__all__ = ["read_results_jsonl", "summarize_run"]
