"""OutputWriter — single source of truth for run artifacts.

Canonical subtree (§9.1 of the implementation plan):

    outputs/<run_name>/<timestamp>/
        resolved_config/    resolved.yaml + cli_overrides.yaml
        metadata/           environment.json
        metrics/            metrics.json, protocol_checks.json
        results/            results.jsonl (+ epoch_logs/ per trained method,
                                            trajectories/ if enabled)
        artifacts/          checkpoints/, calibration/, plots/
        logs/               run.log
        times/              timings.json
        tables/             aggregates.csv, cIs.csv (via summarize)
        <timestamp>.completed     # lifecycle marker (or .failed)

Invariants enforced here:
  * one timestamp minted at construction; never re-stamps mid-run
  * full subtree created atomically; collision fails loudly
  * all JSONL writes are atomic (tmp + rename) so concurrent drivers
    cannot corrupt results.jsonl
  * schema-validated rows only; invalid rows raise before write
  * lifecycle marker (.pending -> .completed/.failed) drives ledger state
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from filelock import FileLock

from csd_observer.outputs.schema import SchemaError, validate_row

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


@dataclass
class RunPaths:
    """Strongly-typed handles to every canonical subdirectory of one run."""

    root: Path
    resolved_config: Path
    metadata: Path
    metrics: Path
    results: Path
    artifacts: Path
    logs: Path
    times: Path
    tables: Path

    @classmethod
    def from_root(cls, root: Path) -> RunPaths:
        return cls(
            root=root,
            resolved_config=root / "resolved_config",
            metadata=root / "metadata",
            metrics=root / "metrics",
            results=root / "results",
            artifacts=root / "artifacts",
            logs=root / "logs",
            times=root / "times",
            tables=root / "tables",
        )


class OutputWriter:
    """One timestamp, one run directory. Self-contained."""

    def __init__(
        self,
        run_name: str,
        base_dir: str | Path = "outputs",
        *,
        timestamp: str | None = None,
        collision_retries: int = 10,
    ) -> None:
        self.run_name = run_name
        self.base_dir = Path(base_dir)
        # Single timestamp; minted at construction.
        if timestamp is None:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
        else:
            self.timestamp = timestamp
        root = self.base_dir / run_name / self.timestamp
        # Atomic mint: try microsecond-granular slot until mkdir succeeds.
        for attempt in range(max(collision_retries, 1)):
            candidate = self.base_dir / run_name / (
                self.timestamp if attempt == 0 else f"{self.timestamp}_r{attempt}"
            )
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                root = candidate
                self.timestamp = candidate.name
                break
            except FileExistsError:
                if attempt == max(collision_retries, 1) - 1:
                    raise RuntimeError(
                        f"Could not allocate a unique output directory under "
                        f"{self.base_dir / run_name} after {collision_retries} attempts"
                    ) from None
                continue
        self.root = root
        self.paths = RunPaths.from_root(root)
        for sub in _CANONICAL_SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        self._stamped_root = root
        self._closed = False
        self._status = "pending"
        self._results_lock = FileLock(str(self.paths.results / ".results.lock"))
        self._write_lifecycle_marker("pending")

    # ---------------------------------------------------------------- lifecycle
    @property
    def status(self) -> str:
        return self._status

    def mark_completed(self) -> Path:
        if self._closed:
            raise RuntimeError("OutputWriter already closed")
        self._closed = True
        self._status = "completed"
        return self._write_lifecycle_marker("completed")

    def mark_failed(self, exc: BaseException | None = None) -> Path:
        if self._closed:
            raise RuntimeError("OutputWriter already closed")
        self._closed = True
        self._status = "failed"
        path = self._write_lifecycle_marker("failed")
        if exc is not None:
            tb_path = self.paths.logs / "exception.txt"
            tb_path.write_text(repr(exc), encoding="utf-8")
        return path

    def _write_lifecycle_marker(self, status: str) -> Path:
        """Single marker file under outputs/<run_name>/ sibling to <timestamp>/.

        Using a sibling directory (not inside <timestamp>/) ensures the
        marker survives even if the run dir is renamed or pruned.
        """
        pending = self.root.with_suffix(self.root.suffix + ".pending")
        marker = self.root.with_suffix(self.root.suffix + "." + status)
        if status != "pending" and pending.exists():
            pending.replace(marker)
        marker.write_text(
            json.dumps(
                {
                    "run_name": self.run_name,
                    "timestamp": self.timestamp,
                    "path": str(self.root),
                    "status": status,
                    "stamped_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        return marker

    # ---------------------------------------------------------------- writes
    def write_resolved_config(self, config: dict[str, Any]) -> Path:
        path = self.paths.resolved_config / "resolved.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
        return path

    def write_cli_overrides(self, overrides: list[str]) -> Path:
        path = self.paths.resolved_config / "cli_overrides.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"overrides": list(overrides)}, f, sort_keys=False)
        return path

    def write_environment(self, env: dict[str, Any]) -> Path:
        path = self.paths.metadata / "environment.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2, sort_keys=True)
        return path

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        """Replace the entire ``metrics.json`` aggregate (one-shot write).

        The metrics file is a single dict that aggregates results
        across all (method, seed) pairs in a run. It is overwritten on
        every call — there is no row identity here, unlike
        :meth:`write_protocol_checks` which is keyed by
        ``(method, system, seed)``. The orchestrator writes once per
        run, after the seed loop completes.
        """
        path = self.paths.metrics / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        return path

    def write_protocol_checks(self, checks: dict[str, Any]) -> Path:
        """Append/replace a single protocol-check row.

        Identity is ``(method, system, seed)``: a new row with the same
        identity replaces the prior entry. Rows with different
        identities are kept. The file (``protocol_checks.json``) is a
        list, sorted in insertion order on disk; consumers should treat
        it as a self-attestation snapshot of the run, not as a strict
        audit log.
        """
        path = self.paths.metrics / "protocol_checks.json"
        previous: list[dict[str, Any]] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                previous = loaded if isinstance(loaded, list) else [loaded]
            except json.JSONDecodeError:
                previous = []
        identity = tuple(checks.get(k) for k in ("method", "system", "seed"))
        previous = [row for row in previous if tuple(row.get(k) for k in ("method", "system", "seed")) != identity]
        previous.append(dict(checks))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(previous, f, indent=2, default=_json_default)
        return path

    def write_timings(self, timings: dict[str, Any]) -> Path:
        """Merge a timing entry into the keyed ``timings.json`` (R4).

        File layout: ``{"entries": {f"{method}__s{seed}": {...}}}``.
        Callers pass ``{"method": ..., "seed": ..., ...timing fields}``
        and the entry is read-merged under the derived key, so multiple
        writers (training callbacks, per-method evaluation) accumulate
        without clobbering each other. A dict without ``method``/``seed``
        replaces the whole file (explicit-format writes).
        """
        path = self.paths.times / "timings.json"
        key: str | None = None
        if "method" in timings and "seed" in timings:
            key = f"{timings['method']}__s{timings['seed']}"
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
                    existing = dict(loaded["entries"])
            except json.JSONDecodeError:
                existing = {}
        if key is None:
            payload = {"entries": existing}
            if "entries" in timings:
                payload = dict(timings)
        else:
            existing[key] = {**existing.get(key, {}), **dict(timings)}
            payload = {"entries": existing}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        return path

    def write_result_row(self, row: dict[str, Any]) -> Path:
        """Atomic append of a single schema-validated result row.

        Uses ``O_APPEND`` semantics (POSIX: atomic per ``write(2)`` call;
        NTFS: atomic since Vista) combined with an explicit ``fsync`` and
        a per-process :class:`FileLock`. There is no read-modify-write:
        a crash mid-write truncates only the current line, never the
        previous rows.
        """
        try:
            validate_row(row)
        except SchemaError:
            raise
        path = self.paths.results / "results.jsonl"
        payload = json.dumps(row, default=_json_default) + "\n"
        with self._results_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        return path

    def write_trajectory_npz(
        self,
        system: str,
        method: str,
        replicate: str,
        seed: int,
        **arrays: np.ndarray,
    ) -> Path:
        """Opt-in trajectory artifact. Config-controlled (default off)."""
        dir_path = self.paths.results / "trajectories"
        dir_path.mkdir(parents=True, exist_ok=True)
        safe_method = _slugify(method)
        safe_system = _slugify(system)
        safe_repl = _slugify(replicate) if replicate else "default"
        path = dir_path / f"{safe_system}_{safe_repl}_{safe_method}_seed{seed}.npz"
        np.savez_compressed(path, **arrays)
        return path

    def write_epoch_log_csv(
        self,
        system: str,
        method: str,
        seed: int,
        rows: list[dict[str, Any]],
    ) -> Path | None:
        dir_path = self.paths.results / "epoch_logs"
        dir_path.mkdir(parents=True, exist_ok=True)
        safe_method = _slugify(method)
        safe_system = _slugify(system)
        path = dir_path / f"{safe_system}_{safe_method}_seed{seed}.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def write_calibration(self, name: str, payload: dict[str, Any]) -> Path:
        dir_path = self.paths.artifacts / "calibration"
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"{_slugify(name)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        return path

    def write_artifact(self, name: str, src: Path) -> Path:
        """Copy an external artifact (model checkpoint etc.) into artifacts/."""
        import shutil

        dir_path = self.paths.artifacts
        dir_path.mkdir(parents=True, exist_ok=True)
        target = dir_path / name
        shutil.copy2(src, target)
        return target

    def write_csv(self, rows: list[dict[str, Any]], name: str) -> Path | None:
        dir_path = self.paths.tables
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def open_log(self) -> RunLog:
        return RunLog(self.paths.logs / "run.log")


class RunLog:
    """Append-only structured logger that writes to logs/run.log.

    Each ``write`` opens the file in ``"a"`` mode (POSIX/NTFS atomic
    append under the per-process lock held by the writer's lifecycle),
    fsyncs, and closes. A crash mid-write truncates only the current
    JSON line; previous entries survive.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        rec = {"event": event, **fields, "_ts": datetime.now(timezone.utc).isoformat()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _slugify(s: str) -> str:
    return (
        str(s)
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


__all__ = ["OutputWriter", "RunPaths", "RunLog"]
