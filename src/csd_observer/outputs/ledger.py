"""Append-only run ledger.

Two views of the same append-only JSONL:

* ``outputs/_ledger/runs.jsonl``  — one row per run, atomic appends
* ``outputs/_ledger/index.json``  — rebuilt from runs.jsonl on demand

Rows mirror the lifecycle markers written by OutputWriter and carry
hashes + paths so analyses never have to walk the filesystem.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock


@dataclass
class LedgerRow:
    run_id: str
    timestamp: str
    run_name: str
    dataset: str
    methods: list[str]
    k_persist: int
    config_hash: str
    git_sha: str
    status: str
    path: str
    bif_types: list[str] = field(default_factory=list)
    is_learned_methods: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunLedger:
    """Append-only JSONL ledger with index.json mirror."""

    def __init__(self, ledger_dir: str | Path = "outputs/_ledger", *, index_throttle: int = 25) -> None:
        """Append-only ledger with throttled index rebuilds.

        ``index_throttle`` controls how often :meth:`append` rebuilds the
        ``index.json`` mirror. The mirror is **always** rebuilt after
        :meth:`update_status` and :meth:`append` is called by
        :meth:`mark_completed` (via :class:`OutputWriter` lifecycle);
        in-between, the reader falls back to ``runs.jsonl`` so the cost
        of a full rebuild is amortised over ``index_throttle`` runs.
        """
        self.dir = Path(ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / "runs.jsonl"
        self.index_path = self.dir / "index.json"
        self.lock = FileLock(str(self.dir / ".ledger.lock"))
        self._index_throttle = max(1, int(index_throttle))
        self._appends_since_rebuild = 0

    def append(self, row: LedgerRow) -> None:
        if not row.created_at:
            row.created_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(row.to_dict()) + "\n"
        with self.lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            self._appends_since_rebuild += 1
            if self._appends_since_rebuild >= self._index_throttle:
                self._rebuild_index_locked()
                self._appends_since_rebuild = 0

    def flush(self) -> None:
        """Force a rebuild of the index mirror (e.g. at end of a batch)."""
        with self.lock:
            self._rebuild_index_locked()
            self._appends_since_rebuild = 0

    def update_status(self, run_id: str, status: str) -> None:
        with self.lock:
            rows = self.read_all()
            found = False
            for r in rows:
                if r.run_id == run_id:
                    r.status = status
                    found = True
            if not found:
                raise KeyError(f"Unknown run_id: {run_id}")
            self._rewrite_all(rows)
            # A status change is a semantically significant event; rebuild
            # the mirror unconditionally so any subsequent reader sees it.
            self._rebuild_index_locked()
            self._appends_since_rebuild = 0

    def read_all(self) -> list[LedgerRow]:
        if not self.jsonl_path.exists():
            return []
        out: list[LedgerRow] = []
        # Ignore partial trailing lines (crash mid-append) so the reader
        # is robust against truncated records.
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                out.append(LedgerRow(**d))
            except (TypeError, ValueError):
                continue
        return out

    def find_by_id(self, run_id: str) -> LedgerRow | None:
        for r in self.read_all():
            if r.run_id == run_id:
                return r
        return None

    def _rewrite_all(self, rows: list[LedgerRow]) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self.jsonl_path.parent), prefix=".ledger_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r.to_dict()) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.jsonl_path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _rebuild_index_locked(self) -> None:
        """Rebuild ``index.json`` from ``runs.jsonl``.

        Caller MUST hold :attr:`lock`. Reads the JSONL from disk so
        we do not depend on in-memory state.
        """
        rows = self.read_all()
        idx = {
            "rebuilt_at": datetime.now(timezone.utc).isoformat(),
            "count": len(rows),
            "runs": [r.to_dict() for r in rows],
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.index_path.parent), prefix=".index_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(idx, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.index_path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


__all__ = ["RunLedger", "LedgerRow"]
