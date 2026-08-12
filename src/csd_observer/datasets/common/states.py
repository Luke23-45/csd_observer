"""Explicit ingestion lifecycle and idempotent processed-cache checks."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from .manifest import read_manifest


class IngestState(str, Enum):
    UNKNOWN = "UNKNOWN"
    RESOLVE = "RESOLVE"
    FETCH_MANIFEST = "FETCH_MANIFEST"
    VERIFY_EXPECTED = "VERIFY_EXPECTED"
    DOWNLOAD_OR_WAIT = "DOWNLOAD_OR_WAIT"
    VERIFY_CHECKSUM = "VERIFY_CHECKSUM"
    EXTRACT = "EXTRACT"
    READY_RAW = "READY_RAW"
    PROCESS = "PROCESS"
    SPLIT = "SPLIT"
    READY_PROCESSED = "READY_PROCESSED"


def cached_manifest(processed_dir: str | Path) -> dict | None:
    """Return the cached manifest iff it parses and has a ``gates.passed`` list.

    Only ``OSError`` (missing/unreadable file) and ``(json.JSONDecodeError,
    ValueError)`` are treated as "no cache"; other errors propagate so a
    silently-corrupted manifest cannot trigger a destructive re-ingest.
    """
    path = Path(processed_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = read_manifest(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    gates = manifest.get("gates", {})
    return manifest if isinstance(gates.get("passed"), list) else None


__all__ = ["IngestState", "cached_manifest"]
