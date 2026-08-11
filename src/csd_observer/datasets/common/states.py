"""Explicit ingestion lifecycle and idempotent processed-cache checks."""
from __future__ import annotations

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
    path = Path(processed_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = read_manifest(path)
    except Exception:
        return None
    gates = manifest.get("gates", {})
    return manifest if "passed" in gates else None


__all__ = ["IngestState", "cached_manifest"]
