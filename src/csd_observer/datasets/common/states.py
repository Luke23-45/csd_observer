"""Explicit ingestion lifecycle and idempotent processed-cache checks."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .manifest import read_manifest


class IngestState(str, Enum):
    UNKNOWN = "UNKNOWN"
    RESOLVE = "RESOLVE"
    FETCH_REMOTE = "FETCH_REMOTE"
    FETCH_MANIFEST = "FETCH_MANIFEST"
    VERIFY_EXPECTED = "VERIFY_EXPECTED"
    DOWNLOAD_OR_WAIT = "DOWNLOAD_OR_WAIT"
    VERIFY_CHECKSUM = "VERIFY_CHECKSUM"
    EXTRACT = "EXTRACT"
    READY_RAW = "READY_RAW"
    PROCESS = "PROCESS"
    SPLIT = "SPLIT"
    READY_PROCESSED = "READY_PROCESSED"


def cached_manifest(processed_dir: str | Path) -> dict[str, Any] | None:
    """Return the cached manifest iff it parses, has ``gates.passed``, and valid split files.

    Only ``OSError`` (missing/unreadable file) and ``(json.JSONDecodeError,
    ValueError)`` are treated as "no cache"; other errors propagate so a
    silently-corrupted manifest cannot trigger a destructive re-ingest.
    """
    processed_path = Path(processed_dir)
    manifest_file = processed_path / "manifest.json"
    if not manifest_file.exists():
        return None
    try:
        manifest = read_manifest(manifest_file)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    gates = manifest.get("gates", {})
    if not isinstance(gates.get("passed"), list):
        return None

    # Verify split files exist on disk
    split_info = manifest.get("split", {})
    splits = split_info.get("splits")
    if isinstance(splits, dict):
        for split_name, s_meta in splits.items():
            s_file = s_meta.get("file") or f"{split_name}.npz"
            s_path = processed_path / split_name / s_file
            if not s_path.is_file():
                return None
    else:
        # Check default train/val split files or legacy arrays.npz
        train_npz = processed_path / "train" / "train.npz"
        val_npz = processed_path / "val" / "val.npz"
        legacy_npz = processed_path / "arrays.npz"
        if not ((train_npz.is_file() and val_npz.is_file()) or legacy_npz.is_file()):
            return None

    return manifest


__all__ = ["IngestState", "cached_manifest"]
