"""Validated, durable processed-dataset manifests."""
import json
from pathlib import Path
from typing import Any

from .errors import DatasetError, DatasetErrorCode


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    required = {"schema_version", "dataset", "processing", "gates", "split", "content_hash"}
    missing = required - set(manifest)
    if missing:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"missing keys: {sorted(missing)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def read_manifest(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, str(exc)) from exc
    if not isinstance(value, dict):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, "manifest root must be an object")
    required = {"schema_version", "dataset", "processing", "gates", "split", "content_hash"}
    missing = required - set(value)
    if missing:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"missing keys: {sorted(missing)}")
    if not isinstance(value["schema_version"], (str, int)) or not isinstance(value["content_hash"], str):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, "invalid schema_version/content_hash types")
    if not isinstance(value["gates"], dict) or not isinstance(value["gates"].get("passed"), list):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, "gates.passed must be a list")
    if not isinstance(value["split"], dict):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, "split must be an object")
    return value


__all__ = ["read_manifest", "write_manifest"]
