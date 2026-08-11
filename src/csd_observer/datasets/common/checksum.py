"""Checksum helpers used by both automatic and manual ingestion."""
import hashlib
from pathlib import Path

from .errors import DatasetError, DatasetErrorCode


def md5_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: str | Path, expected: str) -> str:
    actual = md5_file(path)
    if actual.lower() != expected.lower():
        raise DatasetError(DatasetErrorCode.INGEST_CHECKSUM, f"{path}: expected {expected}, got {actual}")
    return actual


__all__ = ["md5_file", "verify_md5"]
