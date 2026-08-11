"""Stable error taxonomy for dataset ingestion and processing."""
from enum import Enum


class DatasetErrorCode(str, Enum):
    INGEST_NETWORK = "INGEST_NETWORK"
    INGEST_AUTH = "INGEST_AUTH"
    INGEST_MANIFEST_MISMATCH = "INGEST_MANIFEST_MISMATCH"
    INGEST_CHECKSUM = "INGEST_CHECKSUM"
    INGEST_ARCHIVE = "INGEST_ARCHIVE"
    PROCESS_NONFINITE = "PROCESS_NONFINITE"
    PROCESS_SHORT_LENGTH = "PROCESS_SHORT_LENGTH"
    PROCESS_ANNOTATION_MISSING = "PROCESS_ANNOTATION_MISSING"
    SPLIT_IMBALANCE = "SPLIT_IMBALANCE"
    MANIFEST_CORRUPT = "MANIFEST_CORRUPT"


class DatasetError(RuntimeError):
    def __init__(self, code: DatasetErrorCode, message: str) -> None:
        super().__init__(f"[{code.value}] {message}")
        self.code = code


__all__ = ["DatasetError", "DatasetErrorCode"]
