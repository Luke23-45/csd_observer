
"""Shared dataset ingestion, validation, and split primitives."""

from .checksum import md5_file, verify_md5
from .dryad import DryadClient, DryadFile
from .errors import DatasetError, DatasetErrorCode
from .ingest import extract_archive, ingest_raw
from .manifest import read_manifest, write_manifest
from .pipeline import MIN_LENGTH, content_hash, run_pipeline
from .states import IngestState, cached_manifest

__all__ = [
    "DatasetError", "DatasetErrorCode", "DryadClient", "DryadFile", "IngestState",
    "MIN_LENGTH", "cached_manifest", "content_hash", "extract_archive", "ingest_raw",
    "md5_file", "read_manifest", "run_pipeline", "verify_md5", "write_manifest",
]
