"""Shared dataset ingestion, validation, and split primitives."""

from .checksum import md5_file, verify_md5
from .contract import validate_bundle
from .datamodule import BaseCSDDataModule, CSDDataset, compute_alarm_labels, pad_collate_fn
from .dryad import DryadClient, DryadFile
from .errors import DatasetError, DatasetErrorCode
from .ingest import extract_archive, ingest_raw
from .manifest import read_manifest, write_manifest
from .pipeline import MIN_LENGTH, content_hash, run_pipeline
from .remote import fetch_from_huggingface, fetch_remote_dataset
from .split import replicate_split
from .states import IngestState, cached_manifest

__all__ = [
    "BaseCSDDataModule",
    "CSDDataset",
    "DatasetError",
    "DatasetErrorCode",
    "DryadClient",
    "DryadFile",
    "IngestState",
    "MIN_LENGTH",
    "cached_manifest",
    "compute_alarm_labels",
    "content_hash",
    "extract_archive",
    "fetch_from_huggingface",
    "fetch_remote_dataset",
    "ingest_raw",
    "md5_file",
    "pad_collate_fn",
    "read_manifest",
    "replicate_split",
    "run_pipeline",
    "validate_bundle",
    "verify_md5",
    "write_manifest",
]
