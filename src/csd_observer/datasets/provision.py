"""Provisioning: turn a verified raw dataset into a processed bundle (§5.4–5.6).

``provision_dataset`` is the orchestration entry point: it runs the
dataset-agnostic pipeline (ingest -> processor -> gates -> split ->
normalization -> manifest) for the real datasets and short-circuits to
``READY_PROCESSED`` for synthetic (the registry fast path builds those
in memory; nothing needs materializing).

Every real dataset has exactly one processor handle plus an annotation
spot-check validator (§5.4 gate 3). The processor table is the single
place where dataset name -> processor wiring lives; the registry's
``list_datasets`` reports these names so configuration validation accepts
them before the first provisioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.pipeline import run_pipeline
from csd_observer.datasets.common.states import IngestState

#: Registry-resolvable real datasets (each with a processor below).
REAL_DATASETS: tuple[str, ...] = ("tac", "daphnia_ext")


def _processor(name: str):
    if name == "tac":
        from csd_observer.datasets.tac.process import process as _tac_process
        from csd_observer.datasets.tac.process import validate_annotation as _tac_validate

        return _tac_process, _tac_validate
    if name == "daphnia_ext":
        from csd_observer.datasets.daphnia_ext.process import process as _daphnia_process
        from csd_observer.datasets.daphnia_ext.process import (
            validate_annotation as _daphnia_validate,
        )

        return _daphnia_process, _daphnia_validate
    raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"no processor for dataset {name!r}")


def provision_dataset(
    name: str,
    config: dict[str, Any],
    root: str | Path = "datasets",
    *,
    token: str | None = None,
) -> IngestState:
    """Ensure ``<root>/processed/<name>`` exists and is current; idempotent.

    Args:
        name: dataset name (``tac``, ``daphnia_ext``, or a synthetic
            name, which short-circuits without touching disk).
        config: the composed dataset-group config (``expected_files``,
            ``download``, ``doi``, ``processing``, ``split``, ...).
        root: data root (default ``datasets``; the pipeline appends
            ``raw/<name>`` and ``processed/<name>``). Pass the run's
            ``data_root`` override.
        token: optional Dryad API token for auto download mode.

    Returns:
        the reached :class:`IngestState` (``READY_PROCESSED`` when the
        processed manifest is valid and matches the config).
    """
    if str(config.get("source", "synthetic")) == "synthetic":
        return IngestState.READY_PROCESSED
    if name not in REAL_DATASETS:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"no processor for dataset {name!r}")
    processor, validator = _processor(name)
    return run_pipeline(
        config,
        Path(root),
        name,
        processor,
        token=token,
        extra_validator=validator,
    )


__all__ = ["REAL_DATASETS", "provision_dataset"]
