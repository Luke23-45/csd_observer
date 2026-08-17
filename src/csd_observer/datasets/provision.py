"""Provisioning: turn a verified raw dataset into a processed bundle.

``provision_dataset`` is the orchestration entry point: it runs the
dataset-agnostic pipeline (ingest -> processor -> gates -> split ->
normalization -> manifest) for the real datasets and short-circuits to
``READY_PROCESSED`` for synthetic.
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
    if name in ("daphnia_ext", "daphnia"):
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
    force_rebuild: bool = False,
) -> IngestState:
    """Ensure ``<root>/processed/<name>`` exists and is current; idempotent.

    Args:
        name: dataset name (``tac``, ``daphnia_ext``, or a synthetic
            name, which short-circuits without touching disk).
        config: the composed dataset-group config.
        root: data root (default ``datasets``).
        token: optional Dryad API token for auto download mode.
        force_rebuild: force re-running raw processing even if cache is present.

    Returns:
        the reached :class:`IngestState`.
    """
    if str(config.get("source", "synthetic")) == "synthetic":
        return IngestState.READY_PROCESSED
    if name not in REAL_DATASETS and name != "daphnia":
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"no processor for dataset {name!r}")
    processor, validator = _processor(name)
    return run_pipeline(
        config,
        Path(root),
        name,
        processor,
        token=token,
        extra_validator=validator,
        force_rebuild=force_rebuild,
    )


__all__ = ["REAL_DATASETS", "provision_dataset"]
