"""TAC TDMS inspection and explicit-channel processing primitives."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.ingest import extract_archive


def inspect_tdms(archive: str | Path, extraction_dir: str | Path) -> list[dict[str, Any]]:
    """Return group/channel metadata without selecting a channel implicitly."""
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "npTDMS is required to inspect TAC data") from exc
    files = extract_archive(archive, extraction_dir)
    records: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() != ".tdms":
            continue
        try:
            tdms = TdmsFile.read(str(path))
            for group in tdms.groups():
                for channel in group.channels():
                    values = np.asarray(channel[:])
                    records.append({"file": str(path), "group": group.name, "channel": channel.name,
                                    "length": int(values.size), "dtype": str(values.dtype),
                                    "finite": bool(np.isfinite(values).all()) if values.size else True})
        except (OSError, ValueError, TypeError) as exc:
            raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"failed to parse {path}: {exc}") from exc
    if not records:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "archive contains no readable TDMS channels")
    return records


def extract_channel(tdms_path: str | Path, *, group: str, channel: str, min_length: int) -> np.ndarray:
    """Read one explicitly named numeric channel and enforce finite data."""
    try:
        from nptdms import TdmsFile
        values = np.asarray(TdmsFile.read(str(tdms_path))[group][channel][:], dtype=np.float32)
    except (ImportError, OSError, KeyError, ValueError, TypeError) as exc:
        code = DatasetErrorCode.INGEST_ARCHIVE if isinstance(exc, (ImportError, OSError)) else DatasetErrorCode.PROCESS_NONFINITE
        raise DatasetError(code, f"cannot read TDMS channel {group}/{channel}: {exc}") from exc
    if values.ndim != 1 or len(values) < int(min_length):
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, f"channel {group}/{channel} has length {len(values)}")
    if not np.isfinite(values).all():
        raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"channel {group}/{channel} contains non-finite values")
    return values


__all__ = ["extract_channel", "inspect_tdms"]
