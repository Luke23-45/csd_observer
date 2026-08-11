"""Checksum-first raw-file ingestion shared by TAC and DaphniaExt."""
from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Any

from filelock import FileLock

from .checksum import verify_md5
from .dryad import DryadClient
from .errors import DatasetError, DatasetErrorCode
from .states import IngestState, cached_manifest


def expected_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    files = config.get("expected_files")
    if not isinstance(files, list) or not files:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "expected_files must be non-empty")
    return files


def ingest_raw(config: dict[str, Any], root: str | Path, *, token: str | None = None) -> IngestState:
    """Resolve a dataset into ``root/raw``; never accepts an unchecked file.

    Manual mode is offline and succeeds only when every configured file is
    already present with its pinned digest.  Auto mode requires a token and
    still verifies every downloaded file before extraction.
    """
    root = Path(root)
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    if cached_manifest(processed) is not None:
        return IngestState.READY_PROCESSED
    lock = FileLock(str(root / ".ingest.lock"))
    with lock:
        files = expected_files(config)
        mode = str(config.get("download", {}).get("mode", "manual"))
        if mode not in {"manual", "auto"}:
            raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"unsupported download mode {mode!r}")
        if mode == "auto":
            client = DryadClient(token=token or os.environ.get(str(config.get("download", {}).get("env_token_var", "DRYAD_API_TOKEN"))),
                                 retries=int(config.get("download", {}).get("retries", 3)),
                                 backoff=float(config.get("download", {}).get("backoff", 1.0)))
            if not client.token:
                raise DatasetError(DatasetErrorCode.INGEST_AUTH, "auto mode requires a Dryad API token")
            remote = {f.path: f for f in client.files_for_doi(str(config["doi"]))}
            for spec in files:
                name = str(spec["path"])
                if name not in remote:
                    raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"remote file missing: {name}")
                destination = raw / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or _digest(destination) != str(spec["md5"]).lower():
                    client.download(remote[name], str(destination))
                verify_md5(destination, str(spec["md5"]))
        else:
            _manual_drop(config, files, raw)
        return IngestState.READY_RAW


def _manual_drop(config: dict[str, Any], files: list[dict[str, Any]], raw: Path) -> None:
    """§5.3 manual mode: poll up to ``download.wait_minutes`` for files.

    Prints the pinned file names + MD5s once at poll start so an operator
    knows exactly what to drop into ``raw/``. READY_RAW only on match;
    a timeout raises ``INGEST_CHECKSUM`` listing the still-missing files.
    """
    expected = {str(spec["path"]): str(spec["md5"]) for spec in files}
    missing = [name for name in expected if not (raw / name).exists()]
    if missing:
        print(f"[ingest] manual mode: drop the following files into {raw}:")
        for name, md5 in expected.items():
            print(f"  {name}  (md5 {md5})")
    wait_minutes = float(config.get("download", {}).get("wait_minutes", 0.0))
    if missing and wait_minutes > 0:
        interval = float(config.get("download", {}).get("poll_interval_sec", 2.0))
        deadline = time.monotonic() + wait_minutes * 60.0
        while missing and time.monotonic() < deadline:
            time.sleep(interval)
            missing = [name for name in missing if not (raw / name).exists()]
    if missing:
        raise DatasetError(
            DatasetErrorCode.INGEST_CHECKSUM,
            f"manual files not found (pinned md5s above): {', '.join(missing)}",
        )
    for name, md5 in expected.items():
        verify_md5(raw / name, md5)


def extract_archive(archive: str | Path, destination: str | Path) -> list[Path]:
    """Extract safely, rejecting absolute paths and path traversal."""
    archive, destination = Path(archive), Path(destination).resolve()
    if not zipfile.is_zipfile(archive):
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"not a ZIP archive: {archive}")
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = (destination / member.filename).resolve()
                if target != destination and destination not in target.parents:
                    raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"unsafe archive member: {member.filename}")
            zf.extractall(destination)
            extracted = [destination / m.filename for m in zf.infolist() if not m.is_dir()]
    except DatasetError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, str(exc)) from exc
    return extracted


def _digest(path: Path) -> str:
    from .checksum import md5_file
    return md5_file(path).lower()


__all__ = ["expected_files", "extract_archive", "ingest_raw"]
