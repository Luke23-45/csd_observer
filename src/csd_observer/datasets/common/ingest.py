"""Checksum-first raw-file ingestion shared by TAC and DaphniaExt.

Manual mode is offline and succeeds only when every configured file is
already present with its pinned digest. Auto mode requires a token and
still verifies every downloaded file before extraction.

The ``ingest_raw`` driver is safe under repeated invocation:

* a valid existing processed manifest short-circuits to READY_PROCESSED
  *inside* the per-process :class:`FileLock` so concurrent processes do
  not race the cache check;
* on checksum mismatch in auto or manual mode the bad file is moved to
  ``<name>.bad-<unix-ts>`` so subsequent runs don't re-trigger the
  identical failure;
* archive extraction rejects path traversal (incl. absolute entries and
  ``..`` segments) on both POSIX and Windows drives, strips setuid/setgid
  bits, and uses ``zipfile.extractall(..., filter="data")`` on the
  Python interpreters that accept the parameter (3.12 ≤ x < 3.14).
"""
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

# Manual-mode log emitter; set by ``set_run_log`` so manual-drop instructions
# land in the run's structured log rather than in stdout. Falls back to a
# no-op so the dataset pipeline is usable outside a benchmark run.
_log_emitter: Any = None


def set_run_log(emitter: Any | None) -> None:
    """Install a logger used for manual-ingest progress events.

    ``emitter`` exposes ``write(event: str, **fields) -> None``. Pass
    ``None`` to disable.
    """
    global _log_emitter
    _log_emitter = emitter


def _emit(event: str, **fields: Any) -> None:
    if _log_emitter is not None:
        try:
            _log_emitter.write(event, **fields)
        except Exception:
            pass


def expected_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    files = config.get("expected_files")
    if not isinstance(files, list) or not files:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "expected_files must be non-empty")
    return files


def ingest_raw(
    config: dict[str, Any],
    root: str | Path,
    name: str,
    *,
    token: str | None = None,
    force_processing: bool = False,
) -> IngestState:
    """Resolve a dataset into ``<root>/raw/<dir(name)>``; never accepts an unchecked file.

    ``root`` is the data root (e.g. ``datasets``); per-dataset paths are
    derived via the :func:`pipeline.data_dir` mapping so the on-disk
    layout is ``<root>/raw/<name>`` and ``<root>/processed/<name>``.

    ``force_processing`` bypasses the cached-manifest short-circuit: the
    caller (:func:`pipeline.run_pipeline`) already determined the cached
    bundle is stale, so a re-process must not be swallowed by an
    unrelated existing manifest.
    """
    from .pipeline import data_dir

    root = Path(root)
    raw = root / "raw" / data_dir(name)
    processed = root / "processed" / data_dir(name)
    raw.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / f".ingest.{data_dir(name)}.lock"))
    with lock:
        # Second cache check inside the lock: a concurrent process may have
        # populated the manifest while we were waiting.
        if not force_processing and cached_manifest(processed) is not None:
            return IngestState.READY_PROCESSED
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
                fname = str(spec["path"])
                if fname not in remote:
                    raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"remote file missing: {fname}")
                destination = raw / fname
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or _digest(destination) != str(spec["md5"]).lower():
                    client.download(remote[fname], str(destination))
                try:
                    verify_md5(destination, str(spec["md5"]))
                except DatasetError:
                    _quarantine(destination)
                    raise
        else:
            _manual_drop(config, files, raw)
        return IngestState.READY_RAW


def _manual_drop(config: dict[str, Any], files: list[dict[str, Any]], raw: Path) -> None:
    """§5.3 manual mode: poll up to ``download.wait_minutes`` for files.

    Logs the pinned file names + MD5s once at poll start (both via the
    run log and stdout). READY_RAW only on match; a timeout raises
    ``INGEST_CHECKSUM`` listing the still-missing files. Bad files are
    quarantined instead of silently re-validated.

    Entries whose ``md5`` is empty are treated as directory presence
    checks (the extracted raw layout places files under a named
    directory rather than inside a downloadable archive).
    """
    expected = {str(spec["path"]): str(spec["md5"]) for spec in files}
    missing = [name for name, md5 in expected.items() if not (raw / name).exists()]
    if missing:
        instructions = {
            "directory": str(raw),
            "expected": [{"path": name, "md5": md5} for name, md5 in expected.items()],
        }
        _emit("manual_drop_required", **instructions)
        print(f"[ingest] manual mode: drop the following files/dirs into {raw}:")
        for name, md5 in expected.items():
            if md5:
                print(f"  {name}  (md5 {md5})")
            else:
                print(f"  {name}  (directory)")
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
        target = raw / name
        if not md5:
            continue  # directory entry: presence already verified
        try:
            verify_md5(target, md5)
        except DatasetError:
            _quarantine(target)
            raise


def _quarantine(path: Path) -> None:
    """Move a corrupted or mismatched file aside so the next run starts clean."""
    if not path.exists():
        return
    target = path.with_name(f"{path.name}.bad-{int(time.time())}")
    try:
        path.replace(target)
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass


def extract_archive(archive: str | Path, destination: str | Path) -> list[Path]:
    """Extract safely, rejecting absolute paths and path traversal."""
    archive = Path(archive)
    destination = Path(destination).resolve()
    if not zipfile.is_zipfile(archive):
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"not a ZIP archive: {archive}")
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            for member in members:
                if _is_unsafe_member(destination, member):
                    raise DatasetError(
                        DatasetErrorCode.INGEST_ARCHIVE,
                        f"unsafe archive member: {member.filename}",
                    )
            _safe_extract(zf, members, destination)
            extracted = [destination / m.filename for m in members if not m.is_dir()]
    except DatasetError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, str(exc)) from exc
    return extracted


def _is_unsafe_member(destination: Path, member: zipfile.ZipInfo) -> bool:
    """Reject absolute entries, drive-letter entries, and ``..`` traversal."""
    name = member.filename
    if name.startswith("/") or name.startswith("\\"):
        return True
    # Windows-style absolute paths inside the zip: ``C:\foo`` or ``C:/foo``.
    if len(name) >= 3 and name[1] == ":" and name[2] in ("/", "\\"):
        return True
    target = (destination / name).resolve()
    if destination == target:
        return False
    try:
        target.relative_to(destination)
    except ValueError:
        return True
    return False


def _safe_extract(
    zf: zipfile.ZipFile,
    members: list[zipfile.ZipInfo],
    destination: Path,
) -> None:
    """Extract with setuid stripping and (on 3.12-3.13) the ``data`` filter.

    Strips setuid/setgid/sticky bits from extracted entries so an
    untrusted archive cannot elevate privileges. ``filter="data"`` was
    added to ``ZipFile.extractall`` in Python 3.12 (PEP 706) and removed
    in 3.14 — the parameter is only accepted between those releases.
    We probe the signature at runtime so the call is correct on every
    supported interpreter.
    """
    import inspect

    sig = inspect.signature(zf.extractall)
    accepts_filter = "filter" in sig.parameters
    if accepts_filter:
        zf.extractall(destination, members=members, filter="data")
    else:
        zf.extractall(destination, members=members)
    for member in members:
        target = destination / member.filename
        if not target.exists() or member.is_dir():
            continue
        try:
            mode = target.stat().st_mode
        except OSError:
            continue
        # Strip setuid, setgid, and sticky bits. On POSIX an untrusted
        # archive can carry ``0o4755``/``0o2755``/``0o1755`` to attempt
        # privilege elevation; clearing the high three bits leaves the
        # normal rwx triad intact.
        target.chmod(mode & ~0o7000)


def _digest(path: Path) -> str:
    from .checksum import md5_file
    return md5_file(path).lower()


__all__ = ["expected_files", "extract_archive", "ingest_raw", "set_run_log"]
