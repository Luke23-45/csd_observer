"""End-to-end processed-dataset pipeline driver (§5.2–§5.5 of the plan).

    RESOLVE → … → READY_RAW → PROCESS → SPLIT → READY_PROCESSED

The driver is dataset-agnostic: it receives a processor handle
(``raw_dir -> bundle``), runs the §5.4 mandatory gates, applies the
replicate-level split, performs leak-free normalization (statistics fit
on the train split only), and writes ``arrays.npz`` + ``manifest.json``
atomically under ``root/processed``.

Dataset-specific processors (TAC/DaphniaExt) live in their own
subpackages and are wired in at the orchestration layer; the synthetic
processor ships here so the synthetic manifest pipeline (ledger L3.15)
works end to end without network or raw files.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.ingest import ingest_raw
from csd_observer.datasets.common.manifest import write_manifest
from csd_observer.datasets.common.split import replicate_split
from csd_observer.datasets.common.states import IngestState, cached_manifest

# §5.4 gate 2: every trajectory must reach the longest indicator window
# (DFA). Trajectories below this are excluded with logged counts, never
# dropped silently.
MIN_LENGTH = 100

_BUNDLE_KEYS = ("features", "seq_lengths", "bifurcation_times", "is_positive")

# Default arrays file name; pinned so consumers can locate the bundle
# without scanning the manifest for an ``arrays_file`` field.
ARRAYS_FILE = "arrays.npz"

# Registry name -> on-disk data directory name. A single source of truth so
# the raw/ and processed/ folders on disk match the raw layout the user
# placed under ``datasets/raw/`` (the registry name may differ).
_DATA_DIR = {"daphnia_ext": "daphnia"}


def data_dir(name: str) -> str:
    """On-disk folder name for a dataset (defaults to the registry name)."""
    return _DATA_DIR.get(str(name), str(name))


def data_name(folder: str) -> str:
    """Registry name for an on-disk folder (defaults to the folder name)."""
    reverse = {v: k for k, v in _DATA_DIR.items()}
    return reverse.get(str(folder), str(folder))


def raw_dir(data_root: str | Path, name: str) -> Path:
    """Per-dataset raw directory under the data root."""
    return Path(data_root) / "raw" / data_dir(name)


def processed_dir(data_root: str | Path, name: str) -> Path:
    """Per-dataset processed directory under the data root."""
    return Path(data_root) / "processed" / data_dir(name)


def _matches_processing(manifest: dict[str, Any], config: dict[str, Any]) -> bool:
    """§13 staleness guard: a changed pipeline invalidates the cache.

    The manifest's ``processing.params`` are compared against the current
    config (with ``min_length`` defaulted). Normalization policy is also
    considered (it lives at ``manifest.normalization``, not under
    ``processing``, but a change in policy still invalidates the cache
    because it changes the on-disk array bytes). The recorded ``git_sha``
    must also match HEAD so a processor-code change invalidates the
    cache (provisioning re-runs against the new implementation).
    """
    recorded = (manifest.get("processing", {}) or {}).get("params", {})
    current = dict(config.get("processing", {}) or {})
    current.setdefault("min_length", MIN_LENGTH)
    if recorded != current:
        return False
    recorded_norm = (manifest.get("normalization", {}) or {}).get("policy")
    current_norm = str((config.get("processing", {}) or {}).get("normalization", "none"))
    if recorded_norm != current_norm:
        return False
    recorded_sha = (manifest.get("processing", {}) or {}).get("git_sha")
    return recorded_sha == _git_sha()


def _git_sha() -> str:
    """Return the current HEAD SHA, or ``""`` when not in a git repo.

    Consistent with :func:`csd_observer.outputs.metadata._git_sha` so
    the dataset manifest's ``git_sha`` and the run environment's
    ``git_sha`` agree. Empty string is the unambiguous sentinel for
    "no git available"; downstream consumers can treat it as such.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def content_hash(arrays_path: Path, params: dict[str, Any]) -> str:
    """§5.5 ``content_hash``: sha256 over array bytes + canonical params.

    Trust model: the hash binds the on-disk array bytes and the
    ``processing.params`` + split counts + normalization parameters. It
    does *not* cover ``gates.passed`` or ``split.indices`` — those are
    derived from the bundle and the params, so tampering with them while
    preserving the bundle implies recomputing or guessing the params.
    An adversary who edits ``gates.passed`` in the manifest keeps a
    "valid" hash; this is acceptable because ``gates.passed`` is a
    derived self-attestation, not a security boundary.
    """
    import hashlib

    h = hashlib.sha256()
    with open(arrays_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    h.update(b"\0")
    h.update(json.dumps(params, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def run_pipeline(
    config: dict[str, Any],
    root: str | Path,
    name: str,
    processor: Callable[[Path, dict[str, Any]], dict[str, Any]],
    *,
    token: str | None = None,
    extra_validator: Callable[[dict[str, Any]], None] | None = None,
) -> IngestState:
    """Drive a dataset from resolution to a validated processed manifest.

    Idempotent: a valid existing manifest short-circuits to
    ``READY_PROCESSED``. Every step is checksum-first; failures raise
    :class:`DatasetError` with the §5.2 error codes and never leave a
    half-written manifest (arrays + manifest are written atomically).

    ``source == "synthetic"`` skips raw ingestion entirely (no raw
    files exist); the processor builds the bundle from the generators.

    ``root`` is the data root (e.g. ``datasets``); per-dataset raw and
    processed directories are derived via :func:`raw_dir` /
    :func:`processed_dir` so the layout is
    ``<root>/raw/<name>`` and ``<root>/processed/<name>``.
    """
    root = Path(root)
    raw = raw_dir(root, name)
    processed = processed_dir(root, name)
    cached = cached_manifest(processed)
    if cached is not None and _matches_processing(cached, config):
        return IngestState.READY_PROCESSED
    # A cached manifest that no longer matches forces re-processing:
    # without this, ``ingest_raw``'s own short-circuit would return the
    # stale bundle as READY_PROCESSED and the guard above would never
    # trigger for real datasets (only ``source == "synthetic"`` paths,
    # which skip raw ingestion, re-provisioned before this).
    stale = cached is not None

    if str(config.get("source", "unknown")) == "synthetic":
        state = IngestState.READY_RAW
    else:
        state = ingest_raw(config, root, name, token=token, force_processing=stale)
        if state is IngestState.READY_PROCESSED:
            return state
    if state is not IngestState.READY_RAW:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"unexpected ingest state: {state}")

    bundle = processor(raw, config)
    _run_gates(bundle, extra_validator, min_length=int((config.get("processing", {}) or {}).get("min_length", MIN_LENGTH)))

    split_cfg = config.get("split", {}) or {}
    seed = int(split_cfg.get("seed", 42))
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))
    indices = replicate_split(len(bundle["features"]), seed=seed, train_frac=train_frac, val_frac=val_frac)
    counts = {k: int(len(v)) for k, v in indices.items()}

    normalization = _fit_normalization(bundle, indices["train"], config)

    processed.mkdir(parents=True, exist_ok=True)
    arrays_path = processed / ARRAYS_FILE
    _write_arrays_atomic(arrays_path, bundle)

    manifest = _build_manifest(config, bundle, indices, counts, normalization, arrays_path)
    write_manifest(processed / "manifest.json", manifest)
    return IngestState.READY_PROCESSED


def _run_gates(
    bundle: dict[str, Any],
    extra_validator: Callable[[dict[str, Any]], None] | None,
    min_length: int = MIN_LENGTH,
) -> None:
    """§5.4 mandatory gates; dataset-specific gates run via the hook.

    ``min_length`` (R0.4) defaults to the DFA gate but can be lowered by
    the dataset's ``processing.min_length`` for real datasets whose
    trajectory lengths legitimately fall below 100 steps (the DFA method
    is in that case simply not included in the run)."""
    features = np.asarray(bundle.get("features"))
    seq_lengths = np.asarray(bundle.get("seq_lengths"), dtype=np.int64)
    is_positive = np.asarray(bundle.get("is_positive"), dtype=bool)
    if features.ndim != 3 or seq_lengths.shape != (features.shape[0],):
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, "bundle features must be (B, T, C) with seq_lengths (B,)")
    if is_positive.shape != (features.shape[0],):
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, "is_positive must be (B,)")
    if not np.isfinite(features).all():
        raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, "features contain non-finite values")

    short = np.flatnonzero(seq_lengths < min_length)
    if short.size:
        raise DatasetError(
            DatasetErrorCode.PROCESS_SHORT_LENGTH,
            f"{short.size} trajectory/ies shorter than {min_length} steps: policy is exclude-with-counts, not silent drop",
        )

    n_signal = int(is_positive.sum())
    n_null = int((~is_positive).sum())
    if n_signal == 0 or n_null == 0:
        raise DatasetError(
            DatasetErrorCode.SPLIT_IMBALANCE,
            f"processed set needs signal and null replicates, got {n_signal} signal / {n_null} null",
        )

    if extra_validator is not None:
        extra_validator(bundle)


def _fit_normalization(bundle: dict[str, Any], train_idx: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Leak-free per-channel z-score: statistics fit on train only.

    Policy is per-dataset (recorded in the manifest): ``none`` leaves
    features untouched (default; keeps the registry's synthetic fast
    path and the indicators' own detrending bit-for-bit identical);
    ``zscore`` standardizes each channel with train-only statistics.
    """
    policy = str((config.get("processing", {}) or {}).get("normalization", "none"))
    if policy not in {"none", "zscore"}:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"unsupported normalization policy: {policy!r}")
    if policy == "none":
        return {"fit_on_train": True, "policy": "none", "params": None}
    features = np.asarray(bundle["features"])
    train = features[train_idx]
    valid = np.asarray(bundle["seq_lengths"], dtype=np.int64)
    mask = np.zeros_like(train, dtype=bool)
    for i, length in enumerate(train_idx):
        mask[i, : int(valid[length]), :] = True
    flat = train[mask]
    if flat.size == 0:
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, "empty train split for normalization")
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    normalized = (features - mean[None, None, :]) / std[None, None, :]
    bundle["features"] = normalized.astype(np.float32)
    return {
        "fit_on_train": True,
        "policy": "zscore",
        "params": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
    }


def _write_arrays_atomic(path: Path, bundle: dict[str, Any]) -> None:

    payload = {key: np.asarray(bundle[key]) for key in _BUNDLE_KEYS}
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)


def _build_manifest(
    config: dict[str, Any],
    bundle: dict[str, Any],
    indices: dict[str, np.ndarray],
    counts: dict[str, int],
    normalization: dict[str, Any],
    arrays_path: Path,
) -> dict[str, Any]:
    files = [
        {"path": str(spec["path"]), "md5": str(spec["md5"])}
        for spec in config.get("expected_files", [])
    ]
    processing_params = dict(config.get("processing", {}) or {})
    processing_params.setdefault("min_length", MIN_LENGTH)
    if str(config.get("source", "unknown")) == "synthetic":
        for key in ("generator", "difficulty", "n_trajectories", "max_length",
                    "seed", "null_seed", "noise_scale", "obs_noise_scale", "null"):
            if key in config:
                processing_params.setdefault(key, config[key])
    # Effective (derived) processing provenance from the processor — e.g.
    # auto-selected channels, README annotations. Recorded separately so
    # ``_matches_processing`` compares only the user-controlled keys.
    effective_processing = (bundle.get("meta") or {}).get("processing", {}) or {}
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": {
            "doi": config.get("doi"),
            "version_id": config.get("version_id"),
            "files": files,
            "licence": config.get("licence", "unknown"),
            "retrieved_at": config.get("retrieved_at"),
            "bif_type": config.get("bif_type", "unknown"),
            "source": config.get("source", "unknown"),
            "n_trajectories": int(len(bundle["features"])),
        },
        "arrays_file": ARRAYS_FILE,
        "processing": {
            "git_sha": _git_sha(),
            "params": processing_params,
            "package_version": _package_version(),
            "effective": effective_processing,
        },
        "gates": {
            "passed": [
                "nonfinite",
                "min_length",
                "split_balance",
            ],
            "counts": {
                "signal": int(np.asarray(bundle["is_positive"]).sum()),
                "null": int((~np.asarray(bundle["is_positive"], dtype=bool)).sum()),
            },
        },
        "split": {
            "policy": "replicate_based",
            "seed": int(config.get("split", {}).get("seed", 42)),
            "counts": counts,
            "indices": {k: [int(v) for v in idx] for k, idx in indices.items()},
        },
        "normalization": normalization,
        "content_hash": content_hash(arrays_path, {"processing": processing_params, "split": counts, "normalization": normalization}),
    }
    return manifest


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("csd_observer")
    except Exception:
        return "unknown"


__all__ = [
    "MIN_LENGTH",
    "content_hash",
    "data_dir",
    "data_name",
    "processed_dir",
    "raw_dir",
    "run_pipeline",
]
