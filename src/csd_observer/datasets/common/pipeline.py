"""End-to-end processed-dataset pipeline driver.

    RESOLVE → FETCH_REMOTE → READY_RAW → PROCESS → SPLIT → READY_PROCESSED

The driver is dataset-agnostic: it receives a processor handle
(``raw_dir -> bundle``), runs mandatory gates, applies replicate-level
splitting, performs leak-free normalization (statistics fit strictly on
the train split only), and writes structured splits:
    ``root/processed/<dataset>/train/train.npz``
    ``root/processed/<dataset>/val/val.npz``
    ``root/processed/<dataset>/test/test.npz`` (if non-empty)
along with split manifests and a root ``manifest.json``.
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
from csd_observer.datasets.common.remote import fetch_remote_dataset
from csd_observer.datasets.common.split import replicate_split
from csd_observer.datasets.common.states import IngestState, cached_manifest

MIN_LENGTH = 100

_BUNDLE_KEYS = ("features", "seq_lengths", "bifurcation_times", "is_positive")

# Registry name -> on-disk data directory name.
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
    """Staleness guard: a changed pipeline invalidates the cache."""
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
    """Return the current HEAD SHA, or empty string when not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def content_hash(split_paths: list[Path], params: dict[str, Any]) -> str:
    """SHA-256 over split array bytes + canonical processing params."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(split_paths):
        if p.is_file():
            with open(p, "rb") as handle:
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
    force_rebuild: bool = False,
) -> IngestState:
    """Drive a dataset from state resolution to a validated processed split structure."""
    root = Path(root)
    raw = raw_dir(root, name)
    processed = processed_dir(root, name)

    # 1. State: Cache check (unless force_rebuild is True)
    if not force_rebuild:
        cached = cached_manifest(processed)
        if cached is not None and _matches_processing(cached, config):
            return IngestState.READY_PROCESSED
    stale = True

    # 2. State: Synthetic fast path or Remote / Hugging Face download attempt
    is_synthetic = str(config.get("source", "unknown")) == "synthetic"
    if is_synthetic:
        state = IngestState.READY_RAW
    else:
        # Check if remote repository download is configured and try it
        download_cfg = config.get("download", {}) or {}
        if download_cfg.get("hf_repo") or config.get("hf_repo"):
            downloaded = fetch_remote_dataset(config, processed, name, token=token)
            if downloaded:
                cached = cached_manifest(processed)
                if cached is not None:
                    return IngestState.READY_PROCESSED

        # 3. State: Fallback to Raw ingestion
        state = ingest_raw(config, root, name, token=token, force_processing=stale)
        if state is IngestState.READY_PROCESSED:
            return state

    if state is not IngestState.READY_RAW:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"unexpected ingest state: {state}")

    # 4. State: Execute Processor
    bundle = processor(raw, config)
    _run_gates(
        bundle,
        extra_validator,
        min_length=int((config.get("processing", {}) or {}).get("min_length", MIN_LENGTH)),
    )

    # 5. State: Split into Train, Val, Test disjoint subsets
    split_cfg = config.get("split", {}) or {}
    seed = int(split_cfg.get("seed", 42))
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))
    indices = replicate_split(
        len(bundle["features"]),
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    counts = {k: int(len(v)) for k, v in indices.items()}

    # 6. State: Fit normalization strictly on Train, apply to all splits
    normalization, split_bundles = _slice_and_normalize(bundle, indices, config)

    # 7. State: Write Split NPZ files and Split Manifests atomically
    processed.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    splits_meta: dict[str, Any] = {}

    for split_name in ("train", "val", "test"):
        idx_arr = indices.get(split_name)
        if idx_arr is None or len(idx_arr) == 0:
            continue
        s_bundle = split_bundles[split_name]
        split_dir = processed / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_npz = split_dir / f"{split_name}.npz"
        _write_arrays_atomic(split_npz, s_bundle)
        written_paths.append(split_npz)

        s_manifest = {
            "split": split_name,
            "n_trajectories": int(len(s_bundle["features"])),
            "n_signal": int(np.asarray(s_bundle["is_positive"]).sum()),
            "n_null": int((~np.asarray(s_bundle["is_positive"], dtype=bool)).sum()),
            "file": f"{split_name}.npz",
        }
        write_manifest(split_dir / "manifest.json", {
            "schema_version": "1.0",
            "dataset": {"name": name},
            "split": s_manifest,
            "gates": {"passed": ["nonfinite", "min_length", "split_balance"]},
            "processing": dict(config.get("processing", {}) or {}),
            "content_hash": content_hash([split_npz], s_manifest),
        })
        splits_meta[split_name] = s_manifest

    # Write root manifest
    manifest = _build_root_manifest(config, bundle, indices, counts, normalization, splits_meta, written_paths)
    write_manifest(processed / "manifest.json", manifest)
    return IngestState.READY_PROCESSED


def _run_gates(
    bundle: dict[str, Any],
    extra_validator: Callable[[dict[str, Any]], None] | None,
    min_length: int = MIN_LENGTH,
) -> None:
    """Mandatory quality gates."""
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


def _slice_and_normalize(
    bundle: dict[str, Any],
    indices: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Slice bundle into splits and apply leak-free normalization fitted on train only."""
    policy = str((config.get("processing", {}) or {}).get("normalization", "none"))
    if policy not in {"none", "zscore"}:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"unsupported normalization policy: {policy!r}")

    # Slice raw splits
    splits: dict[str, dict[str, Any]] = {}
    for part, idx in indices.items():
        if len(idx) == 0:
            continue
        splits[part] = {
            "features": np.asarray(bundle["features"])[idx].copy(),
            "seq_lengths": np.asarray(bundle["seq_lengths"], dtype=np.int64)[idx].copy(),
            "bifurcation_times": np.asarray(bundle["bifurcation_times"], dtype=np.float64)[idx].copy(),
            "is_positive": np.asarray(bundle["is_positive"], dtype=bool)[idx].copy(),
        }

    if policy == "none":
        return {"fit_on_train": True, "policy": "none", "params": None}, splits

    # Z-score: Fit statistics strictly on train
    train_feats = splits["train"]["features"]
    train_lengths = splits["train"]["seq_lengths"]
    mask = np.zeros_like(train_feats, dtype=bool)
    for i, l_val in enumerate(train_lengths):
        mask[i, : int(l_val), :] = True
    flat = train_feats[mask]
    if flat.size == 0:
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, "empty train split for normalization")

    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std > 1e-12, std, 1.0)

    # Standardize all splits
    for part in splits:
        feats = splits[part]["features"]
        splits[part]["features"] = ((feats - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

    normalization_meta = {
        "fit_on_train": True,
        "policy": "zscore",
        "params": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
    }
    return normalization_meta, splits


def _write_arrays_atomic(path: Path, bundle: dict[str, Any]) -> None:
    payload = {key: np.asarray(bundle[key]) for key in _BUNDLE_KEYS}
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)


def _build_root_manifest(
    config: dict[str, Any],
    bundle: dict[str, Any],
    indices: dict[str, np.ndarray],
    counts: dict[str, int],
    normalization: dict[str, Any],
    splits_meta: dict[str, Any],
    split_paths: list[Path],
) -> dict[str, Any]:
    files = [
        {"path": str(spec["path"]), "md5": str(spec["md5"])}
        for spec in config.get("expected_files", [])
    ]
    processing_params = dict(config.get("processing", {}) or {})
    processing_params.setdefault("min_length", MIN_LENGTH)
    if str(config.get("source", "unknown")) == "synthetic":
        for key in (
            "generator", "difficulty", "n_trajectories", "max_length",
            "seed", "null_seed", "noise_scale", "obs_noise_scale", "null",
        ):
            if key in config:
                processing_params.setdefault(key, config[key])

    effective_processing = (bundle.get("meta") or {}).get("processing", {}) or {}
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": {
            "name": config.get("name"),
            "doi": config.get("doi"),
            "version_id": config.get("version_id"),
            "files": files,
            "licence": config.get("licence", "unknown"),
            "retrieved_at": config.get("retrieved_at"),
            "bif_type": config.get("bif_type", "unknown"),
            "source": config.get("source", "unknown"),
            "n_trajectories": int(len(bundle["features"])),
        },
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
            "splits": splits_meta,
        },
        "normalization": normalization,
        "content_hash": content_hash(
            split_paths,
            {"processing": processing_params, "split": counts, "normalization": normalization},
        ),
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
