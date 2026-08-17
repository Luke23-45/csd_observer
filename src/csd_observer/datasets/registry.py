"""Uniform dataset registry; synthetic data is available without I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common.contract import validate_bundle
from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.manifest import read_manifest
from csd_observer.datasets.common.split import replicate_split
from csd_observer.datasets.synthetic.fold import build_signal_null as _fold_bundle
from csd_observer.datasets.synthetic.hopf import build_signal_null as _hopf_bundle
from csd_observer.datasets.synthetic.logistic import build_signal_null as _logistic_bundle

_SYNTHETIC = {
    "synthetic_fold": _fold_bundle,
    "synthetic_hopf": _hopf_bundle,
    "synthetic_logistic": _logistic_bundle,
}


def get_dataset(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = dict(overrides or {})
    if name in _SYNTHETIC:
        bundle = _SYNTHETIC[name](overrides)
        _validate_subset(bundle.get("signal"), f"get_dataset({name})/signal")
        _validate_subset(bundle.get("null"), f"get_dataset({name})/null")
        return bundle
    return _load_processed(name, overrides)


def _validate_subset(subset: Any, context: str) -> None:
    if not isinstance(subset, dict) or "features" not in subset:
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"{context}: malformed subset bundle")
    validate_bundle(subset, context)


def list_datasets() -> list[str]:
    """Backward-compatible alias of :func:`list_resolvable`."""
    return list_resolvable()


def list_resolvable() -> list[str]:
    """Every dataset name that ``get_dataset`` can load right now:
    synthetic fast paths, the two real datasets with a processor, and any
    additional processed datasets already materialized under
    ``<data_root>/processed/``."""
    from csd_observer.datasets.common.pipeline import data_name
    from csd_observer.datasets.provision import REAL_DATASETS

    names = sorted(_SYNTHETIC) + sorted(REAL_DATASETS)
    root = Path("datasets") / "processed"
    if root.is_dir():
        for entry in root.iterdir():
            if (entry / "manifest.json").is_file():
                names.append(data_name(entry.name))
    return sorted(set(names))


def list_provisionable() -> list[str]:
    """Every dataset name with a registered processor: the synthetic
    fast paths (already provisioned by definition) and the two real
    datasets that can be fetched, ingested and processed."""
    from csd_observer.datasets.provision import REAL_DATASETS

    return sorted(_SYNTHETIC) + sorted(REAL_DATASETS)


def _load_processed(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Load a previously processed real dataset; never trigger ingestion."""
    from csd_observer.datasets.common.pipeline import processed_dir

    root = overrides.get("data_root", "datasets")
    processed = processed_dir(root, name)
    manifest = read_manifest(processed / "manifest.json")
    arrays_path = processed / str(manifest.get("arrays_file", "arrays.npz"))
    if not arrays_path.exists():
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"processed arrays missing: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        required = {"features", "seq_lengths", "bifurcation_times", "is_positive"}
        missing = required - set(loaded.files)
        if missing:
            raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, f"processed arrays missing keys: {sorted(missing)}")
        bundle = {key: np.asarray(loaded[key]) for key in required}
    n = len(bundle["features"])
    if bundle["features"].ndim != 3 or bundle["seq_lengths"].shape != (n,):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, "processed feature/length shapes are inconsistent")
    split = manifest.get("split", {})
    indices = split.get("indices")
    if isinstance(indices, dict) and {"train", "val", "test"} <= set(indices):
        bundle["split_indices"] = {k: np.asarray(v, dtype=np.int64) for k, v in indices.items()}
    else:
        bundle["split_indices"] = replicate_split(n, seed=int(split.get("seed", 42)))
    bundle["meta"] = {"bif_type": manifest.get("bif_type", manifest.get("dataset", {}).get("bif_type", "unknown")),
                      "source": manifest.get("dataset", {}).get("source", "processed"),
                      "licence": manifest.get("dataset", {}).get("licence", "unknown"),
                      "n_replicates": n}
    validate_bundle(bundle, f"_load_processed({name})")
    signal_idx = np.flatnonzero(bundle["is_positive"])
    null_idx = np.flatnonzero(~bundle["is_positive"])
    if signal_idx.size == 0 or null_idx.size == 0:
        raise DatasetError(DatasetErrorCode.SPLIT_IMBALANCE, "processed real dataset needs signal and null replicates")
    return {"signal": _subset(bundle, signal_idx), "null": _subset(bundle, null_idx), "meta": bundle["meta"]}


def _subset(bundle: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    out = {k: v[indices] for k, v in bundle.items() if k not in {"split_indices", "meta"}}
    out["split_indices"] = replicate_split(len(indices), seed=42)
    out["meta"] = bundle["meta"].copy()
    validate_bundle(out, "_subset")
    return out


__all__ = ["get_dataset", "list_datasets", "list_provisionable", "list_resolvable"]
