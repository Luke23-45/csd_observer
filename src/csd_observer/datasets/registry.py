"""Uniform dataset registry; synthetic data is available without I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
        return _SYNTHETIC[name](overrides)
    return _load_processed(name, overrides)


def list_datasets() -> list[str]:
    """Every registry-resolvable dataset name (synthetic fast paths,
    the two real datasets with a processor, and any additional processed
    datasets already materialized under ``final_data/``)."""
    from pathlib import Path

    from csd_observer.datasets.provision import REAL_DATASETS

    names = sorted(_SYNTHETIC) + sorted(REAL_DATASETS)
    root = Path("final_data")
    if root.is_dir():
        names.extend(sorted(p.name for p in root.iterdir() if (p / "processed" / "manifest.json").is_file()))
    return sorted(set(names))


def _load_processed(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Load a previously processed real dataset; never trigger ingestion."""
    root = overrides.get("data_root", "final_data")
    dataset_root = Path(root) / name
    processed = dataset_root / "processed"
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
    signal_idx = np.flatnonzero(bundle["is_positive"])
    null_idx = np.flatnonzero(~bundle["is_positive"])
    if signal_idx.size == 0 or null_idx.size == 0:
        raise DatasetError(DatasetErrorCode.SPLIT_IMBALANCE, "processed real dataset needs signal and null replicates")
    return {"signal": _subset(bundle, signal_idx), "null": _subset(bundle, null_idx), "meta": bundle["meta"]}


def _subset(bundle: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    out = {k: v[indices] for k, v in bundle.items() if k not in {"split_indices", "meta"}}
    out["split_indices"] = replicate_split(len(indices), seed=42)
    out["meta"] = bundle["meta"].copy()
    return out


__all__ = ["get_dataset", "list_datasets"]
