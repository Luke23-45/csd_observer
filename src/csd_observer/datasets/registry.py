"""Uniform dataset registry and DataModule resolver."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common.contract import validate_bundle
from csd_observer.datasets.common.datamodule import BaseCSDDataModule
from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.manifest import read_manifest
from csd_observer.datasets.common.split import replicate_split
from csd_observer.datasets.daphnia_ext.datamodule import DaphniaDataModule
from csd_observer.datasets.synthetic.datamodule import SyntheticDataModule
from csd_observer.datasets.synthetic.fold import build_signal_null as _fold_bundle
from csd_observer.datasets.synthetic.hopf import build_signal_null as _hopf_bundle
from csd_observer.datasets.synthetic.logistic import build_signal_null as _logistic_bundle
from csd_observer.datasets.tac.datamodule import TACDataModule

_SYNTHETIC = {
    "synthetic_fold": _fold_bundle,
    "synthetic_hopf": _hopf_bundle,
    "synthetic_logistic": _logistic_bundle,
}


def get_datamodule(
    name: str,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseCSDDataModule:
    """Return a dedicated DataModule for the dataset serving both indicators and neural baselines."""
    overrides = dict(overrides or {})
    data_root = overrides.get("data_root", "datasets")
    seed = int(overrides.get("seed", 42))

    if name in _SYNTHETIC:
        return SyntheticDataModule(name=name, config=overrides, data_root=data_root, seed=seed, **kwargs)
    if name == "tac":
        return TACDataModule(config=overrides, data_root=data_root, seed=seed, **kwargs)
    if name in ("daphnia_ext", "daphnia"):
        return DaphniaDataModule(config=overrides, data_root=data_root, seed=seed, **kwargs)

    return BaseCSDDataModule(name=name, config=overrides, data_root=data_root, seed=seed, **kwargs)


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
    """Every dataset name that ``get_dataset`` can load right now."""
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
    """Every dataset name with a registered processor."""
    from csd_observer.datasets.provision import REAL_DATASETS

    return sorted(_SYNTHETIC) + sorted(REAL_DATASETS)


def _load_processed(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Load a processed real dataset supporting both split directories and legacy bundles."""
    from csd_observer.datasets.common.pipeline import processed_dir

    root = overrides.get("data_root", "datasets")
    processed = processed_dir(root, name)
    manifest = read_manifest(processed / "manifest.json")

    # Check for split subdirectories (train, val, test)
    split_dirs = [s for s in ("train", "val", "test") if (processed / s / f"{s}.npz").is_file()]
    if split_dirs:
        feats_list: list[np.ndarray] = []
        lens_list: list[np.ndarray] = []
        bifs_list: list[np.ndarray] = []
        pos_list: list[np.ndarray] = []
        split_indices: dict[str, np.ndarray] = {}
        offset = 0

        for split_name in ("train", "val", "test"):
            s_file = processed / split_name / f"{split_name}.npz"
            if not s_file.is_file():
                continue
            with np.load(s_file, allow_pickle=False) as loaded:
                f_arr = np.asarray(loaded["features"])
                l_arr = np.asarray(loaded["seq_lengths"])
                b_arr = np.asarray(loaded["bifurcation_times"])
                p_arr = np.asarray(loaded["is_positive"])

            n_s = len(f_arr)
            split_indices[split_name] = np.arange(offset, offset + n_s, dtype=np.int64)
            offset += n_s

            feats_list.append(f_arr)
            lens_list.append(l_arr)
            bifs_list.append(b_arr)
            pos_list.append(p_arr)

        # Pad features to global max length if split max lengths differ
        max_t = max(f.shape[1] for f in feats_list)
        c_dim = feats_list[0].shape[2]
        total_n = sum(len(f) for f in feats_list)
        features_full = np.zeros((total_n, max_t, c_dim), dtype=np.float32)

        cur = 0
        for f in feats_list:
            n_rows, t_rows = f.shape[0], f.shape[1]
            features_full[cur : cur + n_rows, :t_rows, :] = f
            cur += n_rows

        bundle = {
            "features": features_full,
            "seq_lengths": np.concatenate(lens_list, axis=0),
            "bifurcation_times": np.concatenate(bifs_list, axis=0),
            "is_positive": np.concatenate(pos_list, axis=0),
            "split_indices": split_indices,
        }
    else:
        # Legacy fallback
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
        split = manifest.get("split", {})
        indices = split.get("indices")
        if isinstance(indices, dict) and {"train", "val", "test"} <= set(indices):
            bundle["split_indices"] = {k: np.asarray(v, dtype=np.int64) for k, v in indices.items()}
        else:
            bundle["split_indices"] = replicate_split(n, seed=int(split.get("seed", 42)))

    n = len(bundle["features"])
    if bundle["features"].ndim != 3 or bundle["seq_lengths"].shape != (n,):
        raise DatasetError(DatasetErrorCode.MANIFEST_CORRUPT, "processed feature/length shapes are inconsistent")

    bundle["meta"] = {
        "bif_type": manifest.get("bif_type", manifest.get("dataset", {}).get("bif_type", "unknown")),
        "source": manifest.get("dataset", {}).get("source", "processed"),
        "licence": manifest.get("dataset", {}).get("licence", "unknown"),
        "n_replicates": n,
    }
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


__all__ = [
    "get_datamodule",
    "get_dataset",
    "list_datasets",
    "list_provisionable",
    "list_resolvable",
]
