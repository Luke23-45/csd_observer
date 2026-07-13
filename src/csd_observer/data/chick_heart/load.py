from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from .download import download_csv
from .prepare import prepare
from .transition import detect_transitions


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _data_dir() -> Path:
    return _project_root() / "data" / "chick_heart"


def _load_module_config() -> dict:
    path = Path(__file__).resolve().parent / "_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path))
    return {
        "features": data["features"],
        "seq_lengths": data["seq_lengths"],
        "bifurcation_times": data["bifurcation_times"],
        "is_positive": data["is_positive"],
        "split_indices": {
            "train": data["train_idx"],
            "val": data["val_idx"],
            "test": data["test_idx"],
        },
    }


def build_chick_heart_dataset(
    *,
    force_download: bool = False,
    force_transitions: bool = False,
    force_prepare: bool = False,
) -> Dict[str, np.ndarray]:
    cfg = _load_module_config()
    data_dir = _data_dir()
    csv_path = data_dir / "df_chick.csv"
    transitions_path = data_dir / "df_transitions.pkl"
    dataset_path = data_dir / "dataset.npz"

    if dataset_path.exists() and not force_prepare:
        return _load_npz(dataset_path)

    if transitions_path.exists() and not force_transitions:
        with open(transitions_path, "rb") as f:
            transitions = pickle.load(f)
    else:
        if not csv_path.exists() or force_download:
            download_csv(
                url=cfg["source_url"],
                data_dir=data_dir,
                force=force_download,
            )

        df = pd.read_csv(csv_path)
        transitions = detect_transitions(
            df,
            bandwidth=cfg["detection"]["bandwidth"],
            rolling_window=cfg["detection"]["rolling_window"],
            slope_threshold=cfg["detection"]["slope_threshold"],
            consecutive_beats=cfg["detection"]["consecutive_beats"],
        )
        with open(transitions_path, "wb") as f:
            pickle.dump(transitions, f)

    df = pd.read_csv(csv_path)
    split_cfg = cfg["split"]
    dataset = prepare(
        df, transitions,
        max_length=cfg["max_length"],
        split_seed=split_cfg["seed"],
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        out_dir=data_dir,
    )
    return dataset
