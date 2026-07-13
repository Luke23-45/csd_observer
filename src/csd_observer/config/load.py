from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "configs"


def load_config(
    run_name: str,
    config_root: str | Path | None = None,
) -> Dict[str, Any]:
    if config_root is None:
        config_root = _CONFIG_ROOT
    config_root = Path(config_root)

    run_path = config_root / "run" / f"{run_name}.yaml"
    if not run_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_path}")

    with open(run_path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    meta_refs = {k: raw.get(k) for k in ("_data", "_model", "_training")}
    overrides = {k: raw.get(k, {}) for k in ("data", "model", "training")}

    def _load_ref(subdir: str, ref: str | None) -> dict:
        if ref is None:
            return {}
        path = config_root / subdir / f"{ref}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config template not found: {path}")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    data_cfg = _load_ref("data", meta_refs["_data"])
    model_cfg = _load_ref("model", meta_refs["_model"])
    training_cfg = _load_ref("training", meta_refs["_training"])

    data_cfg.update(overrides["data"])
    model_cfg.update(overrides["model"])
    training_cfg.update(overrides["training"])

    _validate_config(data_cfg, model_cfg, training_cfg)

    return {
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
    }


_REQUIRED_DATA_KEYS = {"noise_scale", "n_patients", "systems", "max_length", "n_seeds"}
_REQUIRED_MODEL_KEYS = {"latent_dim", "lstm_dim"}
_REQUIRED_TRAINING_KEYS = {"epochs", "batch_size", "lr", "patience", "spectral_radius_weight", "spectral_threshold"}


def _validate_config(data: dict, model: dict, training: dict) -> None:
    missing_data = _REQUIRED_DATA_KEYS - data.keys()
    if missing_data:
        raise ValueError(f"Config missing required data keys: {missing_data}")
    missing_model = _REQUIRED_MODEL_KEYS - model.keys()
    if missing_model:
        raise ValueError(f"Config missing required model keys: {missing_model}")
    missing_training = _REQUIRED_TRAINING_KEYS - training.keys()
    if missing_training:
        raise ValueError(f"Config missing required training keys: {missing_training}")
    for key in ("noise_scale", "max_length", "n_patients"):
        val = data[key]
        if not isinstance(val, (int, float)):
            raise TypeError(f"Config data.{key} must be numeric, got {type(val).__name__}")
    for key in ("latent_dim", "lstm_dim"):
        val = model[key]
        if not isinstance(val, int):
            raise TypeError(f"Config model.{key} must be int, got {type(val).__name__}")
    for key in ("epochs", "batch_size", "patience"):
        val = training[key]
        if not isinstance(val, int):
            raise TypeError(f"Config training.{key} must be int, got {type(val).__name__}")
    for key in ("lr", "spectral_radius_weight", "spectral_threshold"):
        val = training[key]
        if not isinstance(val, (int, float)):
            raise TypeError(f"Config training.{key} must be numeric, got {type(val).__name__}")
