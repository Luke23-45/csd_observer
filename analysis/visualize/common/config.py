"""
Path configuration for benchmark visualization.

The experiment data lives under ``analysis/experiment_data`` and the
visualization outputs are written under ``analysis/visualize/outputs``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[DictConfig] = None
_CONFIGS_DIR: Path = Path(__file__).resolve().parent.parent / "configs"
_DEFAULT_CONFIG: Path = _CONFIGS_DIR / "base.yaml"


def _detect_project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def _detect_data_root() -> Path:
    """Return the benchmark experiment-data directory.

    Prefers the current benchmark suite output (repo-root ``outputs/``)
    when it contains runs; falls back to the legacy
    ``analysis/experiment_data`` location.
    """
    project_root = _detect_project_root()
    new_root = project_root / "outputs"
    if any(new_root.glob("**/results.jsonl")):
        return new_root
    return project_root / "analysis" / "experiment_data"


def get_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    use_cache: bool = True,
) -> DictConfig:
    """Load and optionally cache the visualization config."""
    global _CONFIG_CACHE

    if use_cache and _CONFIG_CACHE is not None and overrides is None:
        return _CONFIG_CACHE

    cfg_path = config_path or _DEFAULT_CONFIG
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
    else:
        cfg = OmegaConf.create(
            {
                "paths": {
                    "project_root": None,
                    "data_root": None,
                    "output_root": "analysis/visualize/outputs",
                }
            }
        )

    if cfg.paths.project_root is None:
        OmegaConf.update(cfg, "paths.project_root", str(_detect_project_root().resolve()))
    if cfg.paths.data_root is None:
        OmegaConf.update(cfg, "paths.data_root", str(_detect_data_root().resolve()))

    if overrides:
        override_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()])
        cfg = OmegaConf.merge(cfg, override_cfg)

    if overrides is None:
        _CONFIG_CACHE = cfg

    logger.debug("Configuration loaded: %s", cfg)
    return cfg


def get_data_root(cfg: Optional[DictConfig] = None) -> Path:
    """Return the resolved experiment-data root."""
    if cfg is None:
        cfg = get_config()
    return Path(cfg.paths.data_root)


def get_output_root(cfg: Optional[DictConfig] = None) -> Path:
    """Return the resolved output directory for figures and tables."""
    if cfg is None:
        cfg = get_config()
    project_root = Path(cfg.paths.project_root)
    out_dir = project_root / cfg.paths.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
