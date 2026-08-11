"""Synthetic data: uniform registry bundles with provenance (§5.6 of the plan).

Per-system wrappers (``datasets/synthetic/{fold,hopf,logistic}``) call
:func:`build_bundle` with their system key; provenance records the
generator tier, difficulty, and the generator parameters so every run can
be reproduced exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common import IngestState
from csd_observer.datasets.synthetic.common.generators import (
    _DEFAULT_DRIVE_NOISE,
    _DEFAULT_OBS_NOISE,
    RandomizedBifurcationDataset,
    build_dataset,
)

_VALID_SYSTEMS = ("fold", "hopf", "logistic")

# Parameters that are part of a bundle's provenance (beyond the registry
# contract keys) and are recorded into ``meta["params"]``. ``null_seed``
# is popped before the generator call (it is not a generator argument).
_PARAM_KEYS = ("n_trajectories", "max_length", "noise_scale", "obs_noise_scale",
               "seed", "null_seed", "generator", "difficulty", "null")


def build_bundle(
    system: str,
    *,
    seed: int = 42,
    null: bool = False,
    n_trajectories: int = 500,
    max_length: int = 200,
    noise_scale: float = 0.30,
    obs_noise_scale: float | None = None,
    generator: str = "classic",
    difficulty: str = "standard",
) -> dict[str, Any]:
    """Build one synthetic bundle: ``{"features", "bifurcation_times",
    "is_positive", "seq_lengths", "split_indices", "meta"}``.

    ``meta`` carries ``bif_type``, ``source="synthetic"``,
    ``licence="project-generated"``, ``n_replicates`` and full provenance.
    """
    if system not in _VALID_SYSTEMS:
        raise ValueError(f"Unknown system: {system!r}. Valid options: {sorted(_VALID_SYSTEMS)}")
    kwargs: dict[str, Any] = dict(
        n_trajectories=n_trajectories, max_length=max_length,
        seed=seed, null=null, generator=generator, difficulty=difficulty,
    )
    if obs_noise_scale is not None:
        kwargs["obs_noise_scale"] = obs_noise_scale
    bundle = build_dataset(system, **kwargs)
    effective_obs = obs_noise_scale if obs_noise_scale is not None else _DEFAULT_OBS_NOISE[system]
    effective_drive = noise_scale
    params: dict[str, Any] = dict(
        n_trajectories=int(n_trajectories), max_length=int(max_length),
        noise_scale=float(effective_drive), obs_noise_scale=float(effective_obs),
        seed=int(seed), generator=str(generator), difficulty=str(difficulty),
        null=bool(null),
    )
    bundle["meta"] = {
        "bif_type": system,
        "source": "synthetic",
        "licence": "project-generated",
        "n_replicates": int(bundle["features"].shape[0]),
        "generator": str(generator),
        "difficulty": str(difficulty),
        "params": params,
    }
    return bundle


def ensure_synthetic_processed(config: dict[str, Any], root: str | Path) -> IngestState:
    """Materialize ``root/processed`` via the §5.4 processed-dir pipeline.

    Same manifest contract as the real datasets (ledger L3.15): identical
    gates, replicate split, content hash. The generated arrays are
    bit-identical to the registry's in-memory fast path (same generator
    calls, same seeds), so runs behave the same with or without the
    materialized directory.
    """
    from csd_observer.datasets.common.pipeline import run_pipeline

    return run_pipeline(config, root, _synthetic_processor)


def _synthetic_processor(raw_dir: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Combined signal + null bundle with provenance (§5.6 contract)."""
    system = str(config.get("bif_type", "unknown")).lower()
    if system not in _VALID_SYSTEMS:
        raise ValueError(f"Unknown synthetic system: {system!r}. Valid options: {sorted(_VALID_SYSTEMS)}")
    overrides = {k: v for k, v in config.items() if k in _PARAM_KEYS}
    seed = int(overrides.pop("seed", 42))
    null_seed = int(overrides.pop("null_seed", seed + 1))
    overrides.pop("null", None)
    signal = build_bundle(system, seed=seed, null=False, **overrides)
    null = build_bundle(system, seed=null_seed, null=True, **overrides)
    features = np.concatenate([signal["features"], null["features"]], axis=0)
    n_sig = features.shape[0] - null["features"].shape[0]
    return {
        "features": features.astype(np.float32),
        "seq_lengths": np.concatenate([signal["seq_lengths"], null["seq_lengths"]]).astype(np.int64),
        "bifurcation_times": np.concatenate([signal["bifurcation_times"], null["bifurcation_times"]]).astype(np.float64),
        "is_positive": np.concatenate([np.ones(n_sig, dtype=bool), np.zeros(null["features"].shape[0], dtype=bool)]),
        "meta": signal["meta"],
    }


__all__ = ["build_bundle", "build_dataset", "RandomizedBifurcationDataset",
           "ensure_synthetic_processed",
           "_DEFAULT_DRIVE_NOISE", "_DEFAULT_OBS_NOISE", "_PARAM_KEYS"]
