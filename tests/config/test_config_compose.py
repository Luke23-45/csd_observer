"""L6.6: compose tests for the Hydra run config.

Every documented run composes (all datasets x both evaluations), and
the §10.3 invariants are enforced: bad keys and missing groups fail at
composition time; unknown methods and learned-methods-without-training
fail in ``validate_config``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException, MissingConfigException
from omegaconf import OmegaConf

from csd_observer.config.store import register_configs
from csd_observer.config.validate import validate_config

_REPO = Path(__file__).resolve().parent.parent.parent

# Hydra resolves a relative config_path against the calling file's
# directory, not the cwd — so compute it from the test file location.
_CONFIG_PATH = os.path.relpath(_REPO / "configs", Path(__file__).resolve().parent)

DATASETS = [
    "synthetic_fold",
    "synthetic_hopf",
    "synthetic_logistic",
    "tac",
    "daphnia_ext",
]

EVALUATIONS = ["persistenceaware", "baseline_classic"]


@contextmanager
def _repo_cwd() -> Iterator[None]:
    cwd = os.getcwd()
    os.chdir(_REPO)
    try:
        yield
    finally:
        os.chdir(cwd)


def _compose(overrides: list[str] | None = None) -> dict[str, Any]:
    GlobalHydra.instance().clear()
    register_configs()
    with _repo_cwd(), initialize(version_base=None, config_path=_CONFIG_PATH):
        cfg = compose(config_name="run", overrides=overrides or [])
    return OmegaConf.to_container(cfg, resolve=True)


def test_default_run_composes() -> None:
    config = _compose()
    assert set(config) >= {
        "dataset", "model", "training", "evaluation", "output",
        "models", "seed", "seed_offset", "n_seeds", "dataset_overrides",
    }
    assert config["dataset"]["name"] == "synthetic_fold"
    assert config["evaluation"]["name"] == "persistenceaware"
    assert config["models"] == ["VAR-CSD"]
    assert config["training"]["enabled"] is True


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("evaluation", EVALUATIONS)
def test_documented_run_composes(dataset: str, evaluation: str) -> None:
    config = _compose(
        [f"dataset={dataset}", f"evaluation={evaluation}"]
    )
    assert config["dataset"]["name"] == dataset
    assert config["evaluation"]["name"] == evaluation


@pytest.mark.parametrize("evaluation", EVALUATIONS)
def test_synthetic_runs_validate(evaluation: str) -> None:
    config = _compose([f"evaluation={evaluation}"])
    validate_config(config)


@pytest.mark.parametrize("dataset", ["tac", "daphnia_ext"])
def test_real_datasets_require_processed_data(dataset: str) -> None:
    config = _compose([f"dataset={dataset}"])
    with pytest.raises(ValueError, match="unknown dataset"):
        validate_config(config)


def test_model_group_selects_block() -> None:
    config = _compose(["model=skew_csd"])
    assert config["model"]["skew_csd"]["window_size"] == 30
    assert config["model"]["var_csd"] is None
    assert config["model"]["spectral_drift"] is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["dataset.unknown_key=1"], "Could not override"),
        (["model.bogus=2"], "Could not override"),
    ],
)
def test_bad_key_rejected_at_compose(overrides: list[str], message: str) -> None:
    with pytest.raises(ConfigCompositionException, match=message):
        _compose(overrides)


@pytest.mark.parametrize(
    "overrides",
    [["dataset=missing"], ["model=missing"], ["evaluation=missing"],
     ["training=missing"], ["output=missing"]],
)
def test_missing_group_rejected_at_compose(overrides: list[str]) -> None:
    with pytest.raises(MissingConfigException):
        _compose(overrides)


def test_unknown_method_rejected() -> None:
    config = _compose(["models=[NOT-A-METHOD]"])
    with pytest.raises(ValueError, match="NOT-A-METHOD"):
        validate_config(config)


def test_learned_method_with_training_none_rejected() -> None:
    config = _compose(["models=[LSTM-AlarmNet]", "training=none"])
    with pytest.raises(ValueError, match="learned"):
        validate_config(config)


def test_learned_method_with_training_ok() -> None:
    config = _compose(["models=[LSTM-AlarmNet]", "training=default"])
    validate_config(config)
    assert config["training"]["enabled"] is True


def test_empty_models_rejected() -> None:
    config = _compose(["models=[]"])
    with pytest.raises(ValueError, match="at least one"):
        validate_config(config)


def test_dataset_overrides_whitelist() -> None:
    # `+` required: the composed config is struct, so nested override
    # keys must be added, not assigned into the empty dict.
    config = _compose(["+dataset_overrides.n_trajectories=24"])
    validate_config(config)
    assert config["dataset_overrides"]["n_trajectories"] == 24
    bad = _compose(["+dataset_overrides.bogus=1"])
    with pytest.raises(ValueError, match="whitelist"):
        validate_config(bad)


def test_seed_schedule_knobs() -> None:
    config = _compose(["seed=7", "seed_offset=1000", "n_seeds=3"])
    assert config["seed"] == 7
    assert config["seed_offset"] == 1000
    assert config["n_seeds"] == 3
    validate_config(config)
