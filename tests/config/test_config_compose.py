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
        "models", "seed_offset", "n_seeds", "dataset_overrides",
    }
    # ``seed`` was removed from the run config (R3.2): the §13 schedule
    # is the single seed source.
    assert "seed" not in config
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
def test_real_datasets_validate_and_require_processed_data(dataset: str, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Real dataset names are registry-known, so validation accepts them;
    missing processed data surfaces at load time (provisioning), not at
    config validation."""
    # daphnia_ext lowers processing.min_length to 20 (real data has only
    # ~30-60 census days per replicate); the validate-time >=100 DFA gate
    # is bypassed with the documented CI/smoke escape hatch, exactly like
    # the synthetic smoke runs.
    monkeypatch.setenv("CSD_OBSERVER_SKIP_MIN_LENGTH_GATES", "1")
    config = _compose([f"dataset={dataset}"])
    validate_config(config)
    assert config["dataset"]["name"] == dataset
    expected = 100 if dataset == "tac" else 20
    assert config["dataset"]["processing"]["min_length"] == expected

    from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
    from csd_observer.datasets.registry import get_dataset

    with pytest.raises(DatasetError) as exc_info:
        get_dataset(dataset, {"data_root": str(tmp_path)})
    assert exc_info.value.code == DatasetErrorCode.MANIFEST_CORRUPT


def test_model_group_selects_block() -> None:
    config = _compose(["model=skew_csd"])
    assert config["model"]["skew_csd"]["window_size"] == 30
    assert "var_csd" not in config["model"]
    assert "lstm" not in config["model"]


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


def test_learned_label_window_invariant_fires_for_daphnia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2.4: daphnia min_length=20 makes the default label_window=60 mark
    whole trajectories positive (saturated supervision). The invariant is
    a supervision check, so it fires even under the min-length-gate
    bypass; a dataset-appropriate label_window passes."""
    monkeypatch.setenv("CSD_OBSERVER_SKIP_MIN_LENGTH_GATES", "1")
    bad = _compose(["dataset=daphnia_ext", "models=[LSTM-AlarmNet]"])
    with pytest.raises(ValueError, match="label_window"):
        validate_config(bad)
    ok = _compose(
        ["dataset=daphnia_ext", "models=[LSTM-AlarmNet]", "training.label_window=10"]
    )
    validate_config(ok)


def test_tac_aligned_ramp_window_early_window_invariant() -> None:
    """R2.4: with chunking, TAC aligned ramp windows host the onset at
    ramp_pre_samples; an early_start_delta beyond it would clip every
    early-warning window to nothing and must fail validation."""
    cfg = _compose(["dataset=tac", "evaluation.early_start_delta=3000"])
    with pytest.raises(ValueError, match="ramp_pre_samples"):
        validate_config(cfg)
    ok = _compose(["dataset=tac"])
    validate_config(ok)


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


def test_dataset_group_generator_knob_rejected() -> None:
    # R0.1: ``dataset.n_trajectories=16`` composes (group override) but
    # is a silent no-op — validation must reject it loudly.
    config = _compose(["dataset.n_trajectories=16"])
    assert config["dataset"]["n_trajectories"] == 16
    with pytest.raises(ValueError, match="dataset_overrides"):
        validate_config(config)


def test_null_seed_is_not_a_whitelisted_override() -> None:
    # R0.5 (option a): the schedule is the single seed source; a user
    # attempt to pin ``null_seed`` via the override channel fails.
    bad = _compose(["+dataset_overrides.null_seed=123"])
    with pytest.raises(ValueError, match="whitelist"):
        validate_config(bad)


def test_seed_schedule_knobs() -> None:
    # R3.2: ``seed`` is no longer a run knob; the schedule derives from
    # ``seed_offset``/``n_seeds`` alone.
    config = _compose(["seed_offset=1000", "n_seeds=3"])
    assert "seed" not in config
    assert config["seed_offset"] == 1000
    assert config["n_seeds"] == 3
    validate_config(config)


def test_resolved_yaml_round_trips() -> None:
    # R3.4: the composed config (minus hydra plumbing) survives a
    # write-to-yaml → re-compose cycle with an identical hash.
    from hydra import compose as hydra_compose
    from hydra import initialize as hydra_initialize

    config = _compose(["+dataset_overrides.n_trajectories=24"])
    sans_hydra = {k: v for k, v in config.items() if k != "hydra"}
    round_trip_dir = _REPO / "outputs" / "_round_trip_test"
    round_trip_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    (round_trip_dir / "resolved.yaml").write_text(
        yaml.safe_dump(sans_hydra, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    try:
        # Hydra resolves config_path relative to the calling file's dir.
        rel = os.path.relpath(round_trip_dir, Path(__file__).resolve().parent)
        with hydra_initialize(version_base=None, config_path=rel):
            cfg = hydra_compose(config_name="resolved")
        reread = OmegaConf.to_container(cfg, resolve=True)
        reread = {k: v for k, v in reread.items() if k != "hydra"}
        assert reread == sans_hydra
    finally:
        import shutil

        shutil.rmtree(round_trip_dir, ignore_errors=True)
