"""Hydra ConfigStore: all groups as structured dataclasses (§10.1).

Groups: ``dataset`` (5 named datasets with the §4 pinned facts),
``model`` (spectral-drift, seven indicators, LSTM/TCN), ``training``
(default, none), ``evaluation`` (persistenceaware, baseline_classic),
``output`` (default). The primary config is ``configs/run.yaml``:
dataset/model/training/evaluation are selected by defaults-list
entries; run-level knobs (``models`` multi-select, ``seed``,
``n_seeds``, ``dataset_overrides``) live on ``RunConfig`` and are
applied as CLI overrides.

Structured nodes are registered for every group so a composed YAML file
is schema-checked at compose time (extra/missing/wrong-typed keys fail
before any work starts). Where both a file and a node exist for the
same group/name, Hydra validates the file against the node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpectedFileConfig:
    path: str = ""
    size: int = 0
    md5: str = ""


@dataclass
class DownloadConfig:
    mode: str = "manual"
    env_token_var: str = "DRYAD_API_TOKEN"
    retries: int = 3
    backoff: float = 1.0
    wait_minutes: float = 10.0
    poll_interval_sec: float = 5.0


@dataclass
class DatasetConfig:
    name: str = ""
    source: str = "synthetic"
    bif_type: str = "fold"
    licence: str = "project-generated"
    doi: str | None = None
    version_id: str | None = None
    archive_type: str | None = None
    parser: str | None = None
    expected_files: list[ExpectedFileConfig] = field(default_factory=list)
    download: DownloadConfig | None = None
    generator: str | None = None
    difficulty: str | None = None
    n_trajectories: int | None = None
    max_length: int | None = None
    noise_scale: float | None = None
    obs_noise_scale: float | None = None
    seed: int | None = None
    split: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectralDriftConfig:
    n_particles: int = 500
    c_min: float = 0.001
    delta: float = 0.05
    center_window: int = 50
    q_drift_grid: list[float] = field(
        default_factory=lambda: [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
    )
    sigma_u_grid: list[float] = field(
        default_factory=lambda: [0.15, 0.3, 0.6, 1.0]
    )


@dataclass
class IndicatorConfig:
    window_size: int = 30
    embedding_dim: int | None = None
    rank: int | None = None


@dataclass
class LstmConfig:
    in_channels: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1


@dataclass
class TcnConfig:
    in_channels: int = 1
    hidden_size: int = 32
    kernel_size: int = 5
    levels: int = 3


@dataclass
class ModelConfig:
    spectral_drift: SpectralDriftConfig | None = None
    var_csd: IndicatorConfig | None = None
    ac1_csd: IndicatorConfig | None = None
    skew_csd: IndicatorConfig | None = None
    sratio_csd: IndicatorConfig | None = None
    retrate_csd: IndicatorConfig | None = None
    dfa_csd: IndicatorConfig | None = None
    dmd_csd: IndicatorConfig | None = None
    lstm: LstmConfig | None = None
    tcn: TcnConfig | None = None


@dataclass
class TrainingConfig:
    enabled: bool = True
    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1.0e-5
    patience: int = 5
    spectral_radius_weight: float = 0.01
    spectral_threshold: float = 0.95
    scheduler_eta_min: float = 1.0e-6
    label_window: int = 60
    num_workers: int = 0
    deterministic: bool = True
    accelerator: str = "auto"
    devices: str = "auto"
    progress_bar: bool = False


@dataclass
class EvaluationConfig:
    name: str = "persistenceaware"
    k_persist: int = 5
    fpr_target: float = 0.05
    early_start_delta: float = 50.0
    early_end_delta: float = 5.0


@dataclass
class OutputConfig:
    store_trajectories: bool = False
    base_dir: str = "outputs"


@dataclass
class RunConfig:
    models: list[str] = field(default_factory=lambda: ["VAR-CSD"])
    seed: int = 42
    seed_offset: int = 0
    n_seeds: int = 1
    dataset_overrides: dict[str, Any] = field(default_factory=dict)


def register_configs() -> None:
    """Register every group; idempotent across repeated CLI invocations."""
    from hydra.core.config_store import ConfigStore

    store = ConfigStore.instance()

    store.store(group="dataset", name="tac", node=DatasetConfig(
        name="tac",
        source="dryad",
        doi="10.5061/dryad.4cj4k",
        licence="CC0-1.0",
        bif_type="subcritical_hopf",
        archive_type="zip",
        parser="nptdms",
        expected_files=[ExpectedFileConfig(
            path="Experimental_time_traces_tdms.zip",
            size=302008984,
            md5="82cc5298c245ad509e1ee414e2c34941",
        )],
    ))
    store.store(group="dataset", name="daphnia_ext", node=DatasetConfig(
        name="daphnia_ext",
        source="dryad",
        doi="10.5061/dryad.q3p64",
        licence="CC0-1.0",
        bif_type="transcritical",
        archive_type="zip",
        parser="zip-csv",
        expected_files=[
            ExpectedFileConfig(path="data-and-code.zip", size=13273254, md5="923e08e6bafae412c43250fef0752ac7"),
            ExpectedFileConfig(path="README_for_data-and-code.txt", size=4848, md5="11770ce4f5a2b5369dce6f6f846340c5"),
        ],
    ))
    for name, bif_type in (("synthetic_fold", "fold"), ("synthetic_hopf", "hopf"), ("synthetic_logistic", "logistic")):
        store.store(group="dataset", name=name, node=DatasetConfig(
            name=name,
            source="synthetic",
            bif_type=bif_type,
            generator="classic",
            difficulty="standard",
            n_trajectories=500,
            max_length=200,
            seed=42,
            split={"seed": 42, "train_frac": 0.6, "val_frac": 0.2, "replicate_based": True},
        ))

    store.store(group="model", name="default", node=ModelConfig(
        spectral_drift=SpectralDriftConfig(),
        var_csd=IndicatorConfig(window_size=30),
        ac1_csd=IndicatorConfig(window_size=30),
        skew_csd=IndicatorConfig(window_size=30),
        sratio_csd=IndicatorConfig(window_size=30),
        retrate_csd=IndicatorConfig(window_size=30),
        dfa_csd=IndicatorConfig(window_size=100),
        dmd_csd=IndicatorConfig(window_size=30, embedding_dim=6, rank=2),
        lstm=LstmConfig(),
        tcn=TcnConfig(),
    ))
    store.store(group="model", name="spectral_drift", node=ModelConfig(spectral_drift=SpectralDriftConfig()))
    store.store(group="model", name="var_csd", node=ModelConfig(var_csd=IndicatorConfig(window_size=30)))
    store.store(group="model", name="ac1_csd", node=ModelConfig(ac1_csd=IndicatorConfig(window_size=30)))
    store.store(group="model", name="skew_csd", node=ModelConfig(skew_csd=IndicatorConfig(window_size=30)))
    store.store(group="model", name="sratio_csd", node=ModelConfig(sratio_csd=IndicatorConfig(window_size=30)))
    store.store(group="model", name="retrate_csd", node=ModelConfig(retrate_csd=IndicatorConfig(window_size=30)))
    store.store(group="model", name="dfa_csd", node=ModelConfig(dfa_csd=IndicatorConfig(window_size=100)))
    store.store(group="model", name="dmd_csd", node=ModelConfig(dmd_csd=IndicatorConfig(window_size=30, embedding_dim=6, rank=2)))
    store.store(group="model", name="lstm", node=ModelConfig(lstm=LstmConfig()))
    store.store(group="model", name="tcn", node=ModelConfig(tcn=TcnConfig()))

    store.store(group="training", name="default", node=TrainingConfig())
    store.store(group="training", name="none", node=TrainingConfig(enabled=False))

    store.store(group="evaluation", name="persistenceaware", node=EvaluationConfig())
    store.store(group="evaluation", name="baseline_classic", node=EvaluationConfig(
        name="baseline_classic",
        k_persist=1,
    ))

    store.store(group="output", name="default", node=OutputConfig())


__all__ = [
    "DatasetConfig",
    "DownloadConfig",
    "EvaluationConfig",
    "ExpectedFileConfig",
    "IndicatorConfig",
    "LstmConfig",
    "ModelConfig",
    "OutputConfig",
    "RunConfig",
    "SpectralDriftConfig",
    "TcnConfig",
    "TrainingConfig",
    "register_configs",
]
