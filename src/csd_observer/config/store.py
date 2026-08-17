"""Schema contract for the protocol configuration groups (§10.1).

Groups: ``dataset`` (5 named datasets with the §4 pinned facts),
``model`` (seven indicators, LSTM/TCN/PatchTST), ``training``
(default, none), ``evaluation`` (persistenceaware, baseline_classic),
``output`` (default). The primary config is ``configs/run.yaml``:
dataset/model/training/evaluation/output are selected by defaults-list
entries; run-level knobs (``models`` multi-select, ``seed_offset``,
``n_seeds``, ``dataset_overrides``) live on ``RunConfig`` and are
applied as CLI overrides.

The group *values* live in ``configs/<group>/<name>.yaml`` — the
dataclasses here are the in-code schema contract, **not** Hydra
ConfigStore registrations: a same-name file+node pair would trigger
Hydra's deprecated "validated against ConfigStore schema" path
(``config_loader_impl``), so ``register_configs`` only builds the node
tables. ``validate_config`` type/keys-checks every composed group block
against the dataclasses, and ``tests/config/test_yaml_alignment.py``
pins every yaml file to exactly its node's non-None values.
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
class ProcessingConfig:
    """§5.1 processing parameters (per-dataset; unused fields are ignored).

    Shared fields: ``min_length`` (gate 2 of §5.4) and ``normalization``
    (``none`` default; ``zscore`` fits train-only statistics). TAC fields
    drive the dominant-mode envelope extraction and chunking; DaphniaExt
    fields drive the table mapping, README treatment coding and the
    transition-time alignment. Unknown params fail loudly at processing
    time (never silently assumed) and are recorded in the manifest.
    """

    min_length: int = 100
    normalization: str = "none"
    # --- TAC ---
    envelope: str = "raw"  # "raw" | "bandpass" (Hilbert envelope; §5.4)
    sample_rate_hz: float | None = None
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    analysis_window: int | None = None  # None = whole trace, one trajectory
    channel_group: str | None = None  # explicit TDMS group; else auto-select
    channel_name: str | None = None  # explicit TDMS channel; else auto-select
    section_pattern_stationary: str = "Stationary"
    section_pattern_ramp: str = "Ramp"
    ramp_onset_index: int | None = None  # explicit transition index inside a ramp
    # section; ``None`` = auto-detect from the envelope (§5.4, rate-dependent
    # transition delay is measured from the data, not assumed)
    ramp_onset_detect_window: int | None = None  # rolling window (samples); None = TAC default 5000
    ramp_onset_detect_factor: float | None = None  # growth threshold = factor * baseline; None = 3.0
    ramp_pre_samples: int | None = None  # pre-onset history in an aligned ramp window; None = 200
    ramp_post_samples: int | None = None  # post-onset confirmation in an aligned ramp window; None = 100
    # --- DaphniaExt ---
    replicate_column: str = "replicate"
    time_column: str = "day"
    count_column: str = "count"
    positive_column: str = "treatment"
    tau_annotation_days: int = 110  # Nature 467:456 CSD window (~110 days)
    window_days: int | None = None  # None = full span up to the transition
    data_file: str | None = None  # explicit table inside the archive
    extinctions_file: str | None = None  # explicit label table (real data)
    restart_ids: list[str] = field(default_factory=list)  # restarted IDs
    restart_threshold_day: int | None = None  # recode restart rows at/after this day
    treatments: dict[str, Any] = field(default_factory=dict)  # per-replicate
    # {replicate: {"treatment": "positive"|"null", "extinction_day": int}}


@dataclass
class DatasetConfig:
    """One registry dataset's composed group block (§5.6).

    ``feature_mode`` declares how the *models* package must reduce the
    dataset's feature channels to the scalar alarm mode (R2.2): ``None``
    = auto (legacy system-name fallback), ``radial`` = sqrt(x1^2+x2^2)
    over the first two channels, ``channel_0`` = first channel,
    ``envelope`` = first channel (pre-extracted amplitude envelope,
    TAC). The declaration lives here — never in the models package.
    """

    name: str = ""
    source: str = "synthetic"
    bif_type: str = "fold"
    feature_mode: str | None = None  # "radial" | "channel_0" | "envelope" | None (auto)
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
    processing: ProcessingConfig | None = None
    split: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorConfig:
    window_size: int = 30
    embedding_dim: int | None = None
    rank: int | None = None


@dataclass
class LstmConfig:
    # ``None`` = auto: the runner infers the channel count from the
    # dataset bundle (e.g. 2 for synthetic_hopf, 1 for the real datasets).
    in_channels: int | None = None
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    score_batch_size: int = 64


@dataclass
class TcnConfig:
    in_channels: int | None = None
    hidden_size: int = 32
    kernel_size: int = 5
    levels: int = 3
    score_batch_size: int = 64


@dataclass
class PatchTstConfig:
    """PatchTST-style alarm baseline (§6.4, family b3).

    ``in_channels`` is ``None`` = auto (the runner infers the channel
    count from the dataset bundle, like the LSTM/TCN blocks). ``stride``
    ``None`` = ``patch_len`` (non-overlapping patches, PatchTST default).
    """

    in_channels: int | None = None
    patch_len: int = 16
    stride: int | None = None  # None = patch_len (non-overlapping)
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    score_batch_size: int = 64


@dataclass
class ModelConfig:
    var_csd: IndicatorConfig | None = None
    ac1_csd: IndicatorConfig | None = None
    skew_csd: IndicatorConfig | None = None
    sratio_csd: IndicatorConfig | None = None
    retrate_csd: IndicatorConfig | None = None
    dfa_csd: IndicatorConfig | None = None
    dmd_csd: IndicatorConfig | None = None
    lstm: LstmConfig | None = None
    tcn: TcnConfig | None = None
    patchtst: PatchTstConfig | None = None


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
    """Run-level knobs (§10.2).

    ``seed`` was removed (R3.2): the §13 per-seed schedule is the single
    source of run seeds — ``base = seed_offset + s*1000`` plus the
    per-split deltas ``+101/+202``. A legacy ``seed`` knob silently
    contradicted the schedule and is gone.
    """

    models: list[str] = field(default_factory=lambda: ["VAR-CSD"])
    seed_offset: int = 0
    n_seeds: int = 1
    dataset_overrides: dict[str, Any] = field(default_factory=dict)


#: Registered dataset-group nodes by name (R0.1 defaults source: the
#: validator rejects user-set generator knobs on the ``dataset`` group by
#: comparing against these registered defaults; the yaml files under
#: ``configs/dataset/`` are pinned to the same values by
#: ``tests/config/test_yaml_alignment.py``).
_DATASET_NODES: dict[str, DatasetConfig] = {}

#: Every group node by (group, name), mirroring the yaml files. Used by
#: the schema check in ``validate_config`` and the alignment test.
_GROUP_NODES: dict[tuple[str, str], Any] = {}


def dataset_group_default(name: str) -> dict[str, Any]:
    """Registered default block for a dataset group name (plain dict).

    Returns ``{}`` when the name is not a registered dataset node
    (e.g. a third-party materialized dataset) — callers treat an empty
    dict as "no registered defaults available".
    """
    import dataclasses

    node = _DATASET_NODES.get(name)
    if node is None:
        return {}
    return dataclasses.asdict(node)


def dataset_node(name: str) -> DatasetConfig | None:
    """Registered schema node for a dataset group name.

    ``None`` when the name is not a registered dataset node (e.g. a
    third-party materialized dataset) — callers skip schema checks then.
    """
    return _DATASET_NODES.get(name)


def register_configs() -> None:
    """Build every group node; idempotent across repeated CLI invocations.

    Nodes are deliberately **not** registered into Hydra's ConfigStore:
    the group values live in the yaml files under
    ``configs/<group>/<name>.yaml``, and a same-name file+node pair
    triggers Hydra's deprecated "validated against ConfigStore schema"
    path (``config_loader_impl``). The dataclasses remain the in-code
    schema contract: ``validate_config`` keys/types-checks every composed
    group block against them, and ``tests/config/test_yaml_alignment.py``
    asserts the yaml files carry exactly the node values.
    """
    _register = _GROUP_NODES.__setitem__

    _tac_node = DatasetConfig(
        name="tac",
        source="dryad",
        doi="10.5061/dryad.4cj4k",
        licence="CC0-1.0",
        bif_type="subcritical_hopf",
        feature_mode="envelope",
        archive_type="zip",
        parser="nptdms",
        expected_files=[ExpectedFileConfig(path="Experimental_time_traces_tdms", size=0, md5="")],
        processing=ProcessingConfig(
            envelope="raw",
            sample_rate_hz=None,
            band_low_hz=None,
            band_high_hz=None,
            analysis_window=4096,
            channel_group="Set0",
            channel_name="Mic1",
            section_pattern_stationary="Stationary",
            section_pattern_ramp="Ramp",
            ramp_onset_index=None,
            ramp_onset_detect_window=5000,
            ramp_onset_detect_factor=3.0,
            ramp_pre_samples=2048,
            ramp_post_samples=2048,
        ),
        split={"replicate_based": True, "seed": 42, "train_frac": 0.6, "val_frac": 0.2},
    )
    _daphnia_node = DatasetConfig(
        name="daphnia_ext",
        source="dryad",
        doi="10.5061/dryad.q3p64",
        licence="CC0-1.0",
        bif_type="transcritical",
        feature_mode="channel_0",
        archive_type="zip",
        parser="zip-csv",
        expected_files=[
            ExpectedFileConfig(path="data-and-code", size=0, md5=""),
            ExpectedFileConfig(path="README_for_data-and-code.txt", size=4848, md5="11770ce4f5a2b5369dce6f6f846340c5"),
        ],
        processing=ProcessingConfig(
            min_length=20,
            replicate_column="ID",
            time_column="Date",
            count_column="count",
            positive_column="Deteriorating",
            tau_annotation_days=110,
            window_days=None,
            data_file="timeseries.csv",
            extinctions_file="extinctions.csv",
            restart_ids=["H7", "H9", "J4", "K2", "K10"],
            restart_threshold_day=154,
        ),
        split={"replicate_based": True, "seed": 42, "train_frac": 0.6, "val_frac": 0.2},
    )
    _register(("dataset", "tac"), _tac_node)
    _DATASET_NODES["tac"] = _tac_node
    _register(("dataset", "daphnia_ext"), _daphnia_node)
    _DATASET_NODES["daphnia_ext"] = _daphnia_node
    for name, bif_type, feature_mode in (
        ("synthetic_fold", "fold", "channel_0"),
        ("synthetic_hopf", "hopf", "radial"),
        ("synthetic_logistic", "logistic", "channel_0"),
    ):
        node = DatasetConfig(
            name=name,
            source="synthetic",
            bif_type=bif_type,
            feature_mode=feature_mode,
            generator="classic",
            difficulty="standard",
            n_trajectories=500,
            max_length=200,
            seed=42,
            split={"seed": 42, "train_frac": 0.6, "val_frac": 0.2, "replicate_based": True},
        )
        _register(("dataset", name), node)
        _DATASET_NODES[name] = node

    _register(("model", "default"), ModelConfig(
        var_csd=IndicatorConfig(window_size=30),
        ac1_csd=IndicatorConfig(window_size=30),
        skew_csd=IndicatorConfig(window_size=30),
        sratio_csd=IndicatorConfig(window_size=30),
        retrate_csd=IndicatorConfig(window_size=30),
        dfa_csd=IndicatorConfig(window_size=100),
        dmd_csd=IndicatorConfig(window_size=30, embedding_dim=6, rank=2),
        lstm=LstmConfig(),
        tcn=TcnConfig(),
        patchtst=PatchTstConfig(),
    ))
    _register(("model", "var_csd"), ModelConfig(var_csd=IndicatorConfig(window_size=30)))
    _register(("model", "ac1_csd"), ModelConfig(ac1_csd=IndicatorConfig(window_size=30)))
    _register(("model", "skew_csd"), ModelConfig(skew_csd=IndicatorConfig(window_size=30)))
    _register(("model", "sratio_csd"), ModelConfig(sratio_csd=IndicatorConfig(window_size=30)))
    _register(("model", "retrate_csd"), ModelConfig(retrate_csd=IndicatorConfig(window_size=30)))
    _register(("model", "dfa_csd"), ModelConfig(dfa_csd=IndicatorConfig(window_size=100)))
    _register(("model", "dmd_csd"), ModelConfig(dmd_csd=IndicatorConfig(window_size=30, embedding_dim=6, rank=2)))
    _register(("model", "lstm"), ModelConfig(lstm=LstmConfig()))
    _register(("model", "tcn"), ModelConfig(tcn=TcnConfig()))
    _register(("model", "patchtst"), ModelConfig(patchtst=PatchTstConfig()))

    _register(("training", "default"), TrainingConfig())
    _register(("training", "none"), TrainingConfig(enabled=False))

    _register(("evaluation", "persistenceaware"), EvaluationConfig())
    _register(("evaluation", "baseline_classic"), EvaluationConfig(
        name="baseline_classic",
        k_persist=1,
    ))

    _register(("output", "default"), OutputConfig())


__all__ = [
    "DatasetConfig",
    "DownloadConfig",
    "EvaluationConfig",
    "ExpectedFileConfig",
    "IndicatorConfig",
    "LstmConfig",
    "ModelConfig",
    "OutputConfig",
    "PatchTstConfig",
    "ProcessingConfig",
    "RunConfig",
    "TcnConfig",
    "TrainingConfig",
    "dataset_group_default",
    "dataset_node",
    "register_configs",
]
