"""TAC processor + pipeline tests (fixtures are synthetic extracted TDMS dirs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from tests.datasets import _fixtures

npTDMS = pytest.importorskip("nptdms")

STATIONARY_FILES = 2
RAMP_FILES = 1
N = 256  # >= pipeline MIN_LENGTH 100

build_custom_tac_raw = _fixtures.build_custom_tac_raw
tac_config = _fixtures.tac_config
_write_tdms = _fixtures.write_tdms
_stationary_trace = _fixtures.stationary_trace
_ramp_trace = _fixtures.ramp_trace


def build_tac_raw(tmp_path: Path) -> Path:
    return build_custom_tac_raw(tmp_path, STATIONARY_FILES, RAMP_FILES)


def test_process_bundle_contract(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    raw = build_tac_raw(tmp_path)
    bundle = process(raw, tac_config())

    B = STATIONARY_FILES + RAMP_FILES
    assert bundle["features"].shape == (B, N, 1)
    assert bundle["seq_lengths"].tolist() == [N] * B
    assert bundle["features"].dtype == np.float32
    assert np.isfinite(bundle["features"]).all()
    # Files are sorted by name: "Ramp*" < "Stationary*" lexicographically.
    expected_positive = [True] * RAMP_FILES + [False] * STATIONARY_FILES
    assert bundle["is_positive"].tolist() == expected_positive
    # Null sentinel convention: seq_length + 1.
    for i in range(RAMP_FILES, B):
        assert bundle["bifurcation_times"][i] == N + 1
    for i in range(RAMP_FILES):
        # Ramp onset is auto-detected from the fixture's growing envelope:
        # strictly inside the trajectory so the early-warning protocol is
        # well-defined (a tau of 0 would leave no pre-transition prefix).
        assert 0 < bundle["bifurcation_times"][i] < N
    # Envelope is the analytic amplitude of the channel.
    assert bundle["features"].min() >= 0.0


def test_process_bandpass_envelope(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(7)
    t = np.arange(N)
    signal = np.cos(2 * np.pi * 100.0 * t / 1000.0) + 0.1 * rng.normal(size=N)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    # Both sections are required; assertions target the ramp file only.
    _write_tdms(extracted / "2016-10-20_Ramp0.tdms", "ch", signal)
    _write_tdms(extracted / "2016-10-20_Stationary0.tdms", "ch", _stationary_trace(rng, N))

    cfg = tac_config(processing={
        "envelope": "bandpass",
        "sample_rate_hz": 1000.0,
        "band_low_hz": 50.0,
        "band_high_hz": 150.0,
        # Constant-amplitude sinusoid: no growth to auto-detect, so pin the
        # onset explicitly (this test exercises the bandpass envelope, not
        # the onset detector).
        "ramp_onset_index": 60,
    })
    bundle = process(raw, cfg)
    env = bundle["features"][0, :, 0]
    # Interior envelope of a unit sinusoid is ~1 (ignoring edge ripple).
    assert np.allclose(env[50:-50], 1.0, atol=0.2)


def test_process_bandpass_requires_band(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    raw = build_tac_raw(tmp_path)
    with pytest.raises(ValueError, match="bandpass envelope requires"):
        process(raw, tac_config(processing={"envelope": "bandpass"}))


def test_process_chunking(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    raw = build_tac_raw(tmp_path)
    bundle = process(raw, tac_config(processing={"analysis_window": 128}))
    # Stationary files split into 128-sample null chunks; each ramp file
    # yields ONE window aligned to its auto-detected onset (which must
    # carry pre/post-onset history, unlike the old fixed chunks that all
    # shared a single onset index).
    n_stationary_chunks = 2 * STATIONARY_FILES
    assert bundle["features"].shape[0] == RAMP_FILES + n_stationary_chunks
    assert bundle["is_positive"].tolist() == [True] * RAMP_FILES + [False] * n_stationary_chunks
    for i in range(RAMP_FILES):
        assert 0 < bundle["bifurcation_times"][i] < bundle["seq_lengths"][i]
    for i in range(RAMP_FILES, bundle["features"].shape[0]):
        assert bundle["seq_lengths"][i] == 128
        assert bundle["bifurcation_times"][i] == 129  # sentinel = length + 1


def test_process_auto_detects_ramp_onset(tmp_path: Path) -> None:
    """Auto-detection: a step from quiet noise to a strong oscillation is
    found within one detection window of the true transition."""
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(21)
    n = 4000
    t = np.arange(n)
    amp = np.where(t < 2500, 0.05, 2.0)
    signal = amp * np.cos(0.3 * t) + 0.05 * rng.normal(size=n)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    _write_tdms(extracted / "2016-10-20_Ramp0.tdms", "ch", signal)
    _write_tdms(extracted / "2016-10-20_Stationary0.tdms", "ch", _stationary_trace(rng, n))

    cfg = tac_config(processing={"ramp_onset_detect_window": 100, "ramp_onset_detect_factor": 3.0})
    bundle = process(raw, cfg)
    onset = bundle["bifurcation_times"][0]
    assert abs(onset - 2500) <= 100
    assert 0 < onset < bundle["seq_lengths"][0]


def test_process_ramp_without_growth_fails_loudly(tmp_path: Path) -> None:
    """A constant-amplitude 'ramp' (no detectable growth) is an annotation
    error, never a silent tau=0 positive."""
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(31)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    _write_tdms(extracted / "2016-10-20_Ramp0.tdms", "ch", _stationary_trace(rng, N))
    _write_tdms(extracted / "2016-10-20_Stationary0.tdms", "ch", _stationary_trace(rng, N))
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config())
    assert exc_info.value.code == DatasetErrorCode.PROCESS_ANNOTATION_MISSING
    assert "onset" in str(exc_info.value)


def test_process_aligned_ramp_window_explicit_onset(tmp_path: Path) -> None:
    """With chunking + explicit onset, the ramp yields ONE window aligned
    around the onset (``[onset - pre, onset + post)``) whose tau is the
    relative onset offset."""
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(23)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    _write_tdms(extracted / "2016-10-20_Ramp0.tdms", "ch", _ramp_trace(rng, N))
    _write_tdms(extracted / "2016-10-20_Stationary0.tdms", "ch", _stationary_trace(rng, N))

    cfg = tac_config(processing={
        "analysis_window": 128,
        "ramp_onset_index": 100,
        "ramp_pre_samples": 60,
        "ramp_post_samples": 40,
    })
    bundle = process(raw, cfg)
    # window [100-60, 100+40) = [40, 140), length 100; tau = 100 - 40 = 60.
    assert bundle["seq_lengths"][0] == 100
    assert bundle["bifurcation_times"][0] == 60.0
    assert bundle["is_positive"][0]
    # Stationary chunks stay fixed 128-sample nulls.
    assert bundle["seq_lengths"][1:].tolist() == [128, 128]
    assert bundle["bifurcation_times"][1:].tolist() == [129.0, 129.0]


def test_process_trailing_chunk_fails_loudly(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(9)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    _write_tdms(extracted / "2016-10-20_Stationary0.tdms", "ch", _stationary_trace(rng, 200))
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config(processing={"analysis_window": 128}))
    assert exc_info.value.code == DatasetErrorCode.PROCESS_SHORT_LENGTH
    assert "trailing chunk" in str(exc_info.value)


def test_process_unknown_section_rejected(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(3)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    _write_tdms(extracted / "2016-10-20_Unknown0.tdms", "ch", _stationary_trace(rng, N))
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config())
    assert exc_info.value.code == DatasetErrorCode.INGEST_MANIFEST_MISMATCH


def test_process_explicit_channel_missing(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    raw = build_tac_raw(tmp_path)
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config(processing={"channel_group": "Nope", "channel_name": "ch0"}))
    assert exc_info.value.code == DatasetErrorCode.INGEST_MANIFEST_MISMATCH


def test_auto_select_longest_finite_channel(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(5)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    tdms = extracted / "2016-10-20_Stationary0.tdms"
    from nptdms import ChannelObject, TdmsWriter

    with TdmsWriter(str(tdms)) as writer:
        writer.write_segment([
            ChannelObject("Raw", "short", _stationary_trace(rng, 50).astype(np.float64)),
            ChannelObject("Raw", "long", _stationary_trace(rng, N).astype(np.float64)),
        ])
    _write_tdms(extracted / "2016-10-20_Ramp0.tdms", "ch", _ramp_trace(rng, N))
    bundle = process(raw, tac_config())
    assert bundle["features"].shape == (2, N, 1)
    # selected_channels keys are the extracted paths.
    assert bundle["meta"]["processing"]["selected_channels"][str(tdms)]["channel"] == "long"


def test_process_all_stationary_is_imbalance(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(4)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        _write_tdms(extracted / f"2016-10-20_Stationary{i}.tdms", f"ch{i}", _stationary_trace(rng, N))
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config())
    assert exc_info.value.code == DatasetErrorCode.SPLIT_IMBALANCE


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    from csd_observer.datasets.common.manifest import read_manifest
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.common.states import IngestState
    from csd_observer.datasets.tac.process import process, validate_annotation

    build_tac_raw(tmp_path)
    root = tmp_path  # <root>/raw/tac lives under the pipeline root
    config = tac_config()
    state = run_pipeline(config, root, "tac", process, extra_validator=validate_annotation)
    assert state is IngestState.READY_PROCESSED

    processed = root / "processed" / "tac"
    assert (processed / "manifest.json").exists()
    assert (processed / "train" / "train.npz").exists()
    assert (processed / "val" / "val.npz").exists()
    assert (processed / "test" / "test.npz").exists()
    manifest = read_manifest(processed / "manifest.json")
    assert manifest["dataset"]["bif_type"] == "subcritical_hopf"
    assert manifest["processing"]["effective"]["sections"] == {"stationary": STATIONARY_FILES, "ramp": RAMP_FILES}
    assert manifest["gates"]["counts"] == {"signal": RAMP_FILES, "null": STATIONARY_FILES}
    assert manifest["split"]["policy"] == "replicate_based"
    # Split indices are replicate-level over the bundle.
    indices = manifest["split"]["indices"]
    assert sorted(indices) == ["test", "train", "val"]

    # Idempotent second call short-circuits.
    state2 = run_pipeline(config, root, "tac", process, extra_validator=validate_annotation)
    assert state2 is IngestState.READY_PROCESSED

    # Staleness: a changed processing param forces re-processing.
    config["processing"]["analysis_window"] = 128
    run_pipeline(config, root, "tac", process, extra_validator=validate_annotation)
    manifest2 = read_manifest(processed / "manifest.json")
    assert manifest2["processing"]["params"]["analysis_window"] == 128
    # sections are per raw FILE; stationary chunks double, but each ramp
    # file still yields exactly one aligned onset window (signal count
    # stays at RAMP_FILES).
    assert manifest2["processing"]["effective"]["sections"] == {"stationary": STATIONARY_FILES, "ramp": RAMP_FILES}
    assert manifest2["gates"]["counts"] == {"signal": RAMP_FILES, "null": 2 * STATIONARY_FILES}


def test_run_pipeline_staleness_reprocesses_real_source(tmp_path: Path) -> None:
    """R4.x regression: a changed processing block must re-provision even
    for a real (non-synthetic) dataset, where ``ingest_raw`` would
    otherwise short-circuit on the stale cached manifest."""
    from csd_observer.datasets.common.manifest import read_manifest
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.tac.process import process, validate_annotation

    build_tac_raw(tmp_path)
    root = tmp_path  # <root>/raw/tac; source="dryad" exercises ingest_raw
    config = tac_config(source="dryad")
    run_pipeline(config, root, "tac", process, extra_validator=validate_annotation)
    processed = root / "processed" / "tac"

    config["processing"]["analysis_window"] = 128
    run_pipeline(config, root, "tac", process, extra_validator=validate_annotation)
    manifest = read_manifest(processed / "manifest.json")
    assert manifest["processing"]["params"]["analysis_window"] == 128
    # Stationary chunks double, ramp stays one aligned window per file.
    assert manifest["gates"]["counts"] == {"signal": RAMP_FILES, "null": 2 * STATIONARY_FILES}


def _build_custom_raw(tmp_path: Path, n_stationary: int, n_ramp: int) -> Path:
    return build_custom_tac_raw(tmp_path, n_stationary, n_ramp)


def test_registry_loads_processed_bundle(tmp_path: Path) -> None:
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.registry import get_dataset
    from csd_observer.datasets.tac.process import process, validate_annotation

    n_sig, n_null = 4, 3
    root = tmp_path / "dataset"
    _build_custom_raw(root, n_null, n_sig)
    run_pipeline(tac_config(), root, "tac", process, extra_validator=validate_annotation)

    bundle = get_dataset("tac", {"data_root": str(root)})
    assert set(bundle) == {"signal", "null", "meta"}
    assert bundle["meta"]["bif_type"] == "subcritical_hopf"
    assert bundle["meta"]["n_replicates"] == n_sig + n_null
    assert bundle["signal"]["is_positive"].all()
    assert not bundle["null"]["is_positive"].any()
    assert bundle["signal"]["features"].shape[0] == n_sig
    # Per-class replicate split needs >= 3 replicates per class.
    for subset in ("signal", "null"):
        assert {"train", "val", "test"} <= set(bundle[subset]["split_indices"])
