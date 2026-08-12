"""TAC processor + pipeline tests (fixtures are synthetic TDMS archives)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from tests.datasets import _fixtures

npTDMS = pytest.importorskip("nptdms")

STATIONARY_FILES = 2
RAMP_FILES = 1
N = 256  # >= pipeline MIN_LENGTH 100

build_custom_tac_archive = _fixtures.build_custom_tac_archive
tac_config = _fixtures.tac_config
_write_tdms = _fixtures.write_tdms
_stationary_trace = _fixtures.stationary_trace


def build_tac_archive(tmp_path: Path) -> Path:
    return build_custom_tac_archive(tmp_path, STATIONARY_FILES, RAMP_FILES)


def test_process_bundle_contract(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    build_tac_archive(tmp_path)
    bundle = process(tmp_path / "raw", tac_config())

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
        assert bundle["bifurcation_times"][i] == 0  # ramp_onset_index default
    # Envelope is the analytic amplitude of the channel.
    assert bundle["features"].min() >= 0.0


def test_process_bandpass_envelope(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(7)
    t = np.arange(N)
    signal = np.cos(2 * np.pi * 100.0 * t / 1000.0) + 0.1 * rng.normal(size=N)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Both sections are required; assertions target the ramp file only.
    tdms = tmp_path / "2016-10-20_Ramp0.tdms"
    _write_tdms(tdms, "ch", signal)
    stationary = tmp_path / "2016-10-20_Stationary0.tdms"
    _write_tdms(stationary, "ch", _stationary_trace(rng, N))
    with zipfile.ZipFile(raw / "Experimental_time_traces_tdms.zip", "w") as zf:
        zf.write(tdms, tdms.name)
        zf.write(stationary, stationary.name)

    cfg = tac_config(processing={
        "envelope": "bandpass",
        "sample_rate_hz": 1000.0,
        "band_low_hz": 50.0,
        "band_high_hz": 150.0,
    })
    bundle = process(raw, cfg)
    env = bundle["features"][0, :, 0]
    # Interior envelope of a unit sinusoid is ~1 (ignoring edge ripple).
    assert np.allclose(env[50:-50], 1.0, atol=0.2)


def test_process_bandpass_requires_band(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    build_tac_archive(tmp_path)
    with pytest.raises(ValueError, match="bandpass envelope requires"):
        process(tmp_path / "raw", tac_config(processing={"envelope": "bandpass"}))


def test_process_chunking(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    build_tac_archive(tmp_path)
    bundle = process(tmp_path / "raw", tac_config(processing={"analysis_window": 128}))
    B = 2 * (STATIONARY_FILES + RAMP_FILES)
    assert bundle["features"].shape == (B, 128, 1)
    assert bundle["seq_lengths"].tolist() == [128] * B
    assert bundle["is_positive"].tolist() == [True, True] * RAMP_FILES + [False, False] * STATIONARY_FILES


def test_process_trailing_chunk_fails_loudly(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(9)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tdms = tmp_path / "2016-10-20_Stationary0.tdms"
    _write_tdms(tdms, "ch", _stationary_trace(rng, 200))
    with zipfile.ZipFile(raw / "Experimental_time_traces_tdms.zip", "w") as zf:
        zf.write(tdms, tdms.name)
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config(processing={"analysis_window": 128}))
    assert exc_info.value.code == DatasetErrorCode.PROCESS_SHORT_LENGTH
    assert "trailing chunk" in str(exc_info.value)


def test_process_unknown_section_rejected(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(3)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tdms = tmp_path / "2016-10-20_Unknown0.tdms"
    _write_tdms(tdms, "ch", _stationary_trace(rng, N))
    with zipfile.ZipFile(raw / "Experimental_time_traces_tdms.zip", "w") as zf:
        zf.write(tdms, tdms.name)
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config())
    assert exc_info.value.code == DatasetErrorCode.INGEST_MANIFEST_MISMATCH


def test_process_explicit_channel_missing(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    build_tac_archive(tmp_path)
    with pytest.raises(DatasetError) as exc_info:
        process(tmp_path / "raw", tac_config(processing={"channel_group": "Nope", "channel_name": "ch0"}))
    assert exc_info.value.code == DatasetErrorCode.INGEST_MANIFEST_MISMATCH


def test_auto_select_longest_finite_channel(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(5)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tdms = tmp_path / "2016-10-20_Stationary0.tdms"
    from nptdms import ChannelObject, TdmsWriter

    with TdmsWriter(str(tdms)) as writer:
        writer.write_segment([
            ChannelObject("Raw", "short", _stationary_trace(rng, 50).astype(np.float64)),
            ChannelObject("Raw", "long", _stationary_trace(rng, N).astype(np.float64)),
        ])
    ramp = tmp_path / "2016-10-20_Ramp0.tdms"
    _write_tdms(ramp, "ch", _stationary_trace(rng, N))
    with zipfile.ZipFile(raw / "Experimental_time_traces_tdms.zip", "w") as zf:
        zf.write(tdms, tdms.name)
        zf.write(ramp, ramp.name)
    bundle = process(raw, tac_config())
    assert bundle["features"].shape == (2, N, 1)
    # selected_channels keys are the EXTRACTED paths under raw/_extracted.
    extracted = raw / "_extracted" / tdms.name
    assert bundle["meta"]["processing"]["selected_channels"][str(extracted)]["channel"] == "long"


def test_process_all_stationary_is_imbalance(tmp_path: Path) -> None:
    from csd_observer.datasets.tac.process import process

    rng = np.random.default_rng(4)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / "Experimental_time_traces_tdms.zip", "w") as zf:
        for i in range(2):
            tdms = tmp_path / f"2016-10-20_Stationary{i}.tdms"
            _write_tdms(tdms, f"ch{i}", _stationary_trace(rng, N))
            zf.write(tdms, tdms.name)
    with pytest.raises(DatasetError) as exc_info:
        process(raw, tac_config())
    assert exc_info.value.code == DatasetErrorCode.SPLIT_IMBALANCE


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    from csd_observer.datasets.common.manifest import read_manifest
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.common.states import IngestState
    from csd_observer.datasets.tac.process import process, validate_annotation

    build_tac_archive(tmp_path)
    root = tmp_path  # raw/ lives directly under the pipeline root
    config = tac_config()
    state = run_pipeline(config, root, process, extra_validator=validate_annotation)
    assert state is IngestState.READY_PROCESSED

    processed = root / "processed"
    assert (processed / "arrays.npz").exists()
    manifest = read_manifest(processed / "manifest.json")
    assert manifest["dataset"]["bif_type"] == "subcritical_hopf"
    assert manifest["processing"]["effective"]["sections"] == {"stationary": STATIONARY_FILES, "ramp": RAMP_FILES}
    assert manifest["gates"]["counts"] == {"signal": RAMP_FILES, "null": STATIONARY_FILES}
    assert manifest["split"]["policy"] == "replicate_based"
    # Split indices are replicate-level over the bundle.
    indices = manifest["split"]["indices"]
    assert sorted(indices) == ["test", "train", "val"]

    # Idempotent second call short-circuits.
    state2 = run_pipeline(config, root, process, extra_validator=validate_annotation)
    assert state2 is IngestState.READY_PROCESSED

    # Staleness: a changed processing param forces re-processing.
    config["processing"]["analysis_window"] = 128
    run_pipeline(config, root, process, extra_validator=validate_annotation)
    manifest2 = read_manifest(processed / "manifest.json")
    assert manifest2["processing"]["params"]["analysis_window"] == 128
    # sections are per raw FILE; replicate counts (chunks) double.
    assert manifest2["processing"]["effective"]["sections"] == {"stationary": STATIONARY_FILES, "ramp": RAMP_FILES}
    assert manifest2["gates"]["counts"] == {"signal": 2 * RAMP_FILES, "null": 2 * STATIONARY_FILES}


def _build_custom_archive(tmp_path: Path, n_stationary: int, n_ramp: int) -> Path:
    return build_custom_tac_archive(tmp_path, n_stationary, n_ramp)


def test_registry_loads_processed_bundle(tmp_path: Path) -> None:
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.registry import get_dataset
    from csd_observer.datasets.tac.process import process, validate_annotation

    n_sig, n_null = 4, 3
    _build_custom_archive(tmp_path, n_null, n_sig)
    root = tmp_path / "dataset" / "tac"
    dataset_raw = root / "raw"
    dataset_raw.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "Experimental_time_traces_tdms.zip").replace(dataset_raw / "Experimental_time_traces_tdms.zip")
    run_pipeline(tac_config(), root, process, extra_validator=validate_annotation)

    bundle = get_dataset("tac", {"data_root": str(tmp_path / "dataset")})
    assert set(bundle) == {"signal", "null", "meta"}
    assert bundle["meta"]["bif_type"] == "subcritical_hopf"
    assert bundle["meta"]["n_replicates"] == n_sig + n_null
    assert bundle["signal"]["is_positive"].all()
    assert not bundle["null"]["is_positive"].any()
    assert bundle["signal"]["features"].shape[0] == n_sig
    # Per-class replicate split needs >= 3 replicates per class.
    for subset in ("signal", "null"):
        assert {"train", "val", "test"} <= set(bundle[subset]["split_indices"])
