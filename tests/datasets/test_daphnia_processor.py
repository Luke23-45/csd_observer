"""DaphniaExt processor + pipeline tests (fixtures are synthetic
timeseries.csv / extinctions.csv tables mirroring the real layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from tests.datasets import _fixtures

build_daphnia_raw = _fixtures.build_daphnia_raw
daphnia_config = _fixtures.daphnia_config

N_CENSUS = 40


def test_process_bundle_contract(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    raw = build_daphnia_raw(tmp_path)
    bundle = process(raw, daphnia_config())

    B = 6
    assert bundle["features"].ndim == 3
    assert bundle["features"].dtype == np.float32
    assert np.isfinite(bundle["features"]).all()
    assert bundle["features"].shape == (B, N_CENSUS, 1)
    assert bundle["seq_lengths"].tolist() == [N_CENSUS] * B
    assert bundle["is_positive"].tolist() == [True, True, False, False, True, False]
    # Positives: extinction day 63, tau_days 14 -> census day 49 (index 7).
    assert bundle["bifurcation_times"][:2].tolist() == [7.0, 7.0]
    assert bundle["bifurcation_times"][4] == 7.0
    # Nulls: sentinel seq_length + 1.
    assert bundle["bifurcation_times"][2] == N_CENSUS + 1
    assert bundle["bifurcation_times"][3] == N_CENSUS + 1
    assert bundle["bifurcation_times"][5] == N_CENSUS + 1
    # Chamber-level count sums the two subpops' mean(sample1..3): ~20 alive.
    assert bundle["features"][0, 0, 0] == pytest.approx(20.0, abs=0.05)
    # Positives reach 0 after extinction day; nulls stay positive.
    assert bundle["features"][0, -1, 0] == 0.0
    assert bundle["features"][2, -1, 0] > 0.0


def test_process_restart_recoding(tmp_path: Path) -> None:
    """A restarted population (H7) is split into the first attempt (H7)
    and the restarted population (H72) at restart_threshold_day."""
    from csd_observer.datasets.daphnia_ext.process import process

    cells = ["ID,Subpop,Date,sample1,sample2,sample3"]
    # First attempt: two census days before the threshold.
    for subpop in (1, 2):
        cells.append(f"H7,{subpop},10/27/2008,5,5,5")
        cells.append(f"H7,{subpop},12/8/2008,4,4,4")
    # Restart: two census days at/after the threshold (154 days later).
    for subpop in (1, 2):
        cells.append(f"H7,{subpop},3/30/2009,8,8,8")
        cells.append(f"H7,{subpop},4/6/2009,9,9,9")
    # Null survivor keeps the bundle balanced.
    for index in range(3):
        for subpop in (1, 2):
            cells.append(f"C,{subpop},{_date(10, index * 7)},8,8,8")
    ex = "\n".join(
        ["ID,Start,End,Deteriorating",
         "H7a,10/27/2008,12/8/2008,1",
         "H7,3/30/2009,11/30/2009,1",
         "C,10/27/2008,1/26/2009,0"]
    )
    raw = build_daphnia_raw(tmp_path, csv_text="\n".join(cells), extinctions_text=ex)
    bundle = process(raw, daphnia_config(processing={"min_length": 2}))
    assert bundle["features"].shape[0] == 3
    assert bundle["is_positive"].tolist() == [False, True, True]
    assert bundle["meta"]["processing"]["labels"].keys() == {"C", "H7", "H72"}
    assert bundle["seq_lengths"].tolist() == [3, 2, 2]
    # The restart (H72) extinction day is experiment-relative (Start=3/30/2009
    # is the restart date, NOT the experiment start), so the tau window lands
    # inside the trajectory instead of clamping to index 0.
    assert bundle["meta"]["processing"]["labels"]["H72"]["extinction_day"] == 399
    h72_index = bundle["is_positive"].tolist().index(True, 1)
    assert 0.0 <= bundle["bifurcation_times"][h72_index] < bundle["seq_lengths"][h72_index]
    assert bundle["bifurcation_times"][h72_index] > 0.0


def test_process_missing_population_label_fails(tmp_path: Path) -> None:
    """A population without an extinction record is an annotation error."""
    from csd_observer.datasets.daphnia_ext.process import process

    ex = "\n".join(["ID,Start,End,Deteriorating",
                    "A,10/27/2008,12/29/2008,1",
                    "B,10/27/2008,12/29/2008,1",
                    "C,10/27/2008,1/26/2009,0",
                    "D,10/27/2008,1/26/2009,0",
                    "E,10/27/2008,12/29/2008,1"])
    raw = build_daphnia_raw(tmp_path, extinctions_text=ex)
    with pytest.raises(DatasetError) as exc_info:
        process(raw, daphnia_config())
    assert exc_info.value.code == DatasetErrorCode.PROCESS_ANNOTATION_MISSING
    assert "without an extinction label" in str(exc_info.value)


def test_process_orphan_label_fails(tmp_path: Path) -> None:
    """An extinction record without a matching population is an error."""
    from csd_observer.datasets.daphnia_ext.process import process

    ex = "\n".join(["ID,Start,End,Deteriorating",
                    "A,10/27/2008,12/29/2008,1",
                    "B,10/27/2008,12/29/2008,1",
                    "C,10/27/2008,1/26/2009,0",
                    "D,10/27/2008,1/26/2009,0",
                    "E,10/27/2008,12/29/2008,1",
                    "F,10/27/2008,1/26/2009,0",
                    "Zzz,10/27/2008,1/26/2009,0"])
    raw = build_daphnia_raw(tmp_path, extinctions_text=ex)
    with pytest.raises(DatasetError) as exc_info:
        process(raw, daphnia_config())
    assert exc_info.value.code == DatasetErrorCode.PROCESS_ANNOTATION_MISSING
    assert "without a matching population" in str(exc_info.value)


def test_process_missing_columns(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    raw = build_daphnia_raw(tmp_path, csv_text="ID,Subpop\nA,1\nB,2\n")
    with pytest.raises(DatasetError):
        process(raw, daphnia_config())


def test_process_short_population_excluded(tmp_path: Path) -> None:
    """A population with fewer census days than min_length is excluded and
    logged (dead first attempts) rather than failing the whole run."""
    from csd_observer.datasets.daphnia_ext.process import process

    cells = ["ID,Subpop,Date,sample1,sample2,sample3"]
    for index in range(10):
        for subpop in (1, 2):
            cells.append(f"A,{subpop},{_date(10, index * 7)},5,5,5")
    for index in range(N_CENSUS):
        for subpop in (1, 2):
            cells.append(f"B,{subpop},{_date(10, index * 7)},5,5,5")
    for index in range(N_CENSUS):
        for subpop in (1, 2):
            cells.append(f"C,{subpop},{_date(10, index * 7)},5,5,5")
    ex = "\n".join(["ID,Start,End,Deteriorating",
                    "A,10/27/2008,12/29/2008,1",
                    "B,10/27/2008,12/29/2008,1",
                    "C,10/27/2008,1/26/2009,0"])
    raw = build_daphnia_raw(tmp_path, csv_text="\n".join(cells), extinctions_text=ex)
    bundle = process(raw, daphnia_config())
    assert bundle["features"].shape[0] == 2
    assert bundle["is_positive"].tolist() == [True, False]
    excluded = bundle["meta"]["processing"]["excluded_short"]
    assert [e["population"] for e in excluded] == ["A"]
    assert excluded[0]["census_days"] == 10
    assert excluded[0]["min_length"] == 20
    assert excluded[0]["is_positive"] is True


def _date(month: int, day: int) -> str:
    from datetime import datetime, timedelta

    return (datetime(2008, 10, 27) + timedelta(days=day)).strftime("%m/%d/%Y")


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    from csd_observer.datasets.common.manifest import read_manifest
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.common.states import IngestState
    from csd_observer.datasets.daphnia_ext.process import process, validate_annotation

    build_daphnia_raw(tmp_path)
    root = tmp_path  # <root>/raw/daphnia lives under the pipeline root
    config = daphnia_config()
    state = run_pipeline(config, root, "daphnia_ext", process, extra_validator=validate_annotation)
    assert state is IngestState.READY_PROCESSED

    manifest = read_manifest(root / "processed" / "daphnia" / "manifest.json")
    assert manifest["dataset"]["bif_type"] == "transcritical"
    assert manifest["processing"]["effective"]["data_file"] == "timeseries.csv"
    assert manifest["processing"]["effective"]["extinctions_file"] == "extinctions.csv"
    assert manifest["gates"]["counts"] == {"signal": 3, "null": 3}
    assert manifest["split"]["policy"] == "replicate_based"

    # Staleness: changing tau_annotation_days forces re-processing.
    config["processing"]["tau_annotation_days"] = 30
    run_pipeline(config, root, "daphnia_ext", process, extra_validator=validate_annotation)
    manifest2 = read_manifest(root / "processed" / "daphnia" / "manifest.json")
    assert manifest2["processing"]["params"]["tau_annotation_days"] == 30


def test_registry_loads_processed_bundle(tmp_path: Path) -> None:
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.daphnia_ext.process import process, validate_annotation
    from csd_observer.datasets.registry import get_dataset

    root = tmp_path / "dataset"
    build_daphnia_raw(root)
    run_pipeline(daphnia_config(), root, "daphnia_ext", process, extra_validator=validate_annotation)

    bundle = get_dataset("daphnia_ext", {"data_root": str(root)})
    assert bundle["meta"]["bif_type"] == "transcritical"
    assert bundle["meta"]["n_replicates"] == 6
    assert bundle["signal"]["features"].shape[0] == 3
    assert bundle["null"]["features"].shape[0] == 3
