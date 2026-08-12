"""DaphniaExt processor + pipeline tests (fixtures are synthetic archives)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from tests.datasets import _fixtures

MIN_LENGTH = 100

build_daphnia_fixture = _fixtures.build_daphnia_fixture
DEFAULT_README = _fixtures.DEFAULT_README
daphnia_config = _fixtures.daphnia_config


def test_process_bundle_contract(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    build_daphnia_fixture(tmp_path, readme_text=DEFAULT_README)
    bundle = process(tmp_path / "raw", daphnia_config())

    assert bundle["features"].ndim == 3
    assert bundle["features"].dtype == np.float32
    assert np.isfinite(bundle["features"]).all()
    B = 6
    assert bundle["features"].shape[0] == B
    assert bundle["seq_lengths"].shape == (B,)
    assert bundle["is_positive"].tolist() == [True, True, False, False, True, False]
    # A/B/E: days 0..300 (301 samples); C: 201; D: 251; F: 221 -> padded to 301.
    assert bundle["features"].shape[1] == 301
    assert bundle["seq_lengths"].tolist() == [301, 301, 201, 251, 301, 221]
    # Padding beyond seq_lengths is zero, not NaN.
    assert (bundle["features"][2, 201:, 0] == 0.0).all()
    # Positives: tau index = nearest day to extinction - 110 -> day 190.
    assert bundle["bifurcation_times"][:2].tolist() == [190.0, 190.0]
    # Nulls: sentinel seq_length + 1.
    assert bundle["bifurcation_times"][2] == 202.0
    assert bundle["bifurcation_times"][3] == 252.0
    # Positive counts decay toward 0 (extinction); nulls stay high.
    assert bundle["features"][0, 300, 0] == 0.0
    assert bundle["features"][2, 50, 0] > 50.0


def test_process_readme_annotations_and_column_labels(tmp_path: Path) -> None:
    """Treatments resolve from the README when present; from the
    explicit column otherwise (controls have no README entry)."""
    from csd_observer.datasets.daphnia_ext.process import process

    build_daphnia_fixture(tmp_path, readme_text=DEFAULT_README)
    bundle = process(tmp_path / "raw", daphnia_config())
    assert bundle["meta"]["processing"]["annotations"] == {
        "A": {"treatment": "positive", "extinction_day": 300.0},
        "B": {"treatment": "positive", "extinction_day": 300.0},
        "E": {"treatment": "positive", "extinction_day": 300.0},
        "C": {"treatment": "null"},
        "D": {"treatment": "null"},
        "F": {"treatment": "null"},
    }


def test_process_positive_without_extinction_day_fails(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    readme = """replicate: A
treatment: positive
"""
    build_daphnia_fixture(tmp_path, readme_text=readme)
    with pytest.raises(DatasetError) as exc_info:
        process(tmp_path / "raw", daphnia_config())
    assert exc_info.value.code == DatasetErrorCode.PROCESS_ANNOTATION_MISSING
    assert "extinction" in str(exc_info.value)


def test_process_unlabeled_replicate_fails(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    csv = "\n".join(
        [f"{rep},{day},{val},unknown" for rep, day, val in
         [(r, d, 10.0) for r in ("A", "B", "C", "D") for d in range(MIN_LENGTH + 1)]]
    )
    csv = f"replicate,day,count,treatment\n{csv}"
    build_daphnia_fixture(tmp_path, readme_text="", csv_text=csv)
    with pytest.raises(DatasetError) as exc_info:
        process(tmp_path / "raw", daphnia_config())
    assert exc_info.value.code == DatasetErrorCode.PROCESS_ANNOTATION_MISSING


def test_process_missing_columns(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    build_daphnia_fixture(tmp_path, readme_text="", csv_text="replicate,day\nA,0\nB,1\n")
    with pytest.raises(DatasetError) as exc_info:
        process(tmp_path / "raw", daphnia_config())
    assert exc_info.value.code == DatasetErrorCode.INGEST_MANIFEST_MISMATCH
    assert "lacks columns" in str(exc_info.value)


def test_process_window_days_truncation(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    build_daphnia_fixture(tmp_path, readme_text=DEFAULT_README)
    cfg = daphnia_config(processing={"window_days": 150})
    bundle = process(tmp_path / "raw", cfg)
    # Window days 150..300 inclusive -> 151 samples per replicate.
    assert bundle["features"].shape[1] == 151
    assert bundle["seq_lengths"].tolist() == [151] * 6
    # tau = extinction - 110 = day 190 -> index 40 within the window.
    assert bundle["bifurcation_times"][:2].tolist() == [40.0, 40.0]
    assert bundle["bifurcation_times"][2] == 152.0


def test_process_short_replicate_fails(tmp_path: Path) -> None:
    from csd_observer.datasets.daphnia_ext.process import process

    readme = """replicate: A
treatment: positive
extinction_day: 60
replicate: B
treatment: positive
extinction_day: 300
replicate: C
treatment: null
replicate: D
treatment: null
"""
    csv = "\n".join(
        [f"{rep},{day},{val},{label}" for rep, span, label in (("A", 60, "positive"), ("B", 300, "positive"), ("C", 200, "control"), ("D", 250, "control"))
         for day, val in [(d, max(1.0, 30.0 - d)) for d in range(span + 1)]]
    )
    csv = f"replicate,day,count,treatment\n{csv}"
    build_daphnia_fixture(tmp_path, readme_text=readme, csv_text=csv)
    with pytest.raises(DatasetError) as exc_info:
        process(tmp_path / "raw", daphnia_config())
    assert exc_info.value.code == DatasetErrorCode.PROCESS_SHORT_LENGTH


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    from csd_observer.datasets.common.manifest import read_manifest
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.common.states import IngestState
    from csd_observer.datasets.daphnia_ext.process import process, validate_annotation

    build_daphnia_fixture(tmp_path, readme_text=DEFAULT_README)
    root = tmp_path  # raw/ lives directly under the pipeline root
    config = daphnia_config()
    state = run_pipeline(config, root, process, extra_validator=validate_annotation)
    assert state is IngestState.READY_PROCESSED

    manifest = read_manifest(root / "processed" / "manifest.json")
    assert manifest["dataset"]["bif_type"] == "transcritical"
    assert manifest["processing"]["effective"]["data_file"] == "counts.csv"
    assert manifest["processing"]["effective"]["readme_parsed"] is True
    assert manifest["gates"]["counts"] == {"signal": 3, "null": 3}
    assert manifest["split"]["policy"] == "replicate_based"

    # Staleness: changing tau_annotation_days forces re-processing.
    config["processing"]["tau_annotation_days"] = 130
    run_pipeline(config, root, process, extra_validator=validate_annotation)
    manifest2 = read_manifest(root / "processed" / "manifest.json")
    assert manifest2["processing"]["params"]["tau_annotation_days"] == 130


def test_registry_loads_processed_bundle(tmp_path: Path) -> None:
    from csd_observer.datasets.common.pipeline import run_pipeline
    from csd_observer.datasets.daphnia_ext.process import process, validate_annotation
    from csd_observer.datasets.registry import get_dataset

    build_daphnia_fixture(tmp_path, readme_text=DEFAULT_README)
    root = tmp_path / "dataset" / "daphnia_ext"
    dataset_raw = root / "raw"
    dataset_raw.mkdir(parents=True, exist_ok=True)
    for name in ("data-and-code.zip", "README_for_data-and-code.txt"):
        (tmp_path / "raw" / name).replace(dataset_raw / name)
    run_pipeline(daphnia_config(), root, process, extra_validator=validate_annotation)

    bundle = get_dataset("daphnia_ext", {"data_root": str(tmp_path / "dataset")})
    assert bundle["meta"]["bif_type"] == "transcritical"
    assert bundle["meta"]["n_replicates"] == 6
    assert bundle["signal"]["features"].shape[0] == 3
    assert bundle["null"]["features"].shape[0] == 3


def test_parse_readme_key_value() -> None:
    from csd_observer.datasets.daphnia_ext.process import parse_readme

    text = """replicate: X1
treatment = Treatment
extinction_day: 222
replicate: C1
treatment: control
"""
    parsed = parse_readme(text)
    assert parsed["X1"] == {"treatment": "treatment", "extinction_day": 222}
    assert parsed["C1"] == {"treatment": "control"}
    assert "extinction_day" not in parsed["C1"]
