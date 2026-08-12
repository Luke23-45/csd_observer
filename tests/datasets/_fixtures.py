"""Shared fixtures for the real-dataset processor tests (synthetic archives)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

N = 256  # >= pipeline MIN_LENGTH 100


def write_tdms(path: Path, name: str, data: np.ndarray) -> None:
    from nptdms import ChannelObject, TdmsWriter

    with TdmsWriter(str(path)) as writer:
        writer.write_segment([ChannelObject("Raw", name, data.astype(np.float64))])


def stationary_trace(rng: np.random.Generator, length: int) -> np.ndarray:
    t = np.arange(length)
    return 1.0 * np.cos(0.5 * t) + 0.05 * rng.normal(size=length)


def ramp_trace(rng: np.random.Generator, length: int) -> np.ndarray:
    t = np.arange(length)
    amplitude = 0.1 + 1.0 * (t / length) ** 2
    return amplitude * np.cos(0.5 * t) + 0.05 * rng.normal(size=length)


def build_custom_tac_archive(tmp_path: Path, n_stationary: int, n_ramp: int) -> Path:
    """Write ``raw/Experimental_time_traces_tdms.zip`` with Stationary/Ramp files."""
    rng = np.random.default_rng(11)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "Experimental_time_traces_tdms.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(n_stationary):
            path = tmp_path / f"2016-10-20_Stationary{i}.tdms"
            write_tdms(path, f"ch{i}", stationary_trace(rng, N))
            zf.write(path, path.name)
        for i in range(n_ramp):
            path = tmp_path / f"2016-10-20_Ramp{i}.tdms"
            write_tdms(path, f"ch{i}", ramp_trace(rng, N))
            zf.write(path, path.name)
    return archive


def tac_config(processing: dict | None = None, **extra: object) -> dict:
    cfg: dict = {
        "name": "tac",
        "source": "synthetic",  # run_pipeline skips raw ingestion
        "bif_type": "subcritical_hopf",
        "doi": "10.5061/dryad.4cj4k",
        "expected_files": [{"path": "Experimental_time_traces_tdms.zip", "md5": "0" * 32}],
        "download": {"mode": "manual"},
        "processing": {"envelope": "raw", "min_length": 100},
        "split": {"replicate_based": True, "seed": 42, "train_frac": 0.6, "val_frac": 0.2},
    }
    if processing:
        cfg["processing"].update(processing)
    cfg.update(extra)
    return cfg


POSITIVE_REPS = frozenset({"A", "B", "E"})


def _counts_csv(spans: list[tuple[str, int, float]]) -> str:
    rows = ["replicate,day,count,treatment"]
    for replicate, days, base in spans:
        treatment = "positive" if replicate in POSITIVE_REPS else "control"
        for day in range(days + 1):
            decay = max(0.0, 1.0 - day / (days + 1))
            if day < days:
                rows.append(f"{replicate},{day},{base * decay:.1f},{treatment}")
            elif replicate in POSITIVE_REPS:
                rows.append(f"{replicate},{day},0.0,{treatment}")
            else:
                rows.append(f"{replicate},{day},{base * decay:.1f},{treatment}")
    return "\n".join(rows)


def build_daphnia_fixture(tmp_path: Path, *, readme_text: str = "", csv_text: str | None = None) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "data-and-code.zip"
    default_csv = _counts_csv(
        [("A", 300, 40.0), ("B", 300, 45.0), ("C", 200, 120.0), ("D", 250, 110.0), ("E", 300, 35.0), ("F", 220, 100.0)]
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("counts.csv", csv_text or default_csv)
    (raw / "README_for_data-and-code.txt").write_text(readme_text, encoding="utf-8")
    return archive


DEFAULT_README = """Experiment: Daphnia magna extinction microcosms
replicate: A
treatment: positive
extinction_day: 300
replicate: B
treatment: positive
extinction_day: 300
replicate: E
treatment: positive
extinction_day: 300
"""


def daphnia_config(processing: dict | None = None, **extra: object) -> dict:
    cfg: dict = {
        "name": "daphnia_ext",
        "source": "synthetic",  # run_pipeline skips raw ingestion
        "bif_type": "transcritical",
        "doi": "10.5061/dryad.q3p64",
        "expected_files": [
            {"path": "data-and-code.zip", "md5": "0" * 32},
            {"path": "README_for_data-and-code.txt", "md5": "0" * 32},
        ],
        "download": {"mode": "manual"},
        "processing": {
            "min_length": 100,
            "replicate_column": "replicate",
            "time_column": "day",
            "count_column": "count",
            "positive_column": "treatment",
            "tau_annotation_days": 110,
        },
        "split": {"replicate_based": True, "seed": 42, "train_frac": 0.6, "val_frac": 0.2},
    }
    if processing:
        cfg["processing"].update(processing)
    cfg.update(extra)
    return cfg
