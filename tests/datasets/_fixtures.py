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


def build_custom_tac_raw(tmp_path: Path, n_stationary: int, n_ramp: int) -> Path:
    """Write ``<tmp>/raw/tac/Experimental_time_traces_tdms/*.tdms`` (the
    extracted on-disk layout under the folder-named raw dir). Returns the
    raw directory ``<tmp>/raw/tac``."""
    rng = np.random.default_rng(11)
    raw = tmp_path / "raw" / "tac"
    extracted = raw / "Experimental_time_traces_tdms"
    extracted.mkdir(parents=True, exist_ok=True)
    for i in range(n_stationary):
        path = extracted / f"2016-10-20_Stationary{i}.tdms"
        write_tdms(path, f"ch{i}", stationary_trace(rng, N))
    for i in range(n_ramp):
        path = extracted / f"2016-10-20_Ramp{i}.tdms"
        write_tdms(path, f"ch{i}", ramp_trace(rng, N))
    return raw


def tac_config(processing: dict | None = None, **extra: object) -> dict:
    cfg: dict = {
        "name": "tac",
        "source": "synthetic",  # run_pipeline skips raw ingestion
        "bif_type": "subcritical_hopf",
        "doi": "10.5061/dryad.4cj4k",
        "expected_files": [{"path": "Experimental_time_traces_tdms", "md5": ""}],
        "download": {"mode": "manual"},
        "processing": {"envelope": "raw", "min_length": 100},
        "split": {"replicate_based": True, "seed": 42, "train_frac": 0.6, "val_frac": 0.2},
    }
    if processing:
        cfg["processing"].update(processing)
    cfg.update(extra)
    return cfg


POSITIVE_REPS = frozenset({"A", "B", "E"})


def _date_str(start: str, day: int) -> str:
    from datetime import datetime, timedelta

    return (datetime.strptime(start, "%m/%d/%Y") + timedelta(days=day)).strftime("%m/%d/%Y")


def _daphnia_timeseries_cells(n_census: int = 40, *, sample_base: float = 10.0) -> list[str]:
    """Census rows for populations A..F over ``n_census`` weekly dates.

    Positives (A/B/E) decay linearly to 0 at extinction day 63 (census
    index 9, matching the default extinctions.csv); nulls (C/D/F) hover
    near ``sample_base``. Each population has 2 subpops, so the
    chamber-level count (sum over subpops of mean(sample1..3)) is
    ~2*sample_base while alive.
    """
    rows = ["ID,Subpop,Date,sample1,sample2,sample3"]
    start = "10/27/2008"
    for rep in ("A", "B", "C", "D", "E", "F"):
        for index in range(n_census):
            day = index * 7
            date = _date_str(start, day)
            for subpop in (1, 2):
                if rep in POSITIVE_REPS:
                    s = sample_base * (1.0 - day / 63.0) if day < 63 else 0.0
                else:
                    s = sample_base + (index % 3)
                rows.append(f"{rep},{subpop},{date},{s:.1f},{s:.1f},{s:.1f}")
    return rows


def build_daphnia_raw(tmp_path: Path, *, csv_text: str | None = None,
                      extinctions_text: str | None = None) -> Path:
    """Write ``<tmp>/raw/daphnia/data-and-code/{timeseries,extinctions}.csv``
    (the folder-named raw dir). Returns the raw directory ``<tmp>/raw/daphnia``.

    ``csv_text`` is the ``timeseries.csv`` body (header ``ID,Subpop,Date,sample1..3``);
    ``extinctions_text`` the ``extinctions.csv`` body
    (header ``ID,Start,End,Deteriorating``). Defaults: 40 weekly census
    days; positives A/B/E extinct 12/29/2008 (day 63), nulls survive.
    """
    raw = tmp_path / "raw" / "daphnia"
    data_dir = raw / "data-and-code"
    data_dir.mkdir(parents=True, exist_ok=True)
    if csv_text is None:
        csv_text = "\n".join(_daphnia_timeseries_cells())
    if extinctions_text is None:
        extinctions_text = "\n".join(
            ["ID,Start,End,Deteriorating",
             'A,10/27/2008,12/29/2008,1',
             'B,10/27/2008,12/29/2008,1',
             'C,10/27/2008,1/26/2009,0',
             'D,10/27/2008,1/26/2009,0',
             'E,10/27/2008,12/29/2008,1',
             'F,10/27/2008,1/26/2009,0']
        )
    (data_dir / "timeseries.csv").write_text(csv_text, encoding="utf-8")
    (data_dir / "extinctions.csv").write_text(extinctions_text, encoding="utf-8")
    return raw


def daphnia_config(processing: dict | None = None, **extra: object) -> dict:
    cfg: dict = {
        "name": "daphnia_ext",
        "source": "synthetic",  # run_pipeline skips raw ingestion
        "bif_type": "transcritical",
        "doi": "10.5061/dryad.q3p64",
        "expected_files": [
            {"path": "data-and-code", "md5": ""},
        ],
        "download": {"mode": "manual"},
        "processing": {
            "min_length": 20,
            "data_file": "timeseries.csv",
            "extinctions_file": "extinctions.csv",
            "tau_annotation_days": 14,
            "restart_ids": ["H7", "H9", "J4", "K2", "K10"],
            "restart_threshold_day": 154,
        },
        "split": {"replicate_based": True, "seed": 42, "train_frac": 0.6, "val_frac": 0.2},
    }
    if processing:
        cfg["processing"].update(processing)
    cfg.update(extra)
    return cfg
