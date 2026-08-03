"""Data loading and aggregation for the CSD observer benchmark."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.visualize.common.constants import METHODS, PATIENT_COUNTS, SYSTEMS, SEED_ORDER

logger = logging.getLogger(__name__)

_PATIENT_RE = re.compile(r"patients_(\d+)")
_METHOD_FILE_MAP = {
    "Kalman-BCE": "kalman_bce",
    "Kalman-LSTM-Spec": "kalman_lstm_spec",
    "Kalman-LSTM": "kalman_lstm",
    "Kalman-BCE-Spec": "kalman_bce_spec",
}


@dataclass(frozen=True)
class BenchmarkBatch:
    patient_count: int
    batch_id: str
    batch_dir: Path
    results_file: Path
    n_records: int
    n_unique_triples: int
    n_systems: int
    timestamp: datetime


@dataclass(frozen=True)
class BenchmarkStore:
    data_root: Path
    all_records: pd.DataFrame
    records: pd.DataFrame
    batch_index: pd.DataFrame
    selected_batches: Dict[int, BenchmarkBatch]

    def batch_dir(self, patient_count: int) -> Path:
        return self.selected_batches[patient_count].batch_dir

    def results_file(self, patient_count: int) -> Path:
        return self.selected_batches[patient_count].results_file

    def batch_records(self, patient_count: int) -> pd.DataFrame:
        return self.records[self.records["patient_count"] == patient_count].copy()

    def filtered(self, *, patient_count: Optional[int] = None, system: Optional[str] = None, method: Optional[str] = None) -> pd.DataFrame:
        df = self.records
        if patient_count is not None:
            df = df[df["patient_count"] == patient_count]
        if system is not None:
            df = df[df["system"] == system]
        if method is not None:
            df = df[df["method"] == method]
        return df.copy()

    def representative_seed(self, patient_count: int, system: str, anchor_method: str = "Kalman-BCE") -> int:
        """Return the seed whose anchor-method AUC is closest to the median."""
        df = self.filtered(patient_count=patient_count, system=system, method=anchor_method)
        if df.empty:
            raise ValueError(f"No records found for {patient_count=} {system=} {anchor_method=}")
        med = float(df["ew_auc"].median())
        order = df.assign(distance=(df["ew_auc"] - med).abs()).sort_values(
            ["distance", "seed"],
            ascending=[True, True],
        )
        return int(order.iloc[0]["seed"])

    def run_path(self, patient_count: int, system: str, method: str, seed: int, kind: str = "trajectories") -> Path:
        batch_dir = self.batch_dir(patient_count)
        method_slug = _METHOD_FILE_MAP[method]
        return batch_dir / "results" / kind / f"{system}_{method_slug}_seed{seed}.npz"

    def epoch_log_path(self, patient_count: int, system: str, method: str, seed: int) -> Path:
        batch_dir = self.batch_dir(patient_count)
        method_slug = _METHOD_FILE_MAP[method]
        return batch_dir / "results" / "epoch_logs" / f"{system}_{method_slug}_seed{seed}.csv"

    def load_trajectory(self, patient_count: int, system: str, method: str, seed: int) -> Dict[str, np.ndarray]:
        path = self.run_path(patient_count, system, method, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        return dict(np.load(path))

    def load_epoch_log(self, patient_count: int, system: str, method: str, seed: int) -> pd.DataFrame:
        path = self.epoch_log_path(patient_count, system, method, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)


def _extract_patient_count(path: Path) -> int:
    match = _PATIENT_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not infer patient count from path: {path}")
    return int(match.group(1))


def _extract_batch_dir(results_file: Path) -> Path:
    return results_file.parent.parent


def _parse_timestamp(batch_id: str) -> datetime:
    return datetime.strptime(batch_id, "%Y-%m-%d_%H-%M-%S")


def _load_results_file(results_file: Path) -> pd.DataFrame:
    rows = []
    with results_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["patient_count"] = _extract_patient_count(results_file)
    df["batch_id"] = _extract_batch_dir(results_file).name
    df["batch_dir"] = str(_extract_batch_dir(results_file))
    df["results_file"] = str(results_file)
    df["timestamp"] = pd.to_datetime(df["batch_id"], format="%Y-%m-%d_%H-%M-%S", errors="coerce")
    return df


def _build_batch_index(all_records: pd.DataFrame) -> pd.DataFrame:
    if all_records.empty:
        return pd.DataFrame(
            columns=[
                "patient_count",
                "batch_id",
                "batch_dir",
                "results_file",
                "n_records",
                "n_unique_triples",
                "n_systems",
                "timestamp",
            ]
        )

    grouped = []
    for (patient_count, batch_id), df in all_records.groupby(["patient_count", "batch_id"], sort=False):
        batch_dir = Path(df["batch_dir"].iloc[0])
        results_file = Path(df["results_file"].iloc[0])
        grouped.append(
            {
                "patient_count": int(patient_count),
                "batch_id": batch_id,
                "batch_dir": str(batch_dir),
                "results_file": str(results_file),
                "n_records": int(len(df)),
                "n_unique_triples": int(df[["system", "method", "seed"]].drop_duplicates().shape[0]),
                "n_systems": int(df["system"].nunique()),
                "timestamp": df["timestamp"].iloc[0],
            }
        )
    return pd.DataFrame(grouped)


def _select_batches(batch_index: pd.DataFrame) -> Dict[int, BenchmarkBatch]:
    selected: Dict[int, BenchmarkBatch] = {}
    if batch_index.empty:
        return selected

    for patient_count in PATIENT_COUNTS:
        subset = batch_index[batch_index["patient_count"] == patient_count].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["n_unique_triples", "n_systems", "n_records", "timestamp"],
            ascending=[False, False, False, False],
        )
        row = subset.iloc[0]
        selected[patient_count] = BenchmarkBatch(
            patient_count=int(row["patient_count"]),
            batch_id=str(row["batch_id"]),
            batch_dir=Path(row["batch_dir"]),
            results_file=Path(row["results_file"]),
            n_records=int(row["n_records"]),
            n_unique_triples=int(row["n_unique_triples"]),
            n_systems=int(row["n_systems"]),
            timestamp=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
        )
    return selected


def load_benchmark_store(data_root: Path) -> BenchmarkStore:
    """Load all benchmark results and keep the most complete batch per patient count."""
    result_files = sorted(data_root.rglob("results.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No results.jsonl files found under {data_root}")

    frames = []
    for results_file in result_files:
        df = _load_results_file(results_file)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(f"No benchmark rows could be parsed from {data_root}")

    all_records = pd.concat(frames, ignore_index=True)
    batch_index = _build_batch_index(all_records)
    selected_batches = _select_batches(batch_index)

    if not selected_batches:
        raise ValueError("Could not identify a complete batch for any patient count.")

    selected_frames = []
    for patient_count, batch in selected_batches.items():
        selected_frames.append(
            all_records[
                (all_records["patient_count"] == patient_count)
                & (all_records["batch_id"] == batch.batch_id)
            ]
        )
    records = pd.concat(selected_frames, ignore_index=True).copy()

    # Stable row ordering for downstream tables/plots.
    records["system"] = pd.Categorical(records["system"], categories=SYSTEMS, ordered=True)
    records["method"] = pd.Categorical(records["method"], categories=METHODS, ordered=True)
    records["seed"] = pd.Categorical(records["seed"], categories=SEED_ORDER, ordered=True)
    records = records.sort_values(["patient_count", "system", "method", "seed"]).reset_index(drop=True)

    logger.info("Loaded %d benchmark records from %d batches.", len(records), len(selected_batches))
    return BenchmarkStore(
        data_root=data_root,
        all_records=all_records,
        records=records,
        batch_index=batch_index,
        selected_batches=selected_batches,
    )
