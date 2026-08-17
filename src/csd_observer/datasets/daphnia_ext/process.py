"""DaphniaExt real-data processing (§5.4) over the Drake & Griffen (2010)
``timeseries.csv`` / ``extinctions.csv`` tables.

The processor reproduces the aggregation of the paper's ``preprocess.R``:

* per census row ``y = mean(sample1..3)``;
* ``day`` = days since the experiment start (first census date);
* restart populations ``H7,H9,J4,K2,K10`` are re-identified as ``ID+"2"``
  for ``day >= restart_threshold_day`` (the paper uses 154), so the first
  (pre-treatment, extinct) attempt and the restarted population become
  distinct trajectories;
* ``populations = aggregate(sum of y by (ID2, day))`` — the chamber-level
  daily count is the *sum over subpops* (both Subpop=1 and Subpop=2).

Labels come from ``extinctions.csv``: ``Deteriorating`` (1 = signal,
0 = null) and the extinction day ``End - Start`` in days. The extinction
table uses an ``a`` suffix for the first (pre-treatment) attempt of a
restart population while the recoded time series uses a ``2`` suffix for
the restarted population, so the two label rows map as:

* ``H7a`` -> first attempt ``H7`` (rows with ``day < restart_threshold_day``);
* ``H7`` -> restarted population ``H72`` (rows with ``day >= threshold``).

Non-restart IDs map directly. A population with no matching extinction
record (or vice versa) fails loudly (``PROCESS_ANNOTATION_MISSING``);
the effective mapping is recorded in the manifest.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode

_MIN_LENGTH_DEFAULT = 20

_RESTART_IDS = ["H7", "H9", "J4", "K2", "K10"]
_RESTART_THRESHOLD_DAY = 154


def _parse_date(value: str) -> datetime:
    return datetime.strptime(str(value).strip(), "%m/%d/%Y")


def _days_since(value: str, start: datetime) -> int:
    return int((_parse_date(value) - start).days)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    with open(path, encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [str(h).strip() for h in reader.fieldnames]
        rows = [dict(r) for r in reader]
    if not rows:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, f"empty table {path.name}")
    return rows


def _aggregate_timeseries(rows: list[dict[str, Any]], *, restart_ids: list[str],
                          restart_threshold_day: int, start: datetime) -> dict[str, list[tuple[int, float]]]:
    """Aggregate ``timeseries.csv`` to chamber-level daily counts.

    Returns ``{population_id: [(day, count), ...]}`` after summing the
    per-subpop mean of ``sample1..3`` on each census day.
    """
    pooled: dict[str, dict[int, float]] = {}
    for row in rows:
        population = str(row["ID"]).strip()
        day = _days_since(row["Date"], start)
        try:
            y = (float(row["sample1"]) + float(row["sample2"]) + float(row["sample3"])) / 3.0
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(
                DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                f"timeseries row for {population} lacks numeric sample1..3: {exc}",
            ) from exc
        if not np.isfinite(y):
            raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"non-finite count for {population} on day {day}")
        if population in restart_ids and day >= restart_threshold_day:
            population = f"{population}2"
        pooled.setdefault(population, {})
        pooled[population][day] = pooled[population].get(day, 0.0) + y
    return {population: sorted(days.items()) for population, days in pooled.items()}


def _population_label(ex_id: str, *, restart_ids: list[str]) -> str:
    """Map an ``extinctions.csv`` ID to the aggregated population ID."""
    base = ex_id[:-1] if ex_id.endswith("a") else ex_id
    if ex_id.endswith("a") and base in restart_ids:
        return base
    if ex_id in restart_ids:
        return f"{ex_id}2"
    return ex_id


def _read_extinctions(rows: list[dict[str, Any]], *, restart_ids: list[str],
                      experiment_start: datetime) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        ex_id = str(row["ID"]).strip()
        pop = _population_label(ex_id, restart_ids=restart_ids)
        try:
            start = _parse_date(row["Start"])
            end = _parse_date(row["End"])
        except (KeyError, ValueError) as exc:
            raise DatasetError(
                DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                f"extinctions row for {ex_id} lacks parseable Start/End: {exc}",
            ) from exc
        deteriorating = str(row["Deteriorating"]).strip().lower()
        if deteriorating not in {"1", "0", "true", "false"}:
            raise DatasetError(
                DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                f"extinctions row for {ex_id} has unknown Deteriorating {deteriorating!r}",
            )
        if pop in labels:
            raise DatasetError(
                DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                f"duplicate extinction label for population {pop} ({ex_id} and {labels[pop]['extinction_id']})",
            )
        labels[pop] = {
            "extinction_id": ex_id,
            "is_positive": deteriorating in {"1", "true"},
            # ``t0`` shares the census-day axis (experiment-relative) so the
            # tau window lands inside the trajectory even for restarted
            # populations (whose EX ``Start`` is the restart date, not the
            # experiment start).
            "extinction_day": int((end - experiment_start).days),
            "start": row["Start"].strip(),
            "end": row["End"].strip(),
        }
    return labels


def _nearest_index(times: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(times - target)))


def process(raw_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """§5.4 DaphniaExt processor handle: ``raw_dir -> uniform bundle``.

    Reads ``timeseries.csv`` + ``extinctions.csv`` from the extracted
    ``data-and-code`` layout under ``raw_dir`` (configurable via
    ``processing.data_file`` / ``processing.extinctions_file``).
    """
    raw = Path(raw_dir)
    processing = dict(config.get("processing", {}) or {})
    min_length = int(processing.get("min_length", _MIN_LENGTH_DEFAULT))
    tau_days = int(processing.get("tau_annotation_days", 110))
    restart_ids = list(processing.get("restart_ids", _RESTART_IDS))
    restart_threshold_day = processing.get("restart_threshold_day")
    if restart_threshold_day is None:
        restart_threshold_day = _RESTART_THRESHOLD_DAY
    restart_threshold_day = int(restart_threshold_day)
    window_days = processing.get("window_days")
    if window_days is not None:
        window_days = int(window_days)
        if window_days < min_length:
            raise ValueError(f"processing.window_days {window_days} < min_length {min_length}")

    specs = {str(spec["path"]): spec for spec in config.get("expected_files", [])}
    if not specs:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "daphnia_ext config has no expected_files")
    data_root = raw
    for name in specs:
        candidate = raw / str(name)
        if candidate.is_dir():
            data_root = candidate
            break
    ts_name = str(processing.get("data_file", "timeseries.csv"))
    ex_name = str(processing.get("extinctions_file", "extinctions.csv"))
    ts_path = data_root / ts_name
    ex_path = data_root / ex_name
    for path, what in ((ts_path, "timeseries"), (ex_path, "extinctions")):
        if not path.is_file():
            raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"missing {what} table: {path}")

    ts_rows = _read_csv(ts_path)
    if not ts_rows:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "empty timeseries table")
    required_ts = {"ID", "Subpop", "Date", "sample1", "sample2", "sample3"}
    missing_ts = sorted(required_ts - set(ts_rows[0]))
    if missing_ts:
        raise DatasetError(
            DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
            f"timeseries table missing columns: {missing_ts}",
        )
    dates = sorted(_parse_date(str(r["Date"]).strip()) for r in ts_rows)
    start = dates[0]
    populations = _aggregate_timeseries(
        ts_rows, restart_ids=restart_ids, restart_threshold_day=restart_threshold_day, start=start,
    )
    labels = _read_extinctions(_read_csv(ex_path), restart_ids=restart_ids, experiment_start=start)

    missing_labels = sorted(set(populations) - set(labels))
    if missing_labels:
        raise DatasetError(
            DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
            f"populations without an extinction label: {missing_labels}",
        )
    orphan_labels = sorted(set(labels) - set(populations))
    if orphan_labels:
        raise DatasetError(
            DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
            f"extinction labels without a matching population: {orphan_labels}",
        )

    features: list[np.ndarray] = []
    lengths: list[int] = []
    bifs: list[np.ndarray] = []
    positives: list[bool] = []
    used_labels: dict[str, Any] = {}
    excluded_short: list[dict[str, Any]] = []

    for population in sorted(populations):
        days, counts = zip(*populations[population], strict=True)
        times = np.asarray(days, dtype=np.float64)
        values = np.asarray(counts, dtype=np.float32)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"population {population} has invalid counts")

        label = labels[population]
        is_positive = bool(label["is_positive"])
        used_labels[population] = {
            "extinction_id": label["extinction_id"],
            "is_positive": is_positive,
            "extinction_day": label["extinction_day"],
        }

        t0 = float(label["extinction_day"])
        if is_positive and t0 < times[0]:
            raise DatasetError(
                DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                f"positive population {population} has extinction_day {t0} before its first census {times[0]}",
            )
        if window_days is None:
            selected = list(range(len(times)))
        else:
            selected = [i for i in range(len(times)) if t0 - window_days <= times[i] <= t0]
        if len(selected) < min_length:
            # Exclude too-short populations (e.g. dead first attempts before
            # a restart) and record them; the length gate is checked later by
            # the validate step on the surviving trajectories.
            excluded_short.append({
                "population": population,
                "census_days": len(selected),
                "min_length": min_length,
                "is_positive": is_positive,
            })
            continue
        times_sel = times[selected]
        values_sel = values[selected]

        if is_positive:
            tau_index = _nearest_index(times_sel, t0 - tau_days)
            bif = float(tau_index)
        else:
            bif = float(len(selected) + 1)

        features.append(values_sel[:, None])
        lengths.append(len(selected))
        bifs.append(np.asarray([bif], dtype=np.float64))
        positives.append(is_positive)

    if not features:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "no trajectories produced from DaphniaExt data")
    if not any(positives) or all(positives):
        raise DatasetError(
            DatasetErrorCode.SPLIT_IMBALANCE,
            f"DaphniaExt needs both positive and null populations, got "
            f"{sum(positives)} positive / {len(positives) - sum(positives)} null",
        )

    max_len = max(lengths)
    padded = np.zeros((len(features), max_len, 1), dtype=np.float32)
    for i, feature in enumerate(features):
        padded[i, : lengths[i], 0] = feature[:, 0]

    return {
        "features": padded,
        "seq_lengths": np.asarray(lengths, dtype=np.int64),
        "bifurcation_times": np.concatenate(bifs, axis=0),
        "is_positive": np.asarray(positives, dtype=bool),
        # Provenance (recorded under manifest.processing.effective; NOT
        # part of processing.params so the staleness guard compares only
        # user-controlled keys).
        "meta": {
            "processing": {
                "start_date": start.strftime("%m/%d/%Y"),
                "restart_ids": sorted(restart_ids),
                "restart_threshold_day": restart_threshold_day,
                "data_file": ts_name,
                "extinctions_file": ex_name,
                "labels": used_labels,
                "excluded_short": excluded_short,
            }
        },
    }


def validate_annotation(bundle: dict[str, Any]) -> None:
    """DaphniaExt annotation spot-check (§5.4 gate 3): the CSD-annotation
    window (``tau``) must fall inside every positive trajectory and the
    null sentinel convention must hold."""
    bifs = np.asarray(bundle["bifurcation_times"], dtype=np.float64)
    lengths = np.asarray(bundle["seq_lengths"], dtype=np.int64)
    positive = np.asarray(bundle["is_positive"], dtype=bool)
    for i in range(len(bundle["features"])):
        if positive[i]:
            if not 0.0 <= bifs[i] < lengths[i]:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"positive trajectory {i} has tau index {bifs[i]} outside [0, {lengths[i]})",
                )
        else:
            if bifs[i] != lengths[i] + 1:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"null trajectory {i} has non-sentinel bifurcation_time {bifs[i]}",
                )


__all__ = ["process", "validate_annotation"]
