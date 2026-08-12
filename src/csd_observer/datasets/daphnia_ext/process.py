"""DaphniaExt table processing with explicit column/treatment mappings.

Two layers:

* ``process_table`` — strict, low-level rows-to-array conversion (equal
  aligned lengths, explicit labels; no inference);
* ``process`` — the §5.4 archive processor: README treatment coding +
  per-replicate daily counts, aligned on the transition (t=0 at
  extinction for positives, at the last recorded day for nulls) with
  ``bifurcation_times`` set to the CSD-annotation window offset
  (``processing.tau_annotation_days``, default 110 per Nature 467:456)
  relative to the aligned origin.

The README coding (R3 known-unknown) is parsed with explicit key:value
patterns and never guessed: a replicate without a treatment label from
either the README or ``processing.treatments`` raises
``PROCESS_ANNOTATION_MISSING``; the effective mapping is recorded in the
manifest so layout changes force re-processing.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.ingest import extract_archive

_MIN_LENGTH_DEFAULT = 100

# Generic ``key: value`` / ``key = value`` lines; the config
# ``processing.treatments`` map overrides and is the escape hatch for an
# unanticipated README layout (recorded at L3.18).
_README_KEY_VALUE = re.compile(r"^\s*(replicate|treatment|extinction[_\s]?day|extinction)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)

_TRUE_TOKENS = {"true", "1", "yes", "positive", "treatment", "extinct", "extinction"}
_FALSE_TOKENS = {"false", "0", "no", "null", "control", "negative", "constant"}


def process_table(rows: list[dict[str, Any]], *, replicate_column: str, time_column: str,
                  count_column: str, positive_column: str, min_length: int = 100) -> dict[str, np.ndarray]:
    """Convert already parsed rows; no treatment labels are inferred."""
    required = {replicate_column, time_column, count_column, positive_column}
    if not rows or not required.issubset(rows[0]):
        raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, f"missing explicit columns: {sorted(required)}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[replicate_column]), []).append(row)
    features: list[np.ndarray] = []
    positive: list[bool] = []
    for replicate, group in sorted(groups.items()):
        group = sorted(group, key=lambda r: float(r[time_column]))
        if len(group) < min_length:
            raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, f"replicate {replicate} has {len(group)} rows")
        try:
            values = np.asarray([float(r[count_column]) for r in group], dtype=np.float32)
            label_values = {str(r[positive_column]).strip().lower() for r in group}
        except (TypeError, ValueError) as exc:
            raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, str(exc)) from exc
        if len(label_values) != 1 or not label_values <= (_TRUE_TOKENS | _FALSE_TOKENS):
            raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, f"replicate {replicate} has inconsistent treatment labels")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"replicate {replicate} has invalid counts")
        features.append(values[:, None])
        positive.append(label_values.pop() in _TRUE_TOKENS)
    length = {len(x) for x in features}
    if len(length) != 1:
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, "replicates must have equal aligned lengths")
    return {"features": np.stack(features), "seq_lengths": np.full(len(features), length.pop(), dtype=np.int64),
            "is_positive": np.asarray(positive, dtype=bool)}


def parse_readme(text: str) -> dict[str, dict[str, Any]]:
    """Extract per-replicate ``{treatment, extinction_day}`` from README text.

    Accepts ``key: value`` / ``key = value`` lines (case-insensitive
    keys: ``replicate``, ``treatment``, ``extinction_day``). Returns only
    what is found; callers combine this with ``processing.treatments``
    and must fail loudly on missing labels (no guessing).
    """
    out: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    current_id: str | None = None
    for line in text.splitlines():
        match = _README_KEY_VALUE.match(line)
        if not match:
            continue
        key, value = match.group(1).lower(), match.group(2).strip()
        if key == "replicate":
            current_id = value
            current = out.setdefault(current_id, {})
        elif current is None:
            continue
        elif key == "treatment":
            current["treatment"] = value.lower()
        elif key in ("extinction_day", "extinction"):
            try:
                current["extinction_day"] = int(float(value))
            except ValueError as exc:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"unparseable extinction day {value!r}",
                ) from exc
    return out


def _read_table(data_file: Path) -> list[dict[str, Any]]:
    import pandas as pd

    delimiters = [",", "\t", ";", " ", "|"]
    for delimiter in delimiters:
        try:
            frame = pd.read_csv(data_file, delimiter=delimiter, encoding="utf-8", on_bad_lines="error")
        except Exception:
            continue
        if len(frame.columns) > 1:
            return [dict(row) for row in frame.to_dict(orient="records")]
    raise DatasetError(
        DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
        f"cannot parse table {data_file.name} with any supported delimiter",
    )


def _nearest_index(times: np.ndarray, target: float) -> int:
    idx = int(np.argmin(np.abs(times - target)))
    return idx


def process(raw_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """§5.4 DaphniaExt processor handle: ``raw_dir -> uniform bundle``."""
    raw = Path(raw_dir)
    processing = dict(config.get("processing", {}) or {})
    min_length = int(processing.get("min_length", _MIN_LENGTH_DEFAULT))
    tau_days = int(processing.get("tau_annotation_days", 110))
    window_days = processing.get("window_days")
    if window_days is not None:
        window_days = int(window_days)
        if window_days < min_length:
            raise ValueError(f"processing.window_days {window_days} < min_length {min_length}")

    specs = {str(spec["path"]): spec for spec in config.get("expected_files", [])}
    if not specs:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "daphnia_ext config has no expected_files")
    readme_path = None
    archive_path = None
    for name in specs:
        candidate = raw / name
        if not candidate.exists():
            raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"missing raw file: {candidate}")
        lowered = name.lower()
        if "readme" in lowered or lowered.endswith(".txt"):
            readme_path = candidate
        else:
            archive_path = candidate
    if archive_path is None or not zipfile.is_zipfile(archive_path):
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"expected a data ZIP archive, got {archive_path}")

    # --- treatment coding: README patterns + explicit config map ---
    annotations: dict[str, dict[str, Any]] = {}
    if readme_path is not None:
        try:
            annotations = parse_readme(readme_path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"cannot read README: {exc}") from exc
    configured = dict(processing.get("treatments", {}) or {})
    for replicate, mapping in configured.items():
        annotations.setdefault(str(replicate), {})
        for key, value in mapping.items():
            annotations[str(replicate)][str(key).lower()] = value

    # --- data table ---
    extraction_dir = raw / "_extracted"
    marker = extraction_dir / ".extracted.ok"
    if not marker.is_file():
        extract_archive(archive_path, extraction_dir)
        marker.write_text("ok", encoding="utf-8")
    data_file_name = processing.get("data_file")
    candidates = sorted(p for p in extraction_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".tsv"})
    if data_file_name:
        matches = [p for p in candidates if p.name == data_file_name]
        if not matches:
            raise DatasetError(
                DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
                f"processing.data_file {data_file_name!r} not found in archive; available: {[p.name for p in candidates]}",
            )
        data_file = matches[0]
    else:
        if not candidates:
            raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "archive contains no CSV/TSV/TXT table")
        data_file = candidates[0]
        if data_file_name is None:
            processing["data_file"] = data_file.name

    rows = _read_table(data_file)
    if not rows:
        raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, "data table is empty")

    rep_col = str(processing.get("replicate_column", "replicate"))
    time_col = str(processing.get("time_column", "day"))
    count_col = str(processing.get("count_column", "count"))
    positive_col = str(processing.get("positive_column", "treatment"))
    required = {rep_col, time_col, count_col, positive_col}
    available = set(rows[0])
    missing = required - available
    if missing:
        raise DatasetError(
            DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
            f"table {data_file.name} lacks columns {sorted(missing)}; available: {sorted(available)}",
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[rep_col]), []).append(row)

    features: list[np.ndarray] = []
    lengths: list[int] = []
    bifs: list[np.ndarray] = []
    positives: list[bool] = []
    used_annotations: dict[str, Any] = {}

    for replicate in sorted(groups):
        group = sorted(groups[replicate], key=lambda r: float(r[time_col]))
        try:
            times = np.asarray([float(r[time_col]) for r in group], dtype=np.float64)
            values = np.asarray([float(r[count_col]) for r in group], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise DatasetError(DatasetErrorCode.PROCESS_ANNOTATION_MISSING, str(exc)) from exc
        if not np.isfinite(values).all() or np.any(values < 0):
            raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"replicate {replicate} has invalid counts")

        annotation = annotations.get(replicate, {})
        label_values = {str(r[positive_col]).strip().lower() for r in group}
        if "treatment" in annotation:
            treatment = str(annotation["treatment"]).lower()
            if treatment not in (_TRUE_TOKENS | _FALSE_TOKENS):
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"replicate {replicate}: unknown treatment token {treatment!r}",
                )
            is_positive = treatment in _TRUE_TOKENS
        else:
            if not label_values:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"replicate {replicate} has no treatment label (README or processing.treatments)",
                )
            if len(label_values) > 1 or not label_values <= (_TRUE_TOKENS | _FALSE_TOKENS):
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"replicate {replicate} has inconsistent treatment labels {sorted(label_values)}",
                )
            is_positive = label_values.pop() in _TRUE_TOKENS
            treatment = "positive" if is_positive else "null"
        used_annotations[replicate] = {"treatment": treatment}

        if is_positive:
            extinction_day = annotation.get("extinction_day")
            if extinction_day is None:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"positive replicate {replicate} lacks an extinction day "
                    f"(README or processing.treatments.{replicate}.extinction_day)",
                )
            t0 = float(extinction_day)
            used_annotations[replicate]["extinction_day"] = t0
        else:
            t0 = float(np.max(times))

        if window_days is None:
            selected = group
        else:
            selected = [r for r in group if t0 - window_days <= float(r[time_col]) <= t0]
        if len(selected) < min_length:
            raise DatasetError(
                DatasetErrorCode.PROCESS_SHORT_LENGTH,
                f"replicate {replicate} has {len(selected)} samples in the analysis window (< {min_length})",
            )
        times_sel = np.asarray([float(r[time_col]) for r in selected], dtype=np.float64)
        values_sel = np.asarray([float(r[count_col]) for r in selected], dtype=np.float32)
        if not np.isfinite(values_sel).all() or np.any(values_sel < 0):
            raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"replicate {replicate} has invalid counts in window")

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
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "no trajectories produced from DaphniaExt archive")
    if not any(positives) or all(positives):
        raise DatasetError(
            DatasetErrorCode.SPLIT_IMBALANCE,
            f"DaphniaExt needs both positive and null replicates, got "
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
                "data_file": data_file.name,
                "annotations": used_annotations,
                "readme_parsed": readme_path is not None,
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


__all__ = ["parse_readme", "process", "process_table", "validate_annotation"]
