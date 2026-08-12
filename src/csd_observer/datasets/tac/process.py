"""TAC TDMS inspection, envelope extraction and the §5.4 archive processor.

The processor turns the verified raw archive into the uniform bundle the
pipeline writes to ``final_data/tac/processed``:

1. extract the archive (safe; cached under ``raw/_extracted``);
2. per TDMS file: select the channel (explicit config wins, else the
   longest finite numeric channel), read it, and classify the section
   (stationary operating point -> null trajectory, ramp section ->
   positive trajectory) from the configured filename patterns;
3. extract the dominant-mode envelope (analytic-signal amplitude; raw
   Hilbert envelope by default, band-limited option once the combustor
   eigenfrequency band is pinned from the data, §4 known-unknown R2);
4. chunk long traces into fixed non-overlapping analysis windows
   (``processing.analysis_window``) when configured — a trailing chunk
   shorter than ``min_length`` fails loudly (no silent drops, §5.4);
5. ``bifurcation_times``: ramp sections = ``ramp_onset_index`` relative
   to the chunk (rate-dependent transition delay, §4); stationary
   sections = ``seq_length + 1`` sentinel (same null convention as the
   synthetic generators).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.ingest import extract_archive

_MIN_LENGTH_DEFAULT = 100


def inspect_tdms(archive: str | Path, extraction_dir: str | Path) -> list[dict[str, Any]]:
    """Return group/channel metadata without selecting a channel implicitly."""
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "npTDMS is required to inspect TAC data") from exc
    files = _tdms_files(archive, extraction_dir)
    records: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() != ".tdms":
            continue
        try:
            tdms = TdmsFile.read(str(path))
            for group in tdms.groups():
                for channel in group.channels():
                    values = np.asarray(channel[:])
                    records.append({"file": str(path), "group": group.name, "channel": channel.name,
                                    "length": int(values.size), "dtype": str(values.dtype),
                                    "finite": bool(np.isfinite(values).all()) if values.size else True})
        except (OSError, ValueError, TypeError) as exc:
            raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"failed to parse {path}: {exc}") from exc
    if not records:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "archive contains no readable TDMS channels")
    return records


def extract_channel(tdms_path: str | Path, *, group: str, channel: str, min_length: int) -> np.ndarray:
    """Read one explicitly named numeric channel and enforce finite data."""
    try:
        from nptdms import TdmsFile
        values = np.asarray(TdmsFile.read(str(tdms_path))[group][channel][:], dtype=np.float32)
    except (ImportError, OSError, KeyError, ValueError, TypeError) as exc:
        code = DatasetErrorCode.INGEST_ARCHIVE if isinstance(exc, (ImportError, OSError)) else DatasetErrorCode.PROCESS_NONFINITE
        raise DatasetError(code, f"cannot read TDMS channel {group}/{channel}: {exc}") from exc
    if values.ndim != 1 or len(values) < int(min_length):
        raise DatasetError(DatasetErrorCode.PROCESS_SHORT_LENGTH, f"channel {group}/{channel} has length {len(values)}")
    if not np.isfinite(values).all():
        raise DatasetError(DatasetErrorCode.PROCESS_NONFINITE, f"channel {group}/{channel} contains non-finite values")
    return values


def auto_select_channel(tdms_path: str | Path, records: list[dict[str, Any]]) -> tuple[str, str, np.ndarray]:
    """Pick the longest finite numeric channel; never guesses silently.

    Raises when no finite numeric channel exists, and the choice is
    deterministic (ties break lexicographically) so re-processing is
    reproducible.
    """
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "npTDMS is required to read TAC data") from exc
    file_records = [r for r in records if Path(r["file"]) == Path(tdms_path)]
    numeric = [r for r in file_records if r["finite"] and "float" in str(r["dtype"])]
    if not numeric:
        raise DatasetError(
            DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
            f"no finite numeric channel in {tdms_path}; configure processing.channel_group/channel_name",
        )
    chosen = max(numeric, key=lambda r: (int(r["length"]), -ord(r["group"][0]) if r["group"] else 0, r["group"], r["channel"]))
    values = np.asarray(TdmsFile.read(str(tdms_path))[chosen["group"]][chosen["channel"]][:], dtype=np.float32)
    return chosen["group"], chosen["channel"], values


def bandpass_envelope(
    x: np.ndarray,
    *,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Band-limited analytic-signal envelope (Hilbert amplitude, §5.4).

    ``low_hz``/``high_hz`` must lie strictly inside the Nyquist band of
    ``sample_rate_hz``; config errors raise :class:`ValueError` (they are
    developer-side parameter errors, not data errors).
    """
    from scipy import signal

    if sample_rate_hz <= 0:
        raise ValueError(f"processing.sample_rate_hz must be positive, got {sample_rate_hz}")
    nyquist = sample_rate_hz / 2.0
    if low_hz <= 0 or high_hz >= nyquist or low_hz >= high_hz:
        raise ValueError(
            f"invalid band ({low_hz}, {high_hz}) Hz for fs={sample_rate_hz} "
            f"(need 0 < low < high < {nyquist})"
        )
    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    filtered = signal.sosfilt(sos, np.asarray(x, dtype=np.float64))
    envelope = np.abs(signal.hilbert(filtered))
    return np.asarray(envelope, dtype=np.float32)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Raw analytic-signal amplitude; the band-limited variant is
    ``bandpass_envelope`` (used once the eigenfrequency band is pinned)."""
    from scipy import signal

    return np.asarray(np.abs(signal.hilbert(np.asarray(x, dtype=np.float64))), dtype=np.float32)


def _extract_once(archive: Path, extraction_dir: Path) -> None:
    marker = extraction_dir / ".extracted.ok"
    if marker.is_file():
        return
    extract_archive(archive, extraction_dir)
    marker.write_text("ok", encoding="utf-8")


def _tdms_files(archive: Path, extraction_dir: Path) -> list[Path]:
    """TDMS file list; reuses the extraction when the cache marker exists."""
    marker = extraction_dir / ".extracted.ok"
    if marker.is_file():
        files = sorted(p for p in extraction_dir.rglob("*.tdms") if p.is_file())
    else:
        files = sorted(
            p for p in extract_archive(archive, extraction_dir)
            if p.is_file() and p.suffix.lower() == ".tdms"
        )
    if not files:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "archive contains no TDMS files")
    return files


def _classify_section(path: Path, processing: dict[str, Any]) -> str:
    stem = path.stem
    station_pattern = str(processing.get("section_pattern_stationary", "Stationary"))
    ramp_pattern = str(processing.get("section_pattern_ramp", "Ramp"))
    is_stationary = station_pattern and station_pattern in stem
    is_ramp = ramp_pattern and ramp_pattern in stem
    if is_stationary and is_ramp:
        raise DatasetError(
            DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
            f"{path.name} matches both stationary and ramp section patterns",
        )
    if is_stationary:
        return "stationary"
    if is_ramp:
        return "ramp"
    raise DatasetError(
        DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
        f"{path.name} matches no section pattern "
        f"(configure processing.section_pattern_stationary/section_pattern_ramp)",
    )


def _trajectories_from_trace(
    trace: np.ndarray,
    *,
    section: str,
    analysis_window: int | None,
    min_length: int,
    ramp_onset: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split one trace into (features, bifurcation_times) candidates.

    ``analysis_window=None`` keeps the whole trace as one trajectory
    (long traces stay single trajectories; chunking is opt-in via
    ``processing.analysis_window`` so the policy is always recorded in
    the manifest). Fixed windows are non-overlapping; a trailing chunk
    shorter than ``min_length`` fails loudly (never dropped silently,
    §5.4 gate 2).
    """
    n = int(trace.size)
    if n < min_length:
        raise DatasetError(
            DatasetErrorCode.PROCESS_SHORT_LENGTH,
            f"{section} trace has {n} samples (< min_length {min_length})",
        )
    chunks: list[tuple[int, int]] = []
    if analysis_window is None:
        chunks.append((0, n))
    else:
        if int(analysis_window) < min_length:
            raise ValueError(f"processing.analysis_window {analysis_window} < min_length {min_length}")
        for start in range(0, n, int(analysis_window)):
            end = min(start + int(analysis_window), n)
            if end - start < min_length:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_SHORT_LENGTH,
                    f"trailing chunk of {end - start} samples < min_length {min_length}; "
                    f"choose processing.analysis_window to divide the trace",
                )
            chunks.append((start, end))
    features: list[np.ndarray] = []
    bifs: list[np.ndarray] = []
    for start, end in chunks:
        chunk = trace[start:end]
        length = end - start
        if section == "stationary":
            bif = float(length + 1)
        else:
            onset = int(ramp_onset)
            if onset < 0 or onset >= length:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"ramp_onset_index {onset} outside chunk [0, {length})",
                )
            bif = float(onset)
        features.append(chunk.reshape(1, -1, 1))
        bifs.append(np.asarray([bif], dtype=np.float64))
    return features, bifs


def process(raw_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """§5.4 TAC processor handle: ``raw_dir -> uniform bundle``.

    Expected layout: ``raw/Experimental_time_traces_tdms.zip`` (verified
    by ingestion) containing one TDMS file per experiment section with
    ``StationaryX`` / ``RampY`` names (§4).
    """
    raw = Path(raw_dir)
    processing = dict(config.get("processing", {}) or {})
    min_length = int(processing.get("min_length", _MIN_LENGTH_DEFAULT))

    specs = config.get("expected_files", [])
    if not specs:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "tac config has no expected_files")
    archive = raw / str(specs[0]["path"])
    if not archive.exists():
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, f"missing raw archive: {archive}")
    extraction_dir = raw / "_extracted"
    _extract_once(archive, extraction_dir)

    records = inspect_tdms(archive, extraction_dir)
    tdms_files = sorted({Path(r["file"]) for r in records})
    if not tdms_files:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "archive contains no TDMS files")
    env_mode = str(processing.get("envelope", "raw"))
    if env_mode not in {"raw", "bandpass"}:
        raise ValueError(f"processing.envelope must be 'raw' or 'bandpass', got {env_mode!r}")
    fs = processing.get("sample_rate_hz")
    low = processing.get("band_low_hz")
    high = processing.get("band_high_hz")
    if env_mode == "bandpass":
        if fs is None or low is None or high is None:
            raise ValueError("bandpass envelope requires processing.sample_rate_hz, band_low_hz, band_high_hz")

    features: list[np.ndarray] = []
    bifs: list[np.ndarray] = []
    positives: list[bool] = []
    sections: dict[str, int] = {}
    selected: dict[str, dict[str, str]] = {}

    explicit_group = processing.get("channel_group")
    explicit_channel = processing.get("channel_name")
    if (explicit_group is None) != (explicit_channel is None):
        raise ValueError("processing.channel_group and channel_name must be set together")

    for tdms_path in tdms_files:
        if explicit_group is not None:
            if not any(r["group"] == explicit_group and r["channel"] == explicit_channel for r in records
                       if Path(r["file"]) == tdms_path):
                raise DatasetError(
                    DatasetErrorCode.INGEST_MANIFEST_MISMATCH,
                    f"{tdms_path.name} has no channel {explicit_group}/{explicit_channel}; "
                    f"inspect with inspect_tdms() to see the layout",
                )
            trace = extract_channel(tdms_path, group=explicit_group, channel=explicit_channel, min_length=1)
            group_name, channel_name = explicit_group, explicit_channel
        else:
            group_name, channel_name, trace = auto_select_channel(tdms_path, records)
        selected[str(tdms_path)] = {"group": group_name, "channel": channel_name}
        section = _classify_section(tdms_path, processing)
        if env_mode == "bandpass":
            trace = bandpass_envelope(trace, sample_rate_hz=float(fs), low_hz=float(low), high_hz=float(high))
        else:
            trace = hilbert_envelope(trace)
        feats, bifs_here = _trajectories_from_trace(
            trace,
            section=section,
            analysis_window=processing.get("analysis_window"),
            min_length=min_length,
            ramp_onset=processing.get("ramp_onset_index", 0),
        )
        features.extend(feats)
        bifs.extend(bifs_here)
        positives.extend([section == "ramp"] * len(feats))
        sections[section] = sections.get(section, 0) + 1

    if not features:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "no trajectories produced from TAC archive")
    if not any(positives) or all(positives):
        raise DatasetError(
            DatasetErrorCode.SPLIT_IMBALANCE,
            f"TAC sections must contain both stationary and ramp sections, got {sections}",
        )

    lengths = np.asarray([f.shape[1] for f in features], dtype=np.int64)
    features = np.concatenate(features, axis=0).astype(np.float32)
    is_positive = np.asarray(positives, dtype=bool)

    return {
        "features": features,
        "seq_lengths": lengths,
        "bifurcation_times": np.concatenate(bifs, axis=0),
        "is_positive": is_positive,
        # Provenance (recorded under manifest.processing.effective; NOT
        # part of processing.params so the staleness guard compares only
        # user-controlled keys).
        "meta": {"processing": {"sections": sections, "selected_channels": selected}},
    }


def validate_annotation(bundle: dict[str, Any]) -> None:
    """TAC annotation spot-check (§5.4 gate 3): structural consistency of
    the recorded transition policy (ramp onset inside every positive
    chunk; null sentinel convention)."""
    features = np.asarray(bundle["features"])
    bifs = np.asarray(bundle["bifurcation_times"], dtype=np.float64)
    lengths = np.asarray(bundle["seq_lengths"], dtype=np.int64)
    positive = np.asarray(bundle["is_positive"], dtype=bool)
    for i in range(len(features)):
        if positive[i]:
            if not 0.0 <= bifs[i] < lengths[i]:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"positive trajectory {i} has bifurcation_time {bifs[i]} outside [0, {lengths[i]})",
                )
        else:
            if bifs[i] != lengths[i] + 1:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"null trajectory {i} has non-sentinel bifurcation_time {bifs[i]}",
                )


__all__ = ["auto_select_channel", "bandpass_envelope", "extract_channel", "hilbert_envelope",
           "inspect_tdms", "process", "validate_annotation"]
