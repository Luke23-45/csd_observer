"""TAC TDMS inspection, envelope extraction and the §5.4 archive processor.

The processor turns the raw (already-extracted) experiment directory into
the uniform bundle the pipeline writes to ``datasets/processed/tac``:

1. locate the TDMS files under ``raw`` (``raw/Experimental_time_traces_tdms``
   or ``raw/*.tdms``; the legacy zip-archive layout is still supported for
   the synthetic fixtures);
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
5. ``bifurcation_times``: ramp sections carry the transition onset
   (``processing.ramp_onset_index`` when set explicitly, else auto-
   detected from the envelope as the first sustained amplitude growth —
   the rate-dependent transition delay of §4 is measured, not assumed).
   With chunking enabled each ramp trace yields ONE window aligned to its
   onset (``[onset - ramp_pre_samples, onset + ramp_post_samples)``) so
   the annotated ``tau`` sits inside ``[early_start_delta,
   length - early_end_delta)`` and the window carries pre-transition
   history for the early-warning protocol. Stationary sections =
   ``seq_length + 1`` sentinel (same null convention as the synthetic
   generators).
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
    return _inspect_files(files, TdmsFile)


def _inspect_files(files: list[Path], TdmsFile: Any) -> list[dict[str, Any]]:
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


def _locate_tdms(raw: Path, specs: list[dict[str, Any]]) -> list[Path]:
    """Locate the TDMS files for an already-extracted raw directory.

    ``raw`` holds the extracted experiment layout (e.g. ``raw/<stem>/*.tdms``
    or ``raw/*.tdms``). Raises :class:`DatasetError` when no readable TDMS
    file is present. (The zip-archive path lives in :func:`inspect_tdms` /
    ``_tdms_files`` and is retained for the synthetic fixtures.)
    """
    if not specs:
        raise DatasetError(DatasetErrorCode.INGEST_MANIFEST_MISMATCH, "tac config has no expected_files")
    root = raw
    archive_dir = raw / str(specs[0]["path"])
    if archive_dir.is_dir():
        root = archive_dir
    files = sorted(p for p in root.rglob("*.tdms") if p.is_file())
    if not files:
        raise DatasetError(
            DatasetErrorCode.INGEST_ARCHIVE,
            f"no TDMS files under {root}; expected the extracted "
            f"Experimental_time_traces_tdms layout",
        )
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


def _detect_ramp_onset(
    envelope: np.ndarray,
    *,
    detect_window: int,
    factor: float,
) -> int:
    """First sustained envelope growth beyond ``factor * baseline``.

    The baseline is the 10th percentile of the rolling-window envelope
    means — robust to the intermittency bursts that punctuate the quiet
    combustion-noise regime near onset (Bonciolini et al. 2018). Returns
    the sample index where the first window whose mean exceeds the
    threshold starts; returns 0 when no growth is found (the caller then
    fails loudly rather than fabricating an annotation).

    ``detect_window`` scales down for short traces (``max(2, n // 4)``)
    so the detector works on synthetic fixtures as well as the 600k-sample
    real recordings.
    """
    env = np.asarray(envelope, dtype=np.float64)
    n = int(env.size)
    if n < 2:
        return 0
    window = max(1, min(int(detect_window), max(2, n // 4)))
    if n <= window:
        return 0
    means = np.asarray(
        [float(env[i : i + window].mean()) for i in range(0, n - window + 1, window)],
        dtype=np.float64,
    )
    if means.size == 0:
        return 0
    baseline = float(np.percentile(means, 10))
    threshold = factor * max(baseline, 1e-12)
    for idx, mean in enumerate(means):
        if mean > threshold:
            return idx * window
    return 0


def _stationary_trajectories(
    trace: np.ndarray,
    *,
    analysis_window: int | None,
    min_length: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split a stationary trace into null trajectories.

    ``analysis_window=None`` keeps the whole trace as one trajectory;
    otherwise fixed non-overlapping windows (a trailing chunk shorter
    than ``min_length`` fails loudly — never dropped silently, §5.4).
    Every null gets the ``seq_length + 1`` sentinel bifurcation time.
    """
    n = int(trace.size)
    if analysis_window is None:
        chunks: list[tuple[int, int]] = [(0, n)]
    else:
        if int(analysis_window) < min_length:
            raise ValueError(f"processing.analysis_window {analysis_window} < min_length {min_length}")
        chunks = []
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
        features.append(chunk.reshape(1, -1, 1))
        bifs.append(np.asarray([float(length + 1)], dtype=np.float64))
    return features, bifs


def _ramp_trajectories(
    trace: np.ndarray,
    *,
    analysis_window: int | None,
    min_length: int,
    ramp_onset: int | None,
    ramp_detect_window: int,
    ramp_detect_factor: float,
    ramp_pre_samples: int,
    ramp_post_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Turn one ramp trace into positive trajectory/ies with real pre-onset
    history.

    The onset is ``ramp_onset`` when set explicitly, else auto-detected
    from the envelope (:func:`_detect_ramp_onset`). ``analysis_window=None``
    keeps the whole trace as the positive (``bifurcation_time = onset``);
    with chunking enabled the trace yields ONE window aligned to the onset,
    ``[onset - ramp_pre_samples, onset + ramp_post_samples)``, so the
    annotated ``tau = onset - start`` is comfortably inside the trajectory
    and the early-warning window of the evaluation protocol is fully
    populated. A ramp with no detectable onset (``onset == 0``) is an
    annotation error, never a silent degenerate positive.
    """
    n = int(trace.size)
    if ramp_onset is None:
        onset = _detect_ramp_onset(
            trace,
            detect_window=ramp_detect_window,
            factor=ramp_detect_factor,
        )
    else:
        onset = int(ramp_onset)
    if onset < 1 or onset >= n:
        raise DatasetError(
            DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
            f"ramp onset {onset} outside [1, {n}); auto-detection found no "
            f"sustained envelope growth — set processing.ramp_onset_index "
            f"explicitly or check the ramp section annotation",
        )
    if analysis_window is None:
        start, end, bif = 0, n, float(onset)
    else:
        start = max(0, onset - int(ramp_pre_samples))
        end = min(n, onset + int(ramp_post_samples))
        if end - start < min_length:
            raise DatasetError(
                DatasetErrorCode.PROCESS_SHORT_LENGTH,
                f"aligned ramp window of {end - start} samples < min_length "
                f"{min_length}; reduce processing.ramp_pre_samples/ramp_post_samples",
            )
        bif = float(onset - start)
    features: list[np.ndarray] = [trace[start:end].reshape(1, -1, 1)]
    bifs: list[np.ndarray] = [np.asarray([bif], dtype=np.float64)]
    return features, bifs


def _trajectories_from_trace(
    trace: np.ndarray,
    *,
    section: str,
    analysis_window: int | None,
    min_length: int,
    ramp_onset: int | None,
    ramp_detect_window: int,
    ramp_detect_factor: float,
    ramp_pre_samples: int,
    ramp_post_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split one trace into (features, bifurcation_times) candidates.

    Stationary traces become nulls (whole trace, or fixed ``analysis_window``
    chunks); ramp traces become positives carrying their transition onset
    (see :func:`_ramp_trajectories`). The chunking policy is always
    recorded in the manifest via the processing params.
    """
    n = int(trace.size)
    if n < min_length:
        raise DatasetError(
            DatasetErrorCode.PROCESS_SHORT_LENGTH,
            f"{section} trace has {n} samples (< min_length {min_length})",
        )
    if section == "stationary":
        return _stationary_trajectories(
            trace,
            analysis_window=analysis_window,
            min_length=min_length,
        )
    return _ramp_trajectories(
        trace,
        analysis_window=analysis_window,
        min_length=min_length,
        ramp_onset=ramp_onset,
        ramp_detect_window=ramp_detect_window,
        ramp_detect_factor=ramp_detect_factor,
        ramp_pre_samples=ramp_pre_samples,
        ramp_post_samples=ramp_post_samples,
    )


def process(raw_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """§5.4 TAC processor handle: ``raw_dir -> uniform bundle``.

    Expected layout: the already-extracted ``raw/Experimental_time_traces_tdms``
    directory (or ``raw/*.tdms``) containing one TDMS file per experiment
    section with ``StationaryX`` / ``RampY`` names (§4). The legacy
    zip-archive layout (``raw/Experimental_time_traces_tdms.zip``) is still
    supported for the synthetic fixtures via :func:`_tdms_files`.
    """
    raw = Path(raw_dir)
    processing = dict(config.get("processing", {}) or {})
    min_length = int(processing.get("min_length", _MIN_LENGTH_DEFAULT))

    specs = config.get("expected_files", [])
    tdms_files = _locate_tdms(raw, specs)

    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise DatasetError(DatasetErrorCode.INGEST_ARCHIVE, "npTDMS is required to inspect TAC data") from exc
    records = _inspect_files(tdms_files, TdmsFile)
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
            ramp_onset=processing.get("ramp_onset_index"),
            ramp_detect_window=int(processing.get("ramp_onset_detect_window", 5000)),
            ramp_detect_factor=float(processing.get("ramp_onset_detect_factor", 3.0)),
            ramp_pre_samples=int(processing.get("ramp_pre_samples", 200)),
            ramp_post_samples=int(processing.get("ramp_post_samples", 100)),
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
    max_len = int(lengths.max())
    padded = np.zeros((len(features), max_len, 1), dtype=np.float32)
    for i, feature in enumerate(features):
        padded[i, : lengths[i], 0] = feature[0, :, 0]
    is_positive = np.asarray(positives, dtype=bool)

    return {
        "features": padded,
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
    the recorded transition policy (ramp onset strictly inside every
    positive chunk — a positive with ``tau == 0`` has no pre-transition
    prefix and is un-evaluable under the protocol; null sentinel
    convention)."""
    features = np.asarray(bundle["features"])
    bifs = np.asarray(bundle["bifurcation_times"], dtype=np.float64)
    lengths = np.asarray(bundle["seq_lengths"], dtype=np.int64)
    positive = np.asarray(bundle["is_positive"], dtype=bool)
    for i in range(len(features)):
        if positive[i]:
            if not 0.0 < bifs[i] < lengths[i]:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"positive trajectory {i} has bifurcation_time {bifs[i]} outside (0, {lengths[i]})",
                )
        else:
            if bifs[i] != lengths[i] + 1:
                raise DatasetError(
                    DatasetErrorCode.PROCESS_ANNOTATION_MISSING,
                    f"null trajectory {i} has non-sentinel bifurcation_time {bifs[i]}",
                )


__all__ = ["auto_select_channel", "bandpass_envelope", "extract_channel", "hilbert_envelope",
           "inspect_tdms", "process", "validate_annotation"]
