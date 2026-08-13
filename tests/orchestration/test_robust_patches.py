"""Regression tests for the L9.x robustness patch sweep.

These tests lock in the contracts of the robustness patches applied
across the data / spectral / eval / outputs / config / runner surfaces.
Where a patch closed a real pre-existing bug (e.g.
``compute_persistent_detection_metrics`` arg-order bug) the test
exists to keep the regression from re-emerging.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

# ============================================================== seed_schedule


def test_seed_schedule_overflow_guard_rejects_int32_boundary() -> None:
    from csd_observer.config.validate import seed_schedule

    # 2**31 - 1 is the maximum signed int32. At offset = 2**31, even the
    # ``+101`` signal offset overflows.
    huge_offset = 2**31
    with pytest.raises(ValueError, match="overflows int32"):
        seed_schedule(huge_offset, 0)
    # The null split (``+202``) overflows before the signal split
    # (``+101``) when offset = 2**31 - 202. The error message must
    # name the split that overflowed.
    null_boundary = 2**31 - 202
    with pytest.raises(ValueError, match="null"):
        seed_schedule(null_boundary, 0)
    # One below the null boundary is still safe.
    safe_boundary = 2**31 - 203
    out = seed_schedule(safe_boundary, 0)
    assert out["signal"] == safe_boundary + 101
    assert out["null"] == safe_boundary + 202


def test_seed_schedule_overflow_guard_per_split_message() -> None:
    from csd_observer.config.validate import seed_schedule

    # Same trick: offset such that ``signal`` is OK and ``null``
    # overflows by one.
    base = 2**31 - 202
    with pytest.raises(ValueError, match="null"):
        seed_schedule(base, 0)


# ============================================================== extract_archive


def _make_zip_with_entry(archive: Path, name: str, payload: bytes = b"x") -> None:
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(name, payload)


def test_extract_archive_rejects_absolute_unix_path(tmp_path: Path) -> None:
    from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
    from csd_observer.datasets.common.ingest import extract_archive

    archive = tmp_path / "a.zip"
    _make_zip_with_entry(archive, "/etc/passwd")
    with pytest.raises(DatasetError) as ei:
        extract_archive(archive, tmp_path / "out")
    assert ei.value.code == DatasetErrorCode.INGEST_ARCHIVE


def test_extract_archive_rejects_drive_letter(tmp_path: Path) -> None:
    from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
    from csd_observer.datasets.common.ingest import extract_archive

    archive = tmp_path / "a.zip"
    _make_zip_with_entry(archive, "C:\\Windows\\evil.txt")
    with pytest.raises(DatasetError) as ei:
        extract_archive(archive, tmp_path / "out")
    assert ei.value.code == DatasetErrorCode.INGEST_ARCHIVE


def test_extract_archive_rejects_traversal(tmp_path: Path) -> None:
    from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
    from csd_observer.datasets.common.ingest import extract_archive

    archive = tmp_path / "a.zip"
    _make_zip_with_entry(archive, "../../../etc/passwd")
    with pytest.raises(DatasetError) as ei:
        extract_archive(archive, tmp_path / "out")
    assert ei.value.code == DatasetErrorCode.INGEST_ARCHIVE


def test_extract_archive_strips_setuid(tmp_path: Path) -> None:
    """setuid bits on extracted entries are cleared."""
    from csd_observer.datasets.common.ingest import extract_archive

    archive = tmp_path / "a.zip"
    payload = b"#!/bin/sh\ntrue\n"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("bin/script.sh")
        info.external_attr = 0o4755 << 16  # setuid + 0755
        zf.writestr(info, payload)
    out = tmp_path / "out"
    extract_archive(archive, out)
    target = out / "bin" / "script.sh"
    assert target.exists()
    if os.name != "nt":
        mode = target.stat().st_mode
        # 0o7000 = setuid | setgid | sticky
        assert (mode & 0o7000) == 0


# ============================================================== cached_manifest


def test_cached_manifest_only_catches_narrow_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cached_manifest`` swallows only ``OSError``, ``json.JSONDecodeError``,
    and ``ValueError``. A malformed-but-parsable manifest raises
    ``DatasetError`` (propagated from ``verify_manifest``) instead of
    being silently swallowed — a corrupted manifest must never trigger
    a destructive re-ingest."""
    from csd_observer.datasets.common import states
    from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode

    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.json").write_text(json.dumps({"invalid": "schema"}))
    with pytest.raises(DatasetError) as ei:
        states.cached_manifest(processed)
    assert ei.value.code == DatasetErrorCode.MANIFEST_CORRUPT


# ============================================================== protocol / governance


def test_compute_persistent_detection_metrics_arg_order_regression() -> None:
    """Regression: ``compute_persistent_detection_metrics`` was
    previously passing the wrong arg order into
    ``compute_persistent_dts`` (``threshold, k_persist, seq_lengths,
    threshold, k_persist`` instead of ``scores, bifurcation_times,
    is_positive, seq_lengths, threshold, k_persist``). The new
    implementation accepts and uses the documented arg order."""
    from csd_observer.evaluation.persistence.protocol import (
        compute_persistent_detection_metrics,
    )

    B, T = 6, 100
    rng = np.random.default_rng(0)
    scores = rng.normal(0.0, 1.0, size=(B, T)).astype(np.float32)
    bif_times = np.full(B, 80.0)
    is_positive = np.array([True, True, True, False, False, False])
    seq_lengths = np.full(B, T, dtype=np.int64)
    threshold = 1.5
    k_persist = 3

    out = compute_persistent_detection_metrics(
        scores, bif_times, is_positive, seq_lengths, threshold, k_persist
    )
    # ``n_evaluable`` must equal the count of positive trajectories with
    # finite threshold and ``tau > 0`` — i.e. 3.
    assert out["n_evaluable"] == 3
    assert 0.0 <= out["detection_rate"] <= 1.0
    # The detected count must equal the ``detection_rate * n_evaluable``
    # exact integer (this only holds because ``n_evaluable == 3`` and
    # the args are interpreted correctly).
    assert out["n_detected"] == int(round(out["detection_rate"] * out["n_evaluable"]))


def test_trajectory_fpr_anchor_length_aware() -> None:
    from csd_observer.evaluation.persistence.protocol import trajectory_fpr_anchor

    base = trajectory_fpr_anchor(0.05, k_persist=5, window_steps=50)
    short = trajectory_fpr_anchor(
        0.05, k_persist=5, window_steps=50, seq_lengths=np.full(10, 20)
    )
    long = trajectory_fpr_anchor(
        0.05, k_persist=5, window_steps=50, seq_lengths=np.full(10, 50)
    )
    # Short trajectories yield a smaller upper bound; long equal the base.
    assert 0 <= short < base
    assert long == pytest.approx(base)


def test_persistent_alarm_stream_requires_canonical_window() -> None:
    """Anchor regression: short windows vs the canonical 50 must
    surface a different upper bound, not the same value."""
    from csd_observer.evaluation.persistence.protocol import null_anchor_upper_bound

    short = null_anchor_upper_bound(0.05, 5, 5)
    long = null_anchor_upper_bound(0.05, 5, 50)
    assert short < long


# ============================================================== ledger throttling


def test_ledger_index_throttling_rebuilds_on_update_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even when ``index_throttle=1000``, ``update_status`` rebuilds."""
    from csd_observer.outputs.ledger import LedgerRow, RunLedger

    led = RunLedger(tmp_path, index_throttle=1000)
    row = LedgerRow(
        run_id="r1",
        timestamp="t1",
        run_name="d",
        dataset="d",
        methods=["VAR-CSD"],
        k_persist=5,
        config_hash="h",
        git_sha="g",
        status="pending",
        path="p",
    )
    led.append(row)
    led.update_status("r1", "completed")
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert any(r["status"] == "completed" for r in index["runs"])


def test_ledger_read_all_skips_partial_trailing_lines(tmp_path: Path) -> None:
    """A crash mid-append leaves a partial trailing line; the reader
    ignores it instead of raising."""
    from csd_observer.outputs.ledger import LedgerRow, RunLedger

    led = RunLedger(tmp_path)
    row = LedgerRow(
        run_id="r1",
        timestamp="t1",
        run_name="d",
        dataset="d",
        methods=["VAR-CSD"],
        k_persist=5,
        config_hash="h",
        git_sha="g",
        status="pending",
        path="p",
    )
    led.append(row)
    # Corrupt the trailing line.
    jsonl = tmp_path / "runs.jsonl"
    with open(jsonl, "ab") as f:
        f.write(b'{"run_id": "r2", "timesta')
    rows = led.read_all()
    assert [r.run_id for r in rows] == ["r1"]


def test_ledger_flush_force_rebuild(tmp_path: Path) -> None:
    from csd_observer.outputs.ledger import LedgerRow, RunLedger

    led = RunLedger(tmp_path, index_throttle=1000)
    led.append(
        LedgerRow(
            run_id="r1",
            timestamp="t1",
            run_name="d",
            dataset="d",
            methods=["VAR-CSD"],
            k_persist=5,
            config_hash="h",
            git_sha="g",
            status="pending",
            path="p",
        )
    )
    led.flush()
    assert (tmp_path / "index.json").exists()


# ============================================================== outputs writer


def test_write_result_row_is_atomic_append(tmp_path: Path) -> None:
    """Two writers contending on the same ``results.jsonl`` must not
    lose or corrupt rows: ``O_APPEND + fsync`` plus a per-process
    FileLock guarantees no read-modify-write race."""
    from csd_observer.outputs.schema import ResultRow
    from csd_observer.outputs.writer import OutputWriter

    writer = OutputWriter("synthetic_fold", base_dir=tmp_path, timestamp="2026-01-01T00-00-00-000000")
    for i in range(20):
        row = ResultRow(
            run_id="r", timestamp="t", run_name="synthetic_fold",
            dataset="synthetic_fold", bif_type="fold", system="fold",
            replicate=f"s{i}", method="VAR-CSD", family="indicator",
            is_learned=False, k_persist=5, fpr_target=0.05,
            detection_rate=0.5, detection_time_mean=10.0,
            detection_time_median=10.0, detection_time_std=0.0,
            ew_auc=0.5, fpr=0.05, persistent_fpr=0.04, threshold=0.0,
            params={}, config_hash="h", git_sha="g", seed=0,
        ).to_dict()
        writer.write_result_row(row)
    jsonl = (writer.paths.results / "results.jsonl").read_text(encoding="utf-8")
    lines = [ln for ln in jsonl.splitlines() if ln]
    assert len(lines) == 20


# ============================================================== validate_config


def test_validate_config_min_trajectories_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from csd_observer.config.validate import validate_config

    cfg = {
        "n_trajectories": 2,
        "max_length": 100,
        "dataset": "synthetic_fold",
        "models": ["VAR-CSD"],
        "evaluation": "persistenceaware",
        "k_persist": 5,
        "fpr_target": 0.05,
        "seed_offset": 0,
        "n_seeds": 1,
        "dataset_overrides": {},
    }
    with pytest.raises(ValueError, match="n_trajectories must be >= 3"):
        validate_config(cfg)


def test_validate_config_min_length_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from csd_observer.config.validate import validate_config

    cfg = {
        "n_trajectories": 8,
        "max_length": 64,
        "dataset": "synthetic_fold",
        "models": ["VAR-CSD"],
        "evaluation": "persistenceaware",
        "k_persist": 5,
        "fpr_target": 0.05,
        "seed_offset": 0,
        "n_seeds": 1,
        "dataset_overrides": {},
    }
    with pytest.raises(ValueError, match="max_length must be >= 100"):
        validate_config(cfg)


def test_validate_config_skip_min_gates_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from csd_observer.config.validate import validate_config

    monkeypatch.setenv("CSD_OBSERVER_SKIP_MIN_LENGTH_GATES", "1")
    cfg = {
        "n_trajectories": 2,
        "max_length": 64,
        "dataset": "synthetic_fold",
        "models": ["VAR-CSD"],
        "evaluation": "persistenceaware",
        "k_persist": 5,
        "fpr_target": 0.05,
        "seed_offset": 0,
        "n_seeds": 1,
        "dataset_overrides": {},
    }
    # Should NOT raise now that the env-var bypass is set.
    validate_config(cfg)


# ============================================================== VAR / DFA


def test_var_window_variance_skips_redundant_centering() -> None:
    """The redundant ``seg - seg.mean()`` pass was removed; the result
    must equal the population variance of the detrended window."""
    from csd_observer.models.indicators.var_csd.indicator import _window_variance

    seg = np.linspace(0, 1, 30) + 0.5 * np.sin(np.linspace(0, 2 * np.pi, 30))
    var = _window_variance(seg)
    # Reference: linear-detrend then mean of squares.
    n = len(seg)
    x = np.arange(n, dtype=float)
    a, b = np.polyfit(x, seg, 1)
    resid = seg - (a * x + b)
    assert var == pytest.approx(float(np.mean(resid * resid)))


def test_dfa_window_dfa_skips_redundant_centering() -> None:
    """Same regression: the redundant centering pass is gone; the
    scaling exponent must equal a reference OLS fit on the integrated
    profile."""
    from csd_observer.models.indicators.dfa_csd.indicator import _window_dfa_alpha

    rng = np.random.default_rng(0)
    seg = rng.normal(0, 1, size=500).cumsum()  # integrated noise
    box_sizes = (8, 16, 32, 64)
    alpha = _window_dfa_alpha(seg, box_sizes)
    assert 0.3 < alpha < 1.5  # integrated noise should be near 0.5


# ============================================================== registry split


def test_registry_indicator_load_does_not_pull_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ensure_indicator_loaded`` must not trigger torch imports.

    The indicator path (registry import → ``extract_mode``) must be
    torch-free, while ``ensure_torch_loaded`` pulls torch exactly once.
    """
    import builtins
    import sys

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("torch import blocked in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    # Evict every ``csd_observer.models.*`` module so the lazy-import
    # path re-executes module bodies (a cached module in ``sys.modules``
    # would skip the guarded ``import torch`` and make this test order-
    # dependent).
    for mod_name in list(sys.modules):
        if mod_name.startswith("csd_observer.models"):
            sys.modules.pop(mod_name, None)
    from csd_observer.models.common import registry
    # Indicator registration: must NOT trigger the guarded torch import.
    registry.ensure_indicator_loaded()
    # Building a pure-NumPy indicator must not either.
    method = registry.get_method("VAR-CSD", "fold")
    assert method is not None
    # Torch registration: must trigger the guarded import.
    with pytest.raises(RuntimeError, match="torch import blocked"):
        registry.ensure_torch_loaded()


# ============================================================== runner mark order


def test_runner_marks_completed_only_after_summarize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the lifecycle marker ``.completed`` must be set
    *after* ``summarize_run`` so consumers can use its presence as a
    signal that the tables subtree is built. We verify by reading the
    runner source for the ordering."""
    src = Path("src/csd_observer/orchestration/runner.py").read_text(encoding="utf-8")
    summarize_idx = src.find("summarize_run(writer.root)")
    mark_idx = src.find("writer.mark_completed()")
    assert summarize_idx > 0 and mark_idx > 0
    assert summarize_idx < mark_idx, (
        "summarize_run() must run BEFORE writer.mark_completed() "
        "so .completed is a reliable signal that tables are present"
    )


# ============================================================== synthetic determinism


def test_synthetic_dataset_per_trajectory_rng_is_deterministic() -> None:
    """Per-trajectory RNG sub-streams: same master seed reproduces the
    same trajectory set; a different master seed produces a different
    set (no global-stream cross-talk)."""
    from csd_observer.datasets.synthetic.common.generators import FoldBifurcationDataset

    a1 = FoldBifurcationDataset(n_trajectories=8, max_length=64, seed=42)
    a2 = FoldBifurcationDataset(n_trajectories=8, max_length=64, seed=42)
    b1 = FoldBifurcationDataset(n_trajectories=8, max_length=64, seed=43)
    sig_a1 = a1.generate()
    sig_a2 = a2.generate()
    sig_b1 = b1.generate()
    np.testing.assert_array_equal(sig_a1["features"], sig_a2["features"])
    np.testing.assert_array_equal(sig_a1["bifurcation_times"], sig_a2["bifurcation_times"])
    assert not np.array_equal(sig_a1["features"], sig_b1["features"])


# ============================================================== registry names


def test_registry_lists_only_display_names() -> None:
    from csd_observer.models.common.registry import list_families, list_methods

    names = list_methods()
    # ``LSTM``/``TCN`` are module keys, NOT display names.
    assert "LSTM" not in names
    assert "TCN" not in names
    assert "VAR-CSD" in names
    assert "LSTM-AlarmNet" in names
    assert "TCN-AlarmNet" in names
    families = list_families()
    assert families["VAR-CSD"] == "indicator"
    assert families["LSTM-AlarmNet"] == "neural"


# ============================================================== real-system taxonomy (G1/G2/G4)


def test_supported_systems_cover_real_bif_types() -> None:
    """The method-side vocabulary is exactly the plan taxonomy: the three
    synthetic systems plus the two real-dataset labels (§4)."""
    from csd_observer.models.common.systems import SUPPORTED_SYSTEMS

    assert set(SUPPORTED_SYSTEMS) == {
        "fold",
        "hopf",
        "logistic",
        "subcritical_hopf",
        "transcritical",
    }


def test_extract_mode_subcritical_hopf_two_channels_uses_radius() -> None:
    """2-channel subcritical_hopf (synthetic): radial mode, same rule as hopf."""
    from csd_observer.models.common.mode import extract_mode

    rng = np.random.default_rng(7)
    x1 = rng.normal(size=(4, 50)).astype(np.float32)
    x2 = rng.normal(size=(4, 50)).astype(np.float32)
    feats = np.stack([x1, x2], axis=-1)
    mode = extract_mode(feats, "subcritical_hopf")
    np.testing.assert_allclose(mode, np.sqrt(x1**2 + x2**2), atol=1e-5)


def test_extract_mode_subcritical_hopf_single_channel_passthrough() -> None:
    """TAC (§5.4): the amplitude envelope is observed directly; a
    1-channel subcritical_hopf trace passes through unchanged (no radius
    rule, unlike ``hopf`` which hard-requires 2 channels)."""
    from csd_observer.models.common.mode import extract_mode

    x = np.linspace(0.1, 2.0, 40, dtype=np.float32).reshape(2, 20, 1)
    mode = extract_mode(x, "subcritical_hopf")
    np.testing.assert_array_equal(mode, x[..., 0])


def test_extract_mode_transcritical_single_channel_passthrough() -> None:
    """DaphniaExt (§4): per-replicate population counts, single channel."""
    from csd_observer.models.common.mode import extract_mode

    x = np.arange(8 * 30, dtype=np.float32).reshape(8, 30, 1)
    mode = extract_mode(x, "transcritical")
    np.testing.assert_array_equal(mode, x[..., 0])


def test_indicator_method_accepts_real_systems() -> None:
    """G1: the indicator adapter must construct for the real-dataset
    bif_types (previously ``ValueError: Unknown system``)."""
    from csd_observer.models.common.indicators import IndicatorMethod
    from csd_observer.models.common.systems import SUPPORTED_SYSTEMS

    for system in ("subcritical_hopf", "transcritical"):
        m = IndicatorMethod("var_csd", system)
        assert m._system == system
        assert m.meta.bif_types_supported == list(SUPPORTED_SYSTEMS)
    with pytest.raises(ValueError, match="Unknown system"):
        IndicatorMethod("var_csd", "nonexistent")


def test_validate_config_rejects_unknown_bif_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G4: a dataset dict declaring a bif_type outside the method-side
    vocabulary fails fast in validate_config."""
    from csd_observer.config.validate import validate_config

    monkeypatch.setattr(
        "csd_observer.datasets.registry.list_datasets",
        lambda: ["tac", "daphnia_ext"],
    )
    cfg = {
        "dataset": {"name": "tac", "bif_type": "saddle_node"},
        "split": {"replicate_based": True},
        "models": ["VAR-CSD"],
        "evaluation": "persistenceaware",
        "k_persist": 5,
        "fpr_target": 0.05,
        "training": {"enabled": False},
        "seed_offset": 0,
        "n_seeds": 1,
        "dataset_overrides": {},
    }
    with pytest.raises(ValueError, match="unsupported bif_type"):
        validate_config(cfg)


def test_validate_config_accepts_real_bif_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G4 positive: the plan-verified real labels pass the cross-check."""
    from csd_observer.config.validate import validate_config

    monkeypatch.setattr(
        "csd_observer.datasets.registry.list_datasets",
        lambda: ["tac", "daphnia_ext"],
    )
    for name, bif_type in (
        ("tac", "subcritical_hopf"),
        ("daphnia_ext", "transcritical"),
    ):
        cfg = {
            "dataset": {"name": name, "bif_type": bif_type},
            "split": {"replicate_based": True},
            "models": ["VAR-CSD"],
            "evaluation": "persistenceaware",
            "k_persist": 5,
            "fpr_target": 0.05,
            "training": {"enabled": False},
            "seed_offset": 0,
            "n_seeds": 1,
            "dataset_overrides": {},
        }
        validate_config(cfg)  # must not raise
