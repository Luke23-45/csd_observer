"""L1.8/R4: output-tree contract — keyed timings, metrics summary,
protocol-checks CSV, environment provenance.

These tests exercise the writer and summarize layers directly (no full
run), so they stay fast and isolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from csd_observer.outputs.schema import ResultRow
from csd_observer.outputs.summarize import build_metrics_summary, summarize_run
from csd_observer.outputs.writer import OutputWriter


@pytest.fixture()
def writer(tmp_path: Path) -> OutputWriter:
    return OutputWriter("synthetic_fold", tmp_path, timestamp="2026-01-01_00-00-00-000000")


def _row(method: str, seed: int, *, ew_auc: float, fpr: float) -> ResultRow:
    return ResultRow(
        run_id="synthetic_fold-ts",
        timestamp="2026-01-01_00-00-00-000000",
        run_name="synthetic_fold",
        dataset="synthetic_fold",
        bif_type="fold",
        system="fold",
        replicate=f"s{seed}",
        method=method,
        family="indicator",
        is_learned=False,
        seed=seed,
        k_persist=5,
        fpr_target=0.05,
        detection_rate=0.8,
        detection_time_mean=10.0,
        detection_time_median=9.0,
        detection_time_std=2.0,
        ew_auc=ew_auc,
        fpr=fpr,
        persistent_fpr=fpr * 0.5,
        threshold=0.7,
        params={"window_size": 30},
        config_hash="abc",
        git_sha="def",
        achieved_fpr_target=fpr,
        extra={"persistence_ew_auc": ew_auc - 0.1},
    )


def test_timings_accumulate_by_method_seed_key(writer: OutputWriter) -> None:
    writer.write_timings({"method": "VAR-CSD", "seed": 0, "fit_seconds": 1.5})
    writer.write_timings({"method": "VAR-CSD", "seed": 0, "score_seconds": 0.2})
    writer.write_timings({"method": "LSTM-AlarmNet", "seed": 1, "fit_seconds": 9.0})
    payload = json.loads((writer.paths.times / "timings.json").read_text(encoding="utf-8"))
    assert set(payload) == {"entries"}
    assert set(payload["entries"]) == {"VAR-CSD__s0", "LSTM-AlarmNet__s1"}
    assert payload["entries"]["VAR-CSD__s0"]["fit_seconds"] == 1.5
    assert payload["entries"]["VAR-CSD__s0"]["score_seconds"] == 0.2
    assert payload["entries"]["LSTM-AlarmNet__s1"]["fit_seconds"] == 9.0


def test_metrics_summary_aggregates_across_seeds(writer: OutputWriter) -> None:
    summary = build_metrics_summary(
        [_row("VAR-CSD", 0, ew_auc=0.8, fpr=0.04), _row("VAR-CSD", 1, ew_auc=0.9, fpr=0.06)]
    )
    assert summary["n_seeds"] == 2
    assert summary["rows"] == 2
    entry = summary["summary"]["VAR-CSD"]
    assert entry["ew_auc"] == pytest.approx(0.85)
    assert entry["fpr"] == pytest.approx(0.05)
    assert entry["n_nan_ew_auc"] == 0


def test_summarize_run_writes_metrics_json_and_protocol_checks_csv(writer: OutputWriter) -> None:
    for seed in (0, 1):
        writer.write_result_row(_row("VAR-CSD", seed, ew_auc=0.8, fpr=0.04).to_dict())
    writer.write_protocol_checks({
        "method": "VAR-CSD", "system": "fold", "seed": 0,
        "k_persist": 5, "fpr_target": 0.05,
        "achieved_step_fpr": 0.04, "achieved_persistent_fpr": 0.02,
        "achieved_persistent_trajectory_fpr": 0.4,
        "null_anchor_upper_bound": 0.5,
        "trajectory_anchor_upper_bound": 0.8,
        "anchor_brackets_empirical": True,
        "drift_step_fpr": False, "drift_persistent_fpr": False, "drift": "ok",
    })

    paths = summarize_run(writer.root)
    assert paths["metrics"].is_file()
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert "summary" in metrics and metrics["rows"] == 2

    checks_csv = paths["protocol_checks"]
    assert checks_csv is not None and checks_csv.is_file()
    text = checks_csv.read_text(encoding="utf-8")
    assert "method,VAR-CSD" not in text  # method is a column, not adjacent to the value
    assert "VAR-CSD,fold,0,5,0.05,0.04,0.02,0.4,0.5,0.8,True,False,False,ok" in text
    assert "method,system,seed" in text
    assert "null_anchor_upper_bound" in text


def test_protocol_checks_use_renamed_anchor_keys(writer: OutputWriter) -> None:
    """R0.2: the anchor keys are ``null_anchor_upper_bound`` /
    ``trajectory_anchor_upper_bound`` (no ``_50step`` suffix)."""
    writer.write_protocol_checks({
        "method": "VAR-CSD", "system": "fold", "seed": 0,
        "k_persist": 5, "fpr_target": 0.05,
        "null_anchor_upper_bound": 0.5,
        "trajectory_anchor_upper_bound": 0.8,
        "drift": "ok",
    })
    payload = json.loads((writer.paths.metrics / "protocol_checks.json").read_text(encoding="utf-8"))
    assert "null_anchor_upper_bound" in payload[0]
    assert "trajectory_anchor_upper_bound" in payload[0]
    assert "null_anchor_upper_bound_50step" not in payload[0]
