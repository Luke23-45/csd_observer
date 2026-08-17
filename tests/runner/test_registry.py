"""Unit tests for the runner state registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from csd_observer.runner.registry import RegistryError, RunnerState


def test_runner_state_crud(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state = RunnerState(state_file)

    assert not state.is_complete("VAR-CSD", 0, "eval")
    assert state.get("VAR-CSD", 0, "eval") is None

    # Mark completed
    state.mark("VAR-CSD", 0, "eval", run_dir="outputs/fold/run1")
    assert state.is_complete("VAR-CSD", 0, "eval")
    entry = state.get("VAR-CSD", 0, "eval")
    assert entry is not None
    assert entry["status"] == "completed"
    assert entry["run_dir"] == "outputs/fold/run1"

    # Reload from disk
    reloaded = RunnerState(state_file)
    assert reloaded.is_complete("VAR-CSD", 0, "eval")

    # Mark failed
    reloaded.mark_failed("LSTM-AlarmNet", 0, "stage1", error="OOM")
    assert not reloaded.is_complete("LSTM-AlarmNet", 0, "stage1")
    failed_entry = reloaded.get("LSTM-AlarmNet", 0, "stage1")
    assert failed_entry is not None
    assert failed_entry["status"] == "failed"
    assert failed_entry["error"] == "OOM"


def test_runner_state_corrupt(tmp_path: Path) -> None:
    state_file = tmp_path / "corrupt.json"
    state_file.write_text("invalid json", encoding="utf-8")
    with pytest.raises(RegistryError):
        RunnerState(state_file)
