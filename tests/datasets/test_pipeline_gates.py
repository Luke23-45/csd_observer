"""R0.4/R0.6: common-gate min_length policy (dataset common pipeline).

``_run_gates`` must honor the dataset's configured ``processing.min_length``
(default 100 = DFA gate): a bundle containing a shorter trajectory is
rejected with the exclude-with-counts error, never silently dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from csd_observer.datasets.common.errors import DatasetError, DatasetErrorCode
from csd_observer.datasets.common.pipeline import _run_gates


def _bundle(short_length: int = 50) -> dict:
    """4 trajectories: one of ``short_length`` steps, three of 150."""
    lengths = np.array([short_length, 150, 150, 150], dtype=np.int64)
    features = np.zeros((4, 150, 1), dtype=np.float32)
    return {
        "features": features,
        "seq_lengths": lengths,
        "bifurcation_times": np.array([140.0, 140.0, 140.0, 140.0]),
        "is_positive": np.array([True, True, False, False], dtype=bool),
    }


def test_min_length_gate_rejects_short_trajectory() -> None:
    with pytest.raises(DatasetError) as exc:
        _run_gates(_bundle(short_length=50), None, min_length=200)
    assert exc.value.code == DatasetErrorCode.PROCESS_SHORT_LENGTH
    assert "shorter than 200 steps" in str(exc.value)


def test_min_length_gate_accepts_at_default() -> None:
    # default min_length=100: the 50-step trajectory is rejected too
    with pytest.raises(DatasetError) as exc:
        _run_gates(_bundle(short_length=50), None)
    assert exc.value.code == DatasetErrorCode.PROCESS_SHORT_LENGTH
    # a bundle where every trajectory meets 100 steps passes the gate
    _run_gates(_bundle(short_length=150), None)


def test_min_length_gate_rejects_when_configured_lower() -> None:
    # processing.min_length can be lowered for real datasets; the gate
    # must accept what the dataset declares.
    _run_gates(_bundle(short_length=50), None, min_length=50)
