"""Unit tests for the runner protocol loader and plan builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from csd_observer.runner.protocol import (
    Method,
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    load_protocol,
)


def _sample_raw_protocol() -> dict:
    return {
        "name": "test_protocol",
        "task": "SyntheticFold",
        "description": "Test protocol description",
        "seeds": [0, 1],
        "defaults": ["+test_override=1"],
        "methods": [
            {
                "index": 1,
                "name": "VAR-CSD",
                "role": "indicator",
                "model": "default",
                "data": "synthetic_fold",
                "stages": [],
                "evaluate": True,
                "evaluate_mode": "persistenceaware",
            },
            {
                "index": 2,
                "name": "LSTM-AlarmNet",
                "role": "neural",
                "model": "lstm",
                "data": "synthetic_fold",
                "stages": [1],
                "evaluate": True,
                "evaluate_mode": "persistenceaware",
            },
        ],
    }


def test_load_protocol_success(tmp_path: Path) -> None:
    p_path = tmp_path / "protocol.json"
    p_path.write_text(json.dumps(_sample_raw_protocol()), encoding="utf-8")
    proto = load_protocol(p_path)

    assert proto.name == "test_protocol"
    assert proto.seeds == (0, 1)
    assert len(proto.methods) == 2
    assert proto.methods[0].name == "VAR-CSD"
    assert proto.methods[1].name == "LSTM-AlarmNet"
    assert proto.methods[1].stages == (1,)


def test_load_protocol_validation_errors(tmp_path: Path) -> None:
    # Invalid JSON
    p_path = tmp_path / "bad.json"
    p_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_protocol(p_path)

    # Missing required name
    p_path.write_text(json.dumps({"task": "fold", "seeds": [0], "methods": []}), encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_protocol(p_path)

    # Duplicate index
    raw = _sample_raw_protocol()
    raw["methods"][1]["index"] = 1
    p_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_protocol(p_path)


def test_select_methods() -> None:
    m1 = Method(1, "VAR-CSD", "role", "default", "synthetic_fold", (), None, True)
    m2 = Method(2, "LSTM-AlarmNet", "role", "lstm", "synthetic_fold", (1,), None, True)
    proto = Protocol("proto", "fold", "desc", (0, 1), (), (m1, m2))

    selected = proto.select_methods(["1"])
    assert selected == [m1]

    selected = proto.select_methods(["LSTM-AlarmNet"])
    assert selected == [m2]

    with pytest.raises(ProtocolError):
        proto.select_methods(["unknown"])


def test_build_plan() -> None:
    m1 = Method(1, "VAR-CSD", "role", "default", "synthetic_fold", (), None, True)
    m2 = Method(2, "LSTM-AlarmNet", "role", "lstm", "synthetic_fold", (1,), None, True)
    proto = Protocol("proto", "fold", "desc", (0, 1), (), (m1, m2))

    # Full plan
    plan = build_plan(proto, [m1, m2])
    # m1 has eval for seed 0 and seed 1 (2 steps)
    # m2 has stage1 + eval for seed 0 and seed 1 (4 steps) -> total 6 steps
    assert len(plan) == 6
    assert plan[0] == Step("eval", m1, seed=0)
    assert plan[1] == Step("eval", m1, seed=1)
    assert plan[2] == Step("train", m2, seed=0, stage=1)
    assert plan[3] == Step("eval", m2, seed=0)
    assert plan[4] == Step("train", m2, seed=1, stage=1)
    assert plan[5] == Step("eval", m2, seed=1)

    # Eval only
    plan_eval = build_plan(proto, [m1, m2], eval_only=True)
    assert len(plan_eval) == 4
    assert all(s.kind == "eval" for s in plan_eval)

    # Stage 1 only
    plan_stage1 = build_plan(proto, [m1, m2], stage=1)
    assert len(plan_stage1) == 2
    assert all(s.kind == "train" and s.stage == 1 for s in plan_stage1)
