"""Neural-baseline smoke on a multi-channel dataset (regression L7.x).

synthetic_hopf emits a 2-channel bundle (radius + frequency); the
learned baselines must build their input projection for the real
channel count instead of the hard-coded 1 (runner bug fixed at G3):
training and scoring run end-to-end and produce schema-valid rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from csd_observer.config.store import register_configs
from csd_observer.config.validate import validate_config
from csd_observer.outputs.ledger import RunLedger
from csd_observer.outputs.schema import validate_row

_REPO = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = os.path.relpath(_REPO / "configs", Path(__file__).resolve().parent)


@pytest.fixture(params=["LSTM-AlarmNet,TCN-AlarmNet", "PatchTST-AlarmNet"])
def hopf_config(tmp_path: Path, request: pytest.FixtureRequest) -> dict:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    register_configs()
    with initialize(version_base=None, config_path=_CONFIG_PATH):
        cfg = compose(
            config_name="run",
            overrides=[
                "dataset=synthetic_hopf",
                f"models=[{request.param}]",
                "training.epochs=1",
                "training.patience=2",
                "+dataset_overrides.n_trajectories=16",
                "+dataset_overrides.max_length=128",
            ],
        )
    config = OmegaConf.to_container(cfg, resolve=True)
    config["output"]["base_dir"] = str(tmp_path)
    validate_config(config)
    return config


def test_neural_multichannel_smoke(hopf_config: dict, tmp_path: Path) -> None:
    from csd_observer.orchestration.runner import run_benchmark

    writer = run_benchmark(hopf_config)

    marker = writer.root.with_suffix(writer.root.suffix + ".completed")
    assert marker.is_file(), f"missing .completed marker at {marker}"

    rows = [
        json.loads(line)
        for line in (writer.root / "results" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_methods = hopf_config["models"]
    assert {r["method"] for r in rows} == set(expected_methods)
    for row in rows:
        validate_row(row)
        assert row["bif_type"] == "hopf"
        # Regression: the input projection must be built for the real
        # channel count (2 for synthetic_hopf), not hard-coded 1.
        assert row["params"]["in_channels"] == 2

    ledger = RunLedger(tmp_path / "_ledger")
    entry = ledger.read_all()[-1]
    assert entry.status == "completed"
    assert entry.is_learned_methods == expected_methods


def test_patchtst_net_causal_and_piecewise() -> None:
    """PatchTST emission policy: no future leak, neutral warm-up.

    A score at step t comes only from patches completed at or before t:
    (1) steps before the first completion are the neutral logit 0;
    (2) the signal is piecewise-constant between completions; (3) values
    inside an uncompleted patch must not change earlier emissions.
    """
    import torch

    from csd_observer.models.neural.patchtst.model import PatchTstAlarmNet

    net = PatchTstAlarmNet(
        in_channels=1, patch_len=8, stride=8, d_model=16,
        n_heads=2, n_layers=1, dropout=0.0,
    )
    net.eval()
    x = torch.randn(2, 32, 1)
    with torch.no_grad():
        logits = net(x)
    assert logits.shape == (2, 32)
    # Neutral logit while no patch (window 0..7, ends at step 7) is complete.
    assert torch.all(logits[:, :7] == 0.0)
    # Piecewise-constant: block [7,15) <- patch 0, [15,23) <- patch 1, ...
    for start in range(7, 32, 8):
        block = logits[:, start : start + 8]
        assert torch.allclose(block, block[:, :1].expand_as(block))

    # Perturbing x inside patch 2 (indices 16..23) changes only the
    # emissions of patch 2 (steps 23..31); earlier steps must be bit-identical.
    x2 = x.clone()
    x2[:, 17:20] += 0.5
    with torch.no_grad():
        logits2 = net(x2)
    assert torch.equal(logits[:, :23], logits2[:, :23])
    assert not torch.equal(logits[:, 23:], logits2[:, 23:])
