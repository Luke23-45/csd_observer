"""L5.4: training determinism — identical seed ⇒ identical artifacts.

The strict-determinism contract (R2.6) guarantees that two fits with
the same run seed produce the same loss curve and a byte-identical
best checkpoint, so a reproducibility check on disk is exact.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from csd_observer.models.neural.lstm.lit_module import LstmAlarmLitModule
from csd_observer.outputs.writer import OutputWriter
from csd_observer.training.common.losses import alarm_bce
from csd_observer.training.common.trainer_factory import fit_model, seed_everything

pytestmark = pytest.mark.heavy


def _bundles() -> list[dict]:
    rng = np.random.default_rng(2)
    n, length = 16, 32
    split = {
        "train": np.arange(0, 10),
        "val": np.arange(10, 13),
        "test": np.arange(13, n),
    }
    bundle = {
        "features": rng.standard_normal((n, length, 1)).astype(np.float32),
        "seq_lengths": np.full(n, length, dtype=np.int64),
        "bifurcation_times": np.full(n, 24.0, dtype=np.float64),
        "is_positive": np.ones(n, dtype=bool),
        "split_indices": split,
    }
    return [
        {k: v[:10] for k, v in bundle.items() if k != "split_indices"} | {"split_indices": {"train": np.arange(10)}},
        {k: v[10:13] for k, v in bundle.items() if k != "split_indices"} | {"split_indices": {"val": np.arange(3)}},
    ]


def _fit_once(tmp_path: Path, seed: int, *, k_persist: int = 5) -> Path:
    writer = OutputWriter("det", tmp_path, timestamp=f"ts-{seed}-k{k_persist}")
    config = {"training": {"deterministic": True, "epochs": 1, "batch_size": 8,
                           "label_window": 8, "patience": 2, "progress_bar": False}}
    # Production order (train_method): seed first, then construct the
    # module so weight init is deterministic (R2.6).
    seed_everything(seed, deterministic=True)
    module = LstmAlarmLitModule(
        {"in_channels": 1, "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
        config["training"],
        partial(alarm_bce, k_persist=k_persist),
    )
    train_bundle, val_bundle = _bundles()
    trainer = fit_model(
        module, [train_bundle], [val_bundle], writer=writer, method="LSTM-AlarmNet",
        seed=seed, config=config,
    )
    ckpt = trainer.checkpoint_callback.best_model_path
    assert ckpt, "no checkpoint produced"
    return Path(ckpt)


def _state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load the checkpoint's weights (the zip embeds run-dir paths, so
    raw bytes are not a fair comparison across different tmp dirs)."""
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]


def test_same_seed_reproduces_byte_identical_checkpoint(tmp_path: Path) -> None:
    a = _state_dict(_fit_once(tmp_path / "a", seed=101))
    b = _state_dict(_fit_once(tmp_path / "b", seed=101))
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), (
            f"same-seed fits must produce identical weights ({key}) (R2.6)"
        )


def test_different_seed_differs(tmp_path: Path) -> None:
    a = _state_dict(_fit_once(tmp_path / "a", seed=101))
    b = _state_dict(_fit_once(tmp_path / "b", seed=202))
    assert not all(torch.equal(a[k], b[k]) for k in a)


def test_k_persist_changes_loss_for_same_seed(tmp_path: Path) -> None:
    """R0.3/R0.6: the persistence knob is folded into the loss at
    training time — same seed, different k_persist ⇒ different curves."""
    a = _fit_once(tmp_path / "k5", seed=101, k_persist=5)
    b = _fit_once(tmp_path / "k1", seed=101, k_persist=1)
    # the checkpoint filename embeds the monitored val_loss
    va = float(a.stem.rsplit("val_loss=", 1)[1])
    vb = float(b.stem.rsplit("val_loss=", 1)[1])
    assert va != vb
    sa = _state_dict(a)
    sb = _state_dict(b)
    assert not all(torch.equal(sa[k], sb[k]) for k in sa)
