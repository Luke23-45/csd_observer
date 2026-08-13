"""Deterministic Lightning trainer construction."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from csd_observer.models.neural.base.data import build_dataloaders
from csd_observer.outputs.writer import OutputWriter
from csd_observer.training.common.callbacks import FingerprintCallback, LedgerCallback


def _determinism_env_override() -> bool:
    return os.environ.get("CSD_OBSERVER_ALLOW_NONDETERMINISM", "").lower() in {"1", "true", "yes"}


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG and pin deterministic algorithms (R2.6).

    ``torch.use_deterministic_algorithms(True, warn_only=False)`` is
    the strict contract: a non-deterministic kernel is a hard error, not
    a warning, so a "reproducible" run can never silently drift. The
    escape hatch ``CSD_OBSERVER_ALLOW_NONDETERMINISM=1`` (documented in
    AGENTS.md) restores the lenient ``warn_only=True`` for machines
    whose kernels lack deterministic implementations.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    if deterministic:
        warn_only = _determinism_env_override()
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.benchmark = False


def fit_model(module: pl.LightningModule, train_bundles: list[dict[str, Any]],
              val_bundles: list[dict[str, Any]], *, writer: OutputWriter,
              method: str, seed: int, config: dict[str, Any]) -> pl.Trainer:
    training = dict(config.get("training", {}) or {})
    seed_everything(seed, deterministic=bool(training.get("deterministic", True)))
    train_loader, val_loader = build_dataloaders(
        train_bundles, val_bundles, batch_size=int(training.get("batch_size", 32)),
        label_window=int(training.get("label_window", 60)),
        num_workers=int(training.get("num_workers", 0)), seed=seed,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(writer.paths.artifacts / "checkpoints"),
        filename=f"{method}-seed{seed}-{{epoch:03d}}-{{val_loss:.5f}}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    callbacks = [checkpoint, EarlyStopping(monitor="val_loss", mode="min", patience=int(training.get("patience", 10))),
                 LedgerCallback(writer, method=method, seed=seed), FingerprintCallback(writer)]
    trainer = pl.Trainer(
        default_root_dir=str(writer.root), max_epochs=int(training.get("epochs", 50)),
        accelerator=training.get("accelerator", "auto"), devices=training.get("devices", "auto"),
        logger=False, enable_progress_bar=bool(training.get("progress_bar", False)),
        deterministic=bool(training.get("deterministic", True)), callbacks=callbacks,
    )
    # Lightning's deterministic handling re-applies
    # ``torch.use_deterministic_algorithms(True, warn_only=True)`` at
    # fit time, silently downgrading the strict contract; re-assert it
    # after the Trainer is constructed (R2.6).
    if bool(training.get("deterministic", True)) and not _determinism_env_override():
        torch.use_deterministic_algorithms(True, warn_only=False)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer


__all__ = ["fit_model", "seed_everything"]
