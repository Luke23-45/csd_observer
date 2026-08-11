"""Training callbacks that write through the run output contract."""
from __future__ import annotations

import time

import pytorch_lightning as pl

from csd_observer.outputs.metadata import collect_environment
from csd_observer.outputs.writer import OutputWriter


class LedgerCallback(pl.Callback):
    def __init__(self, writer: OutputWriter, *, method: str, seed: int) -> None:
        self.writer = writer
        self.method = method
        self.seed = int(seed)
        self.started = 0.0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.started = time.perf_counter()
        self.writer.open_log().write("training_started", method=self.method, seed=self.seed)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        elapsed = time.perf_counter() - self.started
        self.writer.write_timings({"method": self.method, "seed": self.seed, "fit_seconds": elapsed})
        self.writer.open_log().write("training_finished", method=self.method, seed=self.seed, seconds=elapsed)


class FingerprintCallback(pl.Callback):
    def __init__(self, writer: OutputWriter) -> None:
        self.writer = writer

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.writer.write_environment(collect_environment(extra={"stage": "fit"}))


__all__ = ["FingerprintCallback", "LedgerCallback"]
