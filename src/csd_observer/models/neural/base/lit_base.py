"""Shared LightningModule for the neural alarm baselines.

Both baselines (LSTM, TCN) share the training/validation step, the
masked alarm loss application, the AdamW + cosine-schedule
configuration and the metric logging; only the network differs. The
loss is injected by the orchestration runner (plan A7:
``training/common/losses.py`` owns the alarm BCE; the one-way import
rule keeps ``models`` free of ``training`` imports).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from csd_observer.models.neural.base.labels import valid_mask_tensor

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]


class BaseAlarmLitModule(pl.LightningModule):
    """Shared Lightning training loop for the neural baselines.

    Subclasses must set ``self.net`` in their ``__init__``: an
    ``nn.Module`` mapping ``(B, T, C)`` features to ``(B, T)`` logits.
    """

    def __init__(
        self,
        model_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
        loss_fn: LossFn,
    ) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg or {})
        self.training_cfg = dict(training_cfg or {})
        self.loss_fn = loss_fn
        self.net: nn.Module = nn.Identity()  # replaced by subclasses
        self.save_hyperparameters(ignore=["loss_fn"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------ steps
    def _shared_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        feats, lens, targets = batch  # (B,T,C), (B,), (B,T)
        logits = self.net(feats)
        probs = torch.sigmoid(logits)
        mask = valid_mask_tensor(lens, targets.shape[-1])
        loss, logs = self.loss_fn(probs, targets, mask)
        return {"loss": loss, **logs}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out = self._shared_step(batch, batch_idx)
        self.log("train_loss", out["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_bce_raw", out.get("bce_raw", out["loss"]), on_epoch=True)
        return out["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        out = self._shared_step(batch, batch_idx)
        self.log("val_loss", out["loss"], prog_bar=True, on_epoch=True)
        self.log("val_bce_raw", out.get("bce_raw", out["loss"]), on_epoch=True)
        return out

    # ---------------------------------------------------- optimizer / lr
    def _lr(self) -> float:
        return float(
            self.model_cfg.get("lr", self.training_cfg.get("lr", 1e-3))
        )

    def configure_optimizers(self) -> Any:
        lr = self._lr()
        weight_decay = float(
            self.model_cfg.get("weight_decay", self.training_cfg.get("weight_decay", 1e-5))
        )
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        t_max = max(1, int(self.training_cfg.get("epochs", 50)))
        eta_min = float(self.training_cfg.get("scheduler_eta_min", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


__all__ = ["BaseAlarmLitModule"]
