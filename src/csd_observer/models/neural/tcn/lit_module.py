"""Lightning wrapper for the TCN alarm baseline."""
from __future__ import annotations

from typing import Any

from csd_observer.models.neural.base.lit_base import BaseAlarmLitModule
from csd_observer.models.neural.tcn.model import TcnAlarmNet


class TcnAlarmLitModule(BaseAlarmLitModule):
    def __init__(self, model_cfg: dict[str, Any], training_cfg: dict[str, Any], loss_fn: Any) -> None:
        super().__init__(model_cfg, training_cfg, loss_fn)
        self.net = TcnAlarmNet(
            in_channels=int(model_cfg.get("in_channels", 1)),
            hidden_size=int(model_cfg.get("hidden_size", 32)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            levels=int(model_cfg.get("levels", 3)),
        )


__all__ = ["TcnAlarmLitModule"]
