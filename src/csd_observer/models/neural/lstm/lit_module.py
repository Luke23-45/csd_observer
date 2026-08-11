"""Lightning training module for the LSTM alarm baseline.

The loss function is *injected* (plan A6.4 + A7): ``models`` must not
import ``training``, so the orchestration runner wires the canonical
alarm loss (``training/common/losses.py``) into the module via the
``loss_fn`` argument. The module itself only owns the network, the
optimizer/scheduler schedule and the metric logging.
"""

from __future__ import annotations

from typing import Any

from csd_observer.models.neural.base.lit_base import BaseAlarmLitModule
from csd_observer.models.neural.lstm.model import LstmAlarmNet


class LstmAlarmLitModule(BaseAlarmLitModule):
    """LightningModule wrapping :class:`LstmAlarmNet`.

    Args:
        model_cfg: ``cfg["model"]["lstm"]`` block (architecture and
            optimizer knobs).
        training_cfg: ``cfg["training"]`` block (epochs, scheduler
            floor).
        loss_fn: injected alarm-loss callable
            ``(probs, targets, mask) -> (loss, logs_dict)``.
    """

    def __init__(
        self,
        model_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
        loss_fn: Any,
    ) -> None:
        super().__init__(model_cfg, training_cfg, loss_fn)
        self.net = LstmAlarmNet(
            in_channels=int(model_cfg.get("in_channels", 1)),
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )


__all__ = ["LstmAlarmLitModule"]
