"""Lightning wrapper for the PatchTST alarm baseline."""
from __future__ import annotations

from typing import Any

from csd_observer.models.neural.base.lit_base import BaseAlarmLitModule
from csd_observer.models.neural.patchtst.model import PatchTstAlarmNet


class PatchTstAlarmLitModule(BaseAlarmLitModule):
    def __init__(self, model_cfg: dict[str, Any], training_cfg: dict[str, Any], loss_fn: Any) -> None:
        super().__init__(model_cfg, training_cfg, loss_fn)
        self.net = PatchTstAlarmNet(
            in_channels=int(model_cfg.get("in_channels", 1)),
            patch_len=int(model_cfg.get("patch_len", 16)),
            stride=model_cfg.get("stride"),
            d_model=int(model_cfg.get("d_model", 64)),
            n_heads=int(model_cfg.get("n_heads", 2)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )


__all__ = ["PatchTstAlarmLitModule"]
