"""Method-interface adapter for the PatchTST alarm baseline.

Same contract as the LSTM/TCN adapters: training is delegated to the
training/orchestration layer, ``fit`` loads the supplied checkpoint.
``in_channels`` is deliberately absent from ``default_params``: the
schema default ``None`` means "auto" (the runner pins the real channel
count from the dataset bundle), so a hard-coded 1 would be stale.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from csd_observer.models.common.interface import MethodMeta
from csd_observer.models.neural.base.scoring import score_batched
from csd_observer.models.neural.patchtst.model import PatchTstAlarmNet


class PatchTstAlarmMethod:
    """Causal patch-transformer scorer with deterministic inference."""

    def __init__(self, key: str, system: str) -> None:
        self.system = system
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg: dict[str, Any] = {}
        self._net: PatchTstAlarmNet | None = None
        self._meta = MethodMeta(
            name="PatchTST-AlarmNet", family="neural", is_learned=True,
            scope_caveat="empirical baseline; evaluated under the same alarm governance",
            default_params={
                "patch_len": 16, "stride": 16, "d_model": 64, "n_heads": 2,
                "n_layers": 2, "mlp_ratio": 4.0, "dropout": 0.1,
            },
            config_path=("patchtst",),
        )

    @property
    def meta(self) -> MethodMeta:
        return self._meta

    def fit(self, train_arrays: dict[str, Any], val_arrays: dict[str, Any], cfg: dict[str, Any]) -> None:
        self._cfg = dict(cfg.get("model", {}).get("patchtst", {}) or {})
        inferred = 1
        if "features" in train_arrays:
            shape = np.asarray(train_arrays["features"]).shape
            if len(shape) != 3:
                raise ValueError(f"features must have shape (B,T,C), got {shape}")
            inferred = int(shape[-1])
        # ``None`` means "auto": pin the channel count to the data.
        if not self._cfg.get("in_channels"):
            self._cfg["in_channels"] = inferred
        self._net = PatchTstAlarmNet(
            in_channels=int(self._cfg["in_channels"]),
            patch_len=int(self._cfg.get("patch_len", 16)),
            stride=self._cfg.get("stride"),
            d_model=int(self._cfg.get("d_model", 64)),
            n_heads=int(self._cfg.get("n_heads", 2)),
            n_layers=int(self._cfg.get("n_layers", 2)),
            mlp_ratio=float(self._cfg.get("mlp_ratio", 4.0)),
            dropout=float(self._cfg.get("dropout", 0.1)),
        )
        self._net.to(self._device).eval()
        checkpoint = self._cfg.get("checkpoint")
        if checkpoint:
            try:
                state = torch.load(checkpoint, map_location=self._device, weights_only=True)
            except TypeError:  # torch versions before the weights_only keyword
                state = torch.load(checkpoint, map_location=self._device)
            sd = state.get("state_dict", state)
            # Lightning saves the net under a "net." prefix; strip it so
            # strict=False can never silently drop every weight.
            if isinstance(sd, dict) and all(str(k).startswith("net.") for k in sd):
                sd = {str(k)[4:]: v for k, v in sd.items()}
            self._net.load_state_dict(sd, strict=False)

    def score(self, features: Any, seq_lengths: Any, cfg: dict[str, Any]) -> np.ndarray:
        if self._net is None:
            self.fit({}, {}, cfg)
        assert self._net is not None
        block = dict(cfg.get("model", {}).get("patchtst", {}) or {})
        batch_size = int(block.get("score_batch_size", 64))
        return score_batched(
            self._net,
            np.asarray(features, dtype=np.float32),
            np.asarray(seq_lengths, dtype=np.int64),
            batch_size=batch_size,
            device=self._device,
        )


def register_patchtst_method() -> None:
    from csd_observer.models.common.registry import register_method
    from csd_observer.models.neural.patchtst.lit_module import PatchTstAlarmLitModule

    register_method("PatchTST-AlarmNet", PatchTstAlarmMethod, "neural", lit_module=PatchTstAlarmLitModule)


__all__ = ["PatchTstAlarmMethod", "register_patchtst_method"]
