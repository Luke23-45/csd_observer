"""Small causal temporal-convolution alarm baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from csd_observer.models.common.interface import MethodMeta
from csd_observer.models.neural.tcn.model import TcnAlarmNet


class TcnAlarmMethod:
    def __init__(self, key: str, system: str) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: TcnAlarmNet | None = None
        self._meta = MethodMeta(
            name="TCN-AlarmNet", family="neural", is_learned=True,
            scope_caveat="empirical baseline; evaluated under the same alarm governance",
            default_params={"in_channels": 1, "hidden_size": 32, "kernel_size": 5},
            config_path=("tcn",),
        )

    @property
    def meta(self) -> MethodMeta:
        return self._meta

    def fit(self, train_arrays: dict[str, Any], val_arrays: dict[str, Any], cfg: dict[str, Any]) -> None:
        c = dict(cfg.get("model", {}).get("tcn", {}) or {})
        in_channels = int(c.get("in_channels") or 1)
        if "features" in train_arrays:
            shape = np.asarray(train_arrays["features"]).shape
            if len(shape) != 3:
                raise ValueError(f"features must have shape (B,T,C), got {shape}")
            # ``None`` means "auto": pin the channel count to the data.
            if not c.get("in_channels"):
                in_channels = int(shape[-1])
        hidden = int(c.get("hidden_size", 32))
        kernel = int(c.get("kernel_size", 5))
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("tcn.kernel_size must be a positive odd integer")
        levels = int(c.get("levels", 3))
        self._net = TcnAlarmNet(in_channels, hidden, kernel, levels).to(self._device).eval()
        checkpoint = c.get("checkpoint")
        if checkpoint:
            try:
                state = torch.load(checkpoint, map_location=self._device, weights_only=True)
            except TypeError:
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
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self._device)
        with torch.no_grad():
            return torch.sigmoid(self._net(x)).cpu().numpy().astype(np.float32)


def register_tcn_method() -> None:
    from csd_observer.models.common.registry import register_method
    register_method("TCN-AlarmNet", TcnAlarmMethod, "neural")


__all__ = ["TcnAlarmMethod", "register_tcn_method"]
