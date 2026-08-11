"""Method-interface adapter for the recurrent alarm baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from csd_observer.models.common.interface import MethodMeta
from csd_observer.models.neural.lstm.model import LstmAlarmNet


class LstmAlarmMethod:
    """Causal LSTM scorer with deterministic, explicit inference semantics.

    Training is deliberately delegated to the training/orchestration layer;
    a supplied ``checkpoint`` is loaded during ``fit`` when present.  This
    prevents the model package from importing the training package.
    """

    def __init__(self, key: str, system: str) -> None:
        self.system = system
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg: dict[str, Any] = {}
        self._net: LstmAlarmNet | None = None
        self._meta = MethodMeta(
            name="LSTM-AlarmNet", family="neural", is_learned=True,
            scope_caveat="empirical baseline; evaluated under the same alarm governance",
            default_params={"in_channels": 1, "hidden_size": 64, "num_layers": 1, "dropout": 0.1},
            config_path=("lstm",),
        )

    @property
    def meta(self) -> MethodMeta:
        return self._meta

    def fit(self, train_arrays: dict[str, Any], val_arrays: dict[str, Any], cfg: dict[str, Any]) -> None:
        self._cfg = dict(cfg.get("model", {}).get("lstm", {}) or {})
        inferred = 1
        if "features" in train_arrays:
            shape = np.asarray(train_arrays["features"]).shape
            if len(shape) != 3:
                raise ValueError(f"features must have shape (B,T,C), got {shape}")
            inferred = int(shape[-1])
        self._cfg.setdefault("in_channels", inferred)
        self._net = LstmAlarmNet(**{k: self._cfg[k] for k in ("in_channels", "hidden_size", "num_layers", "dropout") if k in self._cfg})
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
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self._device)
        with torch.no_grad():
            return torch.sigmoid(self._net(x)).cpu().numpy().astype(np.float32)


def register_lstm_method() -> None:
    from csd_observer.models.common.registry import register_method
    register_method("LSTM-AlarmNet", LstmAlarmMethod, "neural")


__all__ = ["LstmAlarmMethod", "register_lstm_method"]
