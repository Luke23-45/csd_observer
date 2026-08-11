"""MethodInterface adapter for the seven statistical CSD indicators.

One shared adapter class serves all seven indicator implementations
(plan A6.1: "a single helper removes repetition and the risk of
per-method drift"). Each indicator's raw function is pure NumPy and
parameter-free (deterministic); the adapter supplies the common
preprocessing (scalar dominant mode extraction) and parameter
resolution (config block values override hard-coded defaults).

The adapter is constructed per (indicator, system) pair by the model
registry so the ``score`` pipeline matches the legacy benchmark exactly:
``extract_mode(features, system)`` then the raw indicator on the
single-channel mode (``(B, T, 1)``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from csd_observer.models.common.interface import MethodMeta
from csd_observer.models.spectral_drift.preprocess import extract_mode

INDICATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "var_csd": {"window_size": 30},
    "ac1_csd": {"window_size": 30},
    "skew_csd": {"window_size": 30},
    "sratio_csd": {"window_size": 30},
    "retrate_csd": {"window_size": 30},
    "dfa_csd": {"window_size": 100},
    "dmd_csd": {"window_size": 30, "embedding_dim": 6, "rank": 2},
}

# display name, raw function import path (lazy), config block key
_INDICATOR_NAMES: dict[str, tuple[str, str, str]] = {
    "var_csd": ("VAR-CSD", "csd_observer.models.indicators.var_csd", "raw_var_indicator"),
    "ac1_csd": ("AC1-CSD", "csd_observer.models.indicators.ac1_csd", "raw_ac1_indicator"),
    "skew_csd": ("SKEW-CSD", "csd_observer.models.indicators.skew_csd", "raw_skew_indicator"),
    "sratio_csd": ("SRATIO-CSD", "csd_observer.models.indicators.sratio_csd", "raw_sratio_indicator"),
    "retrate_csd": ("RETRATE-CSD", "csd_observer.models.indicators.retrate_csd", "raw_retrate_indicator"),
    "dfa_csd": ("DFA-CSD", "csd_observer.models.indicators.dfa_csd", "raw_dfa_indicator"),
    "dmd_csd": ("DMD-CSD", "csd_observer.models.indicators.dmd_csd", "raw_dmd_indicator"),
}

_SCOPE_CAVEAT = (
    "statistical CSD indicator; theoretically grounded for fold-type "
    "slowing down, empirical elsewhere (known inversions on Hopf for "
    "AC1/SRATIO/RETRATE/DFA, plan A3 caveats)"
)


def _lazy_raw_fn(module_path: str, fn_name: str) -> Callable:
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


class IndicatorMethod:
    """MethodInterface adapter for one statistical CSD indicator.

    Attributes:
        key: registry key (``"var_csd"``, ``"ac1_csd"``, ...).
        system: one of ``"fold"``, ``"hopf"``, ``"logistic"`` (the
            dataset's bifurcation type, used for mode extraction).
    """

    def __init__(self, key: str, system: str) -> None:
        if key not in _INDICATOR_NAMES:
            raise KeyError(
                f"Unknown indicator key {key!r}; valid: {sorted(_INDICATOR_NAMES)}"
            )
        if system not in ("fold", "hopf", "logistic"):
            raise ValueError(f"Unknown system: {system!r}")
        self.key = key
        self._system = system
        display, module_path, fn_name = _INDICATOR_NAMES[key]
        self._raw_fn = _lazy_raw_fn(module_path, fn_name)
        self._meta = MethodMeta(
            name=display,
            family="indicator",
            is_learned=False,
            scope_caveat=_SCOPE_CAVEAT,
            bif_types_supported=["fold"],
            default_params=dict(INDICATOR_DEFAULTS[key]),
            config_path=(key,),
        )

    @property
    def meta(self) -> MethodMeta:
        return self._meta

    def fit(
        self,
        train_arrays: dict[str, Any],
        val_arrays: dict[str, Any],
        cfg: dict[str, Any],
    ) -> None:
        """No-op: statistical indicators are deterministic (buffers only)."""

    def score(
        self,
        features: np.ndarray,
        seq_lengths: np.ndarray,
        cfg: dict[str, Any],
    ) -> np.ndarray:
        """Alarm scores: mode extraction then the raw indicator.

        Args:
            features: ``(B, T, C)`` observed features.
            seq_lengths: ``(B,)`` valid prefix lengths.
            cfg: full run config; the method's block under
                ``cfg["model"][self.key]`` overrides the defaults.

        Returns:
            ``(B, T)`` float32 alarm scores (``NaN`` where undefined).
        """
        mode = extract_mode(features, self._system)[:, :, None]
        params = _resolve_params(self.meta, cfg)
        return np.asarray(
            self._raw_fn(mode, seq_lengths, **params), dtype=np.float32
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"IndicatorMethod(key={self.key!r}, system={self._system!r})"


def _resolve_params(meta: MethodMeta, cfg: dict[str, Any]) -> dict[str, Any]:
    """Config block values override hard-coded defaults; ``fpr_target``
    is an evaluation-level setting, never a method parameter."""
    params: dict[str, Any] = dict(meta.default_params)
    node: Any = cfg.get("model", {})
    for key in meta.config_path:
        if key not in node:
            return params
        node = node[key]
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "fpr_target" or v is None:
                continue
            params[k] = v
    return params


__all__ = ["IndicatorMethod", "_resolve_params"]
