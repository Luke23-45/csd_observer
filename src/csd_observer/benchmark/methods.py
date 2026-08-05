"""Benchmark method catalog.

Single source of truth for the eight benchmark rows
(``docs/plan/benchmark_revision_plan.md`` §2): the proposed
``Kalman-Spectral-Drift`` observer plus the seven published CSD
indicators. The catalog drives method selection (``methods=`` CLI),
config resolution, the results tables, and the registry-vs-config
cross-check that fails fast when the code and ``configs/model`` drift
apart.

Every row consumes the same preprocessed input — the scalar dominant
mode from ``extract_mode`` (plan §2: "preprocessing is identical across
every row") — and is evaluated under the identical fixed-FPR governance
(``csd_observer.utils.evaluation``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from csd_observer.models.ac1_csd import raw_ac1_indicator
from csd_observer.models.dfa_csd import raw_dfa_indicator
from csd_observer.models.dmd_csd import raw_dmd_indicator
from csd_observer.models.retrate_csd import raw_retrate_indicator
from csd_observer.models.skew_csd import raw_skew_indicator
from csd_observer.models.sratio_csd import raw_sratio_indicator
from csd_observer.models.var_csd import raw_var_indicator

_INDICATOR_BLOCK = "csd_indicators"


@dataclass(frozen=True)
class MethodSpec:
    """Static description of one benchmark method.

    Attributes:
        name: display name used in ``METHODS``, ``methods=`` filtering,
            results rows and file slugs.
        family: ``"indicator"`` for the shared indicator driver,
            ``"spectral"`` for the spectral-drift driver.
        scorer: the indicator callable (``(features, seq_lengths, **params)
            -> (B, T)`` float32); unused for the spectral family.
        config_path: nested key path into ``config["model"]`` that
            carries this method's parameters.
        preprocess: ``"mode"`` for mode-only input (indicators),
            ``"mode_center"`` for mode + running-mean centring
            (spectral observer).
        default_params: hard-coded fallbacks, identical to the config
            block values (plan §6: "defaults hard-coded too, so nothing
            breaks without it").
    """

    name: str
    family: str
    scorer: Optional[Callable] = None
    config_path: Tuple[str, ...] = ()
    preprocess: str = "mode"
    default_params: Dict[str, object] = field(default_factory=dict)


METHODS: Tuple[MethodSpec, ...] = (
    MethodSpec(
        name="Kalman-Spectral-Drift",
        family="spectral",
        config_path=("spectral_drift",),
        preprocess="mode_center",
    ),
    MethodSpec(
        name="VAR-CSD",
        family="indicator",
        scorer=raw_var_indicator,
        config_path=(_INDICATOR_BLOCK, "var"),
        default_params={"window_size": 30},
    ),
    MethodSpec(
        name="AC1-CSD",
        family="indicator",
        scorer=raw_ac1_indicator,
        config_path=(_INDICATOR_BLOCK, "ac1"),
        default_params={"window_size": 30},
    ),
    MethodSpec(
        name="SKEW-CSD",
        family="indicator",
        scorer=raw_skew_indicator,
        config_path=(_INDICATOR_BLOCK, "skew"),
        default_params={"window_size": 30},
    ),
    MethodSpec(
        name="SRATIO-CSD",
        family="indicator",
        scorer=raw_sratio_indicator,
        config_path=(_INDICATOR_BLOCK, "sratio"),
        default_params={"window_size": 30},
    ),
    MethodSpec(
        name="DFA-CSD",
        family="indicator",
        scorer=raw_dfa_indicator,
        config_path=(_INDICATOR_BLOCK, "dfa"),
        default_params={"window_size": 100},
    ),
    MethodSpec(
        name="RETRATE-CSD",
        family="indicator",
        scorer=raw_retrate_indicator,
        config_path=(_INDICATOR_BLOCK, "retrate"),
        default_params={"window_size": 30},
    ),
    MethodSpec(
        name="DMD-CSD",
        family="indicator",
        scorer=raw_dmd_indicator,
        config_path=(_INDICATOR_BLOCK, "dmd"),
        default_params={"window_size": 30, "embedding_dim": 6, "rank": 2},
    ),
)

METHOD_NAMES: Tuple[str, ...] = tuple(spec.name for spec in METHODS)


def get_method(name: str) -> MethodSpec:
    for spec in METHODS:
        if spec.name == name:
            return spec
    raise ValueError(
        f"Unknown method: {name!r}. Valid methods: {', '.join(METHOD_NAMES)}"
    )


def resolve_params(spec: MethodSpec, model_cfg: dict) -> Dict[str, object]:
    """Resolve a method's parameters: config block values override the
    hard-coded defaults; ``fpr_target`` is a shared block-level setting
    and is never a method parameter."""
    params: Dict[str, object] = dict(spec.default_params)
    node: dict = model_cfg
    for key in spec.config_path:
        if key not in node:
            raise KeyError(
                f"Config model.{'.'.join(spec.config_path)} is missing for method "
                f"{spec.name!r}"
            )
        node = node[key]
    if not isinstance(node, dict):
        raise TypeError(
            f"Config model.{'.'.join(spec.config_path)} must be a mapping for "
            f"method {spec.name!r}"
        )
    for key, value in node.items():
        if key == "fpr_target":
            continue
        params[key] = value
    return params


def validate_catalog(model_cfg: dict) -> None:
    """Cross-check the catalog against the resolved model config.

    Fails fast if a registered method lost its config block or the
    indicator block carries an unknown method — the same drift guard as
    the config loader, but at method granularity.
    """
    for spec in METHODS:
        try:
            resolve_params(spec, model_cfg)
        except KeyError as exc:
            raise KeyError(
                f"Method {spec.name!r} is registered but missing from the "
                f"model config: {exc}"
            ) from exc
    block = model_cfg.get(_INDICATOR_BLOCK, {})
    if not isinstance(block, dict):
        raise TypeError(
            f"Config model.{_INDICATOR_BLOCK} must be a mapping, "
            f"got {type(block).__name__}"
        )
    known = {spec.config_path[-1] for spec in METHODS if spec.family == "indicator"}
    unknown = set(block) - known - {"fpr_target"}
    if unknown:
        raise ValueError(
            f"Config model.{_INDICATOR_BLOCK} has methods not in the catalog: "
            f"{sorted(unknown)}"
        )


def indicator_fpr_target(model_cfg: dict) -> float:
    block = model_cfg.get(_INDICATOR_BLOCK, {})
    return float(block.get("fpr_target", 0.05))


__all__ = [
    "METHODS",
    "METHOD_NAMES",
    "MethodSpec",
    "get_method",
    "indicator_fpr_target",
    "resolve_params",
    "validate_catalog",
]
