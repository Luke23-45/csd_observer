"""Model registry: the single source of truth for benchmark methods.

Supersedes ``benchmark/methods.py`` (plan A6.1: "Registry
``get_method/list_methods/validate_names`` supersedes
``benchmark/methods.py``"). The registry:

* maps display names to method factories (indicators, spectral-drift
  observer, neural baselines);
* builds a fresh method instance per ``(name, system)`` pair so every
  method carries its own fitted state (spectral grid-search result,
  neural weights);
* validates method names against the config before any run starts.

Methods are imported lazily so importing the registry never pulls in
torch (keeps indicator-only runs light): the seven indicators register
at module import (pure NumPy), while the spectral-drift observer and
the neural baselines register on the first call to
:func:`ensure_loaded` / any registry access.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from csd_observer.models.common.indicators import IndicatorMethod

# name -> (factory(key, system) -> MethodInterface instance, family)
_FACTORIES: dict[str, tuple[Callable[[str, str], Any], str]] = {
    "VAR-CSD": (lambda _key, system: IndicatorMethod("var_csd", system), "indicator"),
    "AC1-CSD": (lambda _key, system: IndicatorMethod("ac1_csd", system), "indicator"),
    "SKEW-CSD": (lambda _key, system: IndicatorMethod("skew_csd", system), "indicator"),
    "SRATIO-CSD": (lambda _key, system: IndicatorMethod("sratio_csd", system), "indicator"),
    "RETRATE-CSD": (lambda _key, system: IndicatorMethod("retrate_csd", system), "indicator"),
    "DFA-CSD": (lambda _key, system: IndicatorMethod("dfa_csd", system), "indicator"),
    "DMD-CSD": (lambda _key, system: IndicatorMethod("dmd_csd", system), "indicator"),
}

_REGISTERED: dict[str, tuple[str, Callable[[str, str], Any], str]] = {}
_LOADED = False


def _register(name: str, key: str, factory: Callable[[str, str], Any], family: str) -> None:
    if name in _REGISTERED:
        raise ValueError(f"Method {name!r} already registered")
    _REGISTERED[name] = (key, factory, family)


def register_method(
    name: str,
    factory: Callable[[str, str], Any],
    family: str,
) -> None:
    """Register a method factory (used by spectral/neural subpackages)."""
    if family not in ("indicator", "spectral", "neural"):
        raise ValueError(f"Unknown family: {family!r}")
    _register(name, name.lower().replace("-", "_"), factory, family)


def ensure_loaded() -> None:
    """Register every method, including torch-dependent ones.

    ``list_methods`` and ``validate_names`` need the full catalog; this
    is the entry point for those. Torch-import-free callers should use
    :func:`ensure_indicator_loaded` instead to avoid pulling in torch.
    """
    ensure_indicator_loaded()
    ensure_torch_loaded()


def ensure_indicator_loaded() -> None:
    """No-op for the seven pure-NumPy indicators; defined for symmetry."""
    # Indicators register at module import (top of this file). Nothing to
    # do here, but having the entry point makes the lazy-loading contract
    # explicit at the call sites (the orchestrator / tests).
    return


def ensure_torch_loaded() -> None:
    """Register torch-dependent methods exactly once (idempotent).

    Spectral-drift and the neural baselines self-register on import;
    importing them pulls in torch, so this happens lazily and only when
    a registry access needs the full catalog.
    """
    global _LOADED
    if _LOADED:
        return
    from csd_observer.models.neural.lstm.method import register_lstm_method  # noqa: E402
    from csd_observer.models.neural.patchtst.method import register_patchtst_method  # noqa: E402
    from csd_observer.models.neural.tcn.method import register_tcn_method  # noqa: E402
    from csd_observer.models.spectral_drift.method import register_spectral_method  # noqa: E402

    register_spectral_method()
    register_lstm_method()
    register_tcn_method()
    register_patchtst_method()
    _LOADED = True


def list_methods() -> list[str]:
    """All registered method display names, in registration order."""
    ensure_loaded()
    return list(_REGISTERED)


def list_families() -> dict[str, str]:
    ensure_loaded()
    return {name: family for name, (_key, _factory, family) in _REGISTERED.items()}


def validate_names(names: Sequence[str]) -> None:
    """Fail fast when any name is not registered (plan A6.1)."""
    ensure_loaded()
    unknown = [n for n in names if n not in _REGISTERED]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {sorted(unknown)}. "
            f"Valid methods: {', '.join(list_methods())}"
        )


def get_method(name: str, system: str = "fold") -> Any:
    """Build one method instance for ``name`` on ``system``.

    Indicator methods are built without importing torch. When the name
    is not among the seven indicators, the torch-bearing modules are
    lazily imported (once) and the lookup is retried — so
    ``get_method("Kalman-Spectral-Drift")`` works on the first call of
    a process, in any test/import order.
    """
    ensure_indicator_loaded()
    if name not in _REGISTERED:
        ensure_torch_loaded()
    if name not in _REGISTERED:
        raise ValueError(
            f"Unknown method: {name!r}. Valid methods: {', '.join(list_methods())}"
        )
    _key, factory, family = _REGISTERED[name]
    return factory(_key, system)


def build_methods(
    names: Sequence[str] | None = None,
    systems: Sequence[str] = ("fold", "hopf", "logistic"),
) -> dict[str, list[Any]]:
    """Build method instances for every (name, system) pair.

    Args:
        names: method names; ``None`` = all registered.
        systems: dataset bifurcation types to build for.

    Returns:
        ``{system: [method instances]}``.
    """
    if names is None:
        names = list_methods()
    validate_names(names)
    return {
        system: [get_method(name, system) for name in names]
        for system in systems
    }


# Register the seven indicators up front (their adapters are torch-free).
for _name, (_factory, _family) in _FACTORIES.items():
    _register(_name, _name.lower().replace("-", "_"), _factory, _family)

__all__ = [
    "build_methods",
    "ensure_indicator_loaded",
    "ensure_loaded",
    "ensure_torch_loaded",
    "get_method",
    "list_families",
    "list_methods",
    "register_method",
    "validate_names",
]
