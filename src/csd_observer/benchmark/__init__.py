"""Benchmark suite: proposed method + published CSD indicator baselines.

Public surface: the method catalog (``METHODS``), the per-configuration
suite runner (``run_config``) and the CLI (``python -m
csd_observer.benchmark``). See ``csd_observer.benchmark.methods`` for
the catalog and ``csd_observer.benchmark.suite`` for the governance
pipeline.
"""

from csd_observer.benchmark.methods import METHOD_NAMES, METHODS, MethodSpec
from csd_observer.benchmark.suite import SystemResult, run_config

__all__ = [
    "METHODS",
    "METHOD_NAMES",
    "MethodSpec",
    "SystemResult",
    "run_config",
]
