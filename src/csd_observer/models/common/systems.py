"""Canonical system (bifurcation-type) vocabulary shared by methods, config validation and tests.

Method-side acceptance set per plan §5.6 ("models/evaluation branch on
``meta``, never on dataset name"): the three synthetic systems plus the
two real-dataset systems verified in §4 (``subcritical_hopf`` for TAC,
``transcritical`` for DaphniaExt). Rows keep the exact ``meta.bif_type``
label unmodified; this module is only the method-side acceptance set and
the configuration cross-check target (validate_config rejects datasets
declaring a bif_type outside this vocabulary).

The synthetic generators in ``datasets/synthetic/common/generators.py``
keep their own narrower set (``fold``/``hopf``/``logistic``); this
vocabulary governs what the runner will attempt to construct a method
for.
"""

from __future__ import annotations

SUPPORTED_SYSTEMS: tuple[str, ...] = (
    "fold",
    "hopf",
    "logistic",
    "subcritical_hopf",
    "transcritical",
)

__all__ = ["SUPPORTED_SYSTEMS"]
