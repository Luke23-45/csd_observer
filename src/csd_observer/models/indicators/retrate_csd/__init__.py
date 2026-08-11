"""RETRATE-CSD baseline: return-rate / recovery-time indicator.

See ``indicator.py`` for the definition and references (Ives 1995,
Ives et al. 2003; Dakos et al. 2012; ``earlywarnings`` ``returnrate =
1/ar1`` with the OLS ``ar.ols`` AR(1) estimator; score =
``-log(return_rate)`` per plan §3).

The baseline has no trainable parameters (buffers only).
"""

from csd_observer.models.indicators.retrate_csd.indicator import raw_retrate_indicator

__all__ = ["raw_retrate_indicator"]
