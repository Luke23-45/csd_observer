"""VAR-CSD baseline: windowed-variance critical-slowing-down indicator.

See ``indicator.py`` for the definition and references (Scheffer et
al. 2009; Carpenter & Brock 2006; Dakos et al. 2012; ``earlywarnings``
``sd``; ewstools ``compute_var``).

The baseline has no trainable parameters (buffers only).
"""

from csd_observer.models.indicators.var_csd.indicator import raw_var_indicator

__all__ = ["raw_var_indicator"]
