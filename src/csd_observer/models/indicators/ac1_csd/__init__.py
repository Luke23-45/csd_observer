"""AC1-CSD baseline: lag-1 autocorrelation critical-slowing-down indicator.

See ``indicator.py`` for the definition and references (Dakos et al.
2012; Bury et al. 2021; ``earlywarnings`` ``acf1``).

The baseline has no trainable parameters (buffers only).
"""

from csd_observer.models.indicators.ac1_csd.indicator import raw_ac1_indicator

__all__ = ["raw_ac1_indicator"]
