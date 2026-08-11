"""SKEW-CSD baseline: absolute-skewness critical-slowing-down indicator.

See ``indicator.py`` for the definition and references (Dakos et al.
2012; ``earlywarnings`` ``sk`` = ``abs(moments::skewness)``; Guttal &
Jayaprakash 2008).

The baseline has no trainable parameters (buffers only).
"""

from csd_observer.models.indicators.skew_csd.indicator import raw_skew_indicator

__all__ = ["raw_skew_indicator"]
