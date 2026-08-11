"""SRATIO-CSD baseline: spectral-density-ratio critical-slowing-down indicator.

See ``indicator.py`` for the definition and references (Dakos et al.
2012; ``earlywarnings`` ``densratio`` = ``spec.ar`` at the lowest
non-zero bin over the Nyquist bin; Scheffer et al. 2009 spectral
reddening).

The baseline has no trainable parameters (buffers only).
"""

from csd_observer.models.indicators.sratio_csd.indicator import raw_sratio_indicator

__all__ = ["raw_sratio_indicator"]
