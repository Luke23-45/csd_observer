"""DFA-CSD baseline: detrended-fluctuation-analysis exponent indicator.

See ``indicator.py`` for the definition and references (Peng et al.
1994; Kantelhardt et al. 2001; Livina & Lenton 2007; Dakos et al. 2012
— alpha in the 10–100-unit short-term regime; window >= 100 points per
plan §5).

The baseline has no trainable parameters (buffers only).
"""

from csd_observer.models.indicators.dfa_csd.indicator import raw_dfa_indicator

__all__ = ["raw_dfa_indicator"]
