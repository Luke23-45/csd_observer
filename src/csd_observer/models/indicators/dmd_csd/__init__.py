"""DMD-CSD baseline: dominant-eigenvalue dynamic-mode-decomposition indicator.

See ``indicator.py`` for the definition and references (Donovan &
Brand 2022, *Physica A* 596:127152 — sliding-window DMD, projected
operator ``U^T X' V S^{-1}``, EWS = ``|lambda_max|``; 2024
corrigendum corrects a sign error in the lung-ventilation *model*
only, not the method; Grziwotz et al. 2023 — dynamic eigenvalues;
time-delay embedding ``m = 6`` is our documented univariate
adaptation per plan A3).

The baseline has no trainable parameters (buffers only).
"""

from csd_observer.models.indicators.dmd_csd.indicator import raw_dmd_indicator

__all__ = ["raw_dmd_indicator"]
