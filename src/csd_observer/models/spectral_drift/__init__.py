"""Spectral-drift Bayesian early warning observer (fold-type).

Implements the formal definition in ``docs/z2/formal_defininition.md``:

* the scalar OU reduction of the dominant mode, ``du = -c(t) u dt + sigma_u dW``;
* the exact discrete-time transition
  ``u_{k+1} = exp(-c_k dt) u_k + eps_k`` with
  ``Var(eps_k) = sigma_u^2 / (2 c_k) (1 - exp(-2 c_k dt))``;
* a positive state process ``c_{k+1} = max(c_min, c_k + eta_k)`` with
  ``eta_k ~ N(0, Q_drift)`` (c_min > 0 keeps the transition variance
  well defined);
* a Rao-Blackwellised particle filter: particles over the spectral gap
  ``c_k``, exact conditional scalar Kalman filter over the mode ``u_k``;
* a Shiryaev-type alarm: the collapse probability
  ``p_collapse,k = Pr(c_k < delta | y_{1:k})``.

The reduction is theoretically grounded for fold (saddle-node) bifurcations
only. On Hopf and period-doubling (logistic) systems the observer is applied
empirically; results there should carry that scope caveat.

The observer has no trainable parameters (buffers only).
"""

from csd_observer.models.spectral_drift.grid_search import (
    grid_search_q_drift,
    grid_search_sigma_u_q_drift,
)
from csd_observer.models.spectral_drift.observer import SpectralDriftObserver
from csd_observer.models.spectral_drift.preprocess import (
    extract_mode,
    running_mean_center,
)

__all__ = [
    "SpectralDriftObserver",
    "extract_mode",
    "grid_search_q_drift",
    "grid_search_sigma_u_q_drift",
    "running_mean_center",
]
