"""MethodInterface wrapper for the spectral-drift observer (plan A6.3).

The physics in ``observer.py`` / ``preprocess.py`` is untouched; this
module only wraps it to the common interface so the governance driver
treats it like every other method:

* ``fit`` = validation grid search over ``(sigma_u, Q_drift)``
  (hyperparameter selection, no training) on the validation split;
* ``score`` = mode extraction + causal running-mean centring + the
  Rao-Blackwellised particle filter's collapse probability stream.

This replicates ``benchmark/experiments.py::evaluate_spectral`` exactly
(preprocessing, grids, observation-noise mapping per system, device
selection) so results are bit-comparable with the historical suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from csd_observer.models.common.interface import MethodMeta
from csd_observer.models.common.systems import SUPPORTED_SYSTEMS
from csd_observer.models.spectral_drift.grid_search import grid_search_sigma_u_q_drift
from csd_observer.models.spectral_drift.observer import SpectralDriftObserver
from csd_observer.models.spectral_drift.preprocess import extract_mode, running_mean_center

# Observation-noise mapping per system (used when the run config does not
# pin ``data.obs_noise_scale``). The real-data defaults mirror the
# synthetic generator scales (Hopf radial amplitude / fold-value
# observation noise); they are refined per dataset during Phase 8 (L8)
# once the archives are in the sandbox.
_OBS_NOISE_DEFAULT = {
    "fold": 0.10,
    "hopf": 0.15,
    "logistic": 0.05,
    "subcritical_hopf": 0.15,
    "transcritical": 0.10,
}

_DEFAULT_Q_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
_DEFAULT_SIGMA_U_GRID = [0.15, 0.3, 0.6, 1.0]


class SpectralDriftMethod:
    """Interface adapter for the Kalman-Spectral-Drift observer.

    Args:
        key: registry key (unused by this method; kept for a uniform
            factory signature).
        system: the dataset's bifurcation type; must be in
            ``systems.SUPPORTED_SYSTEMS`` (synthetic ``fold``/``hopf``/
            ``logistic`` plus real ``subcritical_hopf``/``transcritical``).
            Selects the mode-extraction rule and the observation-noise
            default.
    """

    def __init__(self, key: str, system: str) -> None:
        if system not in SUPPORTED_SYSTEMS:
            raise ValueError(f"Unknown system: {system!r}")
        self.key = key
        self._system = system
        self._fitted: dict[str, Any] | None = None
        self._observer: SpectralDriftObserver | None = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._meta = MethodMeta(
            name="Kalman-Spectral-Drift",
            family="spectral",
            is_learned=False,
            scope_caveat=(
                "theoretically grounded for fold (saddle-node) bifurcations; "
                "empirical elsewhere (Hopf, logistic, transcritical, TAC)"
            ),
            # Advisory (all systems are run; the caveat is the authority).
            bif_types_supported=list(SUPPORTED_SYSTEMS),
            default_params={
                "n_particles": 500,
                "c_min": 1e-3,
                "delta": 0.05,
                "center_window": 50,
                "q_drift_grid": list(_DEFAULT_Q_GRID),
                "sigma_u_grid": list(_DEFAULT_SIGMA_U_GRID),
            },
            config_path=("spectral_drift",),
        )

    @property
    def meta(self) -> MethodMeta:
        return self._meta

    def _spectral_cfg(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return cfg.get("model", {}).get("spectral_drift", {}) or {}

    def fit(
        self,
        train_arrays: dict[str, Any],
        val_arrays: dict[str, Any],
        cfg: dict[str, Any],
    ) -> None:
        """Validation grid search over ``(sigma_u, Q_drift)``.

        Args:
            train_arrays: signal bundle (all splits; the validation
                subset is taken from ``split_indices``).
            val_arrays: null bundle (validation subset used as the
                null anchor of the selection criterion).
            cfg: run config (``model.spectral_drift`` block). The
                orchestration runner stashes the per-seed run seed in
                ``cfg["__run_seed__"]``; we use it to seed the
                observer so per-seed reproducibility holds.
        """
        self._fit_seed = int(cfg.get("__run_seed__", 0) or 0)
        sd = self._spectral_cfg(cfg)
        n_particles = int(sd.get("n_particles", 500))
        c_min = float(sd.get("c_min", 1e-3))
        delta = float(sd.get("delta", 0.05))
        center_window = int(sd.get("center_window", 50))
        q_grid = [float(q) for q in sd.get("q_drift_grid", _DEFAULT_Q_GRID)]
        sigma_u_grid = [float(s) for s in sd.get("sigma_u_grid", _DEFAULT_SIGMA_U_GRID)]

        data_cfg = cfg.get("data", {})
        obs_noise_cfg = data_cfg.get("obs_noise_scale")
        obs_noise = (
            float(obs_noise_cfg)
            if obs_noise_cfg is not None
            else _OBS_NOISE_DEFAULT[self._system]
        )
        r_var = obs_noise ** 2

        lens_sig = train_arrays["seq_lengths"]
        lens_null = val_arrays["seq_lengths"]
        val_idx_s = np.asarray(train_arrays["split_indices"]["val"], dtype=np.int64)
        val_idx_n = np.asarray(val_arrays["split_indices"]["val"], dtype=np.int64)
        if val_idx_s.size == 0 or val_idx_n.size == 0:
            raise ValueError("fit() requires non-empty validation splits")

        mode_sig = running_mean_center(
            extract_mode(train_arrays["features"], self._system),
            lens_sig,
            center_window,
        )
        mode_null = running_mean_center(
            extract_mode(val_arrays["features"], self._system),
            lens_null,
            center_window,
        )

        best_sigma_u, best_q = grid_search_sigma_u_q_drift(
            mode_sig[val_idx_s],
            mode_null[val_idx_n],
            train_arrays["bifurcation_times"][val_idx_s],
            lens_sig[val_idx_s],
            lens_null[val_idx_n],
            sigma_u_grid=sigma_u_grid,
            r=r_var,
            q_grid=q_grid,
            n_particles=n_particles,
            c_min=c_min,
            delta=delta,
            device=self._device,
        )
        self._fitted = {
            "sigma_u": best_sigma_u,
            "best_q": best_q,
            "q_drift": best_q,
            "r": r_var,
            "n_particles": n_particles,
            "c_min": c_min,
            "delta": delta,
            "center_window": center_window,
        }
        self._observer = None  # rebuilt lazily on the fitted device
        self._fit_seed: int | None = None

    def _observer_seed(self) -> int:
        """Return a stable observer RNG seed derived from the run seed.

        ``torch.seed``/``numpy``/Python are seeded in
        :func:`trainer_factory.seed_everything` for learned methods; the
        observer must still produce *different* output across ``n_seeds``
        so per-seed reproducibility holds. We hash ``run_seed`` into a
        32-bit integer and store it in :attr:`_fit_seed` during ``fit``;
        ``score`` falls back to ``0`` if the hash is unavailable (e.g.
        when the method is used outside a governed run).
        """
        return int(self._fit_seed) if self._fit_seed is not None else 0

    def score(
        self,
        features: np.ndarray,
        seq_lengths: np.ndarray,
        cfg: dict[str, Any],
    ) -> np.ndarray:
        """Collapse-probability alarm scores ``(B, T)`` float32."""
        if self._fitted is None:
            raise RuntimeError(
                "SpectralDriftMethod.score() called before fit(); the "
                "governance driver always calls fit() first."
            )
        f = self._fitted
        if self._observer is None:
            self._observer = (
                SpectralDriftObserver(
                    sigma_u=f["sigma_u"],
                    r=f["r"],
                    q_drift=f["q_drift"],
                    n_particles=f["n_particles"],
                    c_min=f["c_min"],
                    delta=f["delta"],
                    seed=self._observer_seed(),
                )
                .to(self._device)
                .eval()
            )
        mode = running_mean_center(
            extract_mode(features, self._system),
            np.asarray(seq_lengths, dtype=np.int64),
            f["center_window"],
        )
        x = torch.from_numpy(np.asarray(mode, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            probs = self._observer(x)["collapse_prob"].cpu().numpy()
        return np.asarray(probs, dtype=np.float32)

    def fitted_params(self) -> dict[str, Any]:
        if self._fitted is None:
            return {}
        return dict(self._fitted)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SpectralDriftMethod(system={self._system!r})"


def register_spectral_method() -> None:
    """Register ``Kalman-Spectral-Drift`` in the model registry."""
    from csd_observer.models.common.registry import register_method

    register_method("Kalman-Spectral-Drift", SpectralDriftMethod, "spectral")


__all__ = ["SpectralDriftMethod", "register_spectral_method"]
