"""Validation EW-AUC grid search for the spectral-drift observer hyperparameters."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from csd_observer.models.spectral_drift.observer import SpectralDriftObserver


def grid_search_q_drift(
    y_val_signal: np.ndarray,
    y_val_null: np.ndarray,
    bifs_val: np.ndarray,
    lens_signal_val: np.ndarray,
    lens_null_val: np.ndarray,
    *,
    sigma_u: float,
    r: float,
    q_grid: Optional[List[float]] = None,
    n_particles: int = 500,
    c_min: float = 1e-3,
    delta: float = 0.05,
    device: Optional[torch.device] = None,
) -> float:
    """Select ``Q_drift`` by validation early-warning AUC.

    Mirrors ``grid_search_q`` in ``csd_observer.models.kalman_lag2``: run
    the observer for each candidate ``q`` on the validation split and keep
    the one with the highest ``compute_early_warning_auc`` over the collapse
    probabilities.

    Args:
        y_val_signal: ``(B_sig, T)`` centred mode sequences (signal).
        y_val_null: ``(B_null, T)`` centred mode sequences (null).
        bifs_val: ``(B_sig,)`` bifurcation times of the signal validation set.
        lens_signal_val: ``(B_sig,)`` signal sequence lengths.
        lens_null_val: ``(B_null,)`` null sequence lengths.
        sigma_u: process noise scale of the mode.
        r: observation noise variance.
        q_grid: candidate drift-noise values (default: six orders of
            magnitude from ``1e-6`` up to ``1e-1``).
        n_particles, c_min, delta: observer hyperparameters.
        device: torch device.

    Returns:
        The ``q`` value achieving the highest validation EW-AUC.
    """
    _, best_q = grid_search_sigma_u_q_drift(
        y_val_signal,
        y_val_null,
        bifs_val,
        lens_signal_val,
        lens_null_val,
        sigma_u_grid=[sigma_u],
        r=r,
        q_grid=q_grid,
        n_particles=n_particles,
        c_min=c_min,
        delta=delta,
        device=device,
    )
    return best_q


def grid_search_sigma_u_q_drift(
    y_val_signal: np.ndarray,
    y_val_null: np.ndarray,
    bifs_val: np.ndarray,
    lens_signal_val: np.ndarray,
    lens_null_val: np.ndarray,
    *,
    sigma_u_grid: Optional[List[float]] = None,
    r: float,
    q_grid: Optional[List[float]] = None,
    n_particles: int = 500,
    c_min: float = 1e-3,
    delta: float = 0.05,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Select ``(sigma_u, Q_drift)`` jointly by validation early-warning AUC.

    The model's per-step noise scale ``sigma_u`` is a free hyperparameter
    (diagnostics: ``docs/notes/spectral_drift_diagnosis.md``): fixing it to
    ``noise_scale`` mis-specifies the innovation variance by up to ~6x on
    some systems (fold, logistic), which pushes ``c_hat`` down on nulls and
    inflates the fixed-FPR threshold. Running the observer over a grid of
    ``sigma_u`` values (each with the full ``q_grid``) and keeping the pair
    with the highest validation EW-AUC lets the data pick its own noise scale,
    exactly like ``grid_search_q_drift`` does for ``Q_drift``. The observer
    remains non-learned (hyperparameter selection, no training).

    Args:
        y_val_signal: ``(B_sig, T)`` centred mode sequences (signal).
        y_val_null: ``(B_null, T)`` centred mode sequences (null).
        bifs_val: ``(B_sig,)`` bifurcation times of the signal validation set.
        lens_signal_val: ``(B_sig,)`` signal sequence lengths.
        lens_null_val: ``(B_null,)`` null sequence lengths.
        sigma_u_grid: candidate process-noise scales (default
            ``[0.15, 0.3, 0.6, 1.0]``). All entries must be positive.
        r: observation noise variance.
        q_grid: candidate drift-noise values (default: six orders of
            magnitude from ``1e-6`` up to ``1e-1``).
        n_particles, c_min, delta: observer hyperparameters.
        device: torch device.

    Returns:
        The ``(sigma_u, q)`` pair achieving the highest validation EW-AUC.
    """
    if sigma_u_grid is None:
        sigma_u_grid = [0.15, 0.3, 0.6, 1.0]
    if q_grid is None:
        q_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sigma_u_grid = [float(s) for s in sigma_u_grid]
    q_grid = [float(q) for q in q_grid]
    if not sigma_u_grid:
        raise ValueError("sigma_u_grid must contain at least one value")
    if any(s <= 0 for s in sigma_u_grid):
        raise ValueError(f"sigma_u_grid entries must be > 0, got {sigma_u_grid}")
    if not q_grid:
        raise ValueError("q_grid must contain at least one value")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from csd_observer.utils.metrics import compute_early_warning_auc

    y_sig_t = torch.from_numpy(
        np.asarray(y_val_signal, dtype=np.float32)
    ).to(device)
    y_null_t = torch.from_numpy(
        np.asarray(y_val_null, dtype=np.float32)
    ).to(device)
    is_pos_sig = np.ones(y_val_signal.shape[0], dtype=bool)

    best_sigma_u = sigma_u_grid[0]
    best_q = q_grid[0]
    best_auc = -1.0

    for sigma_u in sigma_u_grid:
        for q in q_grid:
            observer = SpectralDriftObserver(
                sigma_u=sigma_u,
                r=r,
                q_drift=q,
                n_particles=n_particles,
                c_min=c_min,
                delta=delta,
            ).to(device)
            observer.eval()
            with torch.no_grad():
                probs_sig = observer(y_sig_t)["collapse_prob"].cpu().numpy()
                probs_null = observer(y_null_t)["collapse_prob"].cpu().numpy()

            auc = compute_early_warning_auc(
                probs_sig,
                bifs_val,
                is_pos_sig,
                lens_signal_val,
                probs_null,
                lens_null_val,
            )
            if np.isfinite(auc) and auc > best_auc:
                best_auc = auc
                best_sigma_u = sigma_u
                best_q = q

    return best_sigma_u, best_q
