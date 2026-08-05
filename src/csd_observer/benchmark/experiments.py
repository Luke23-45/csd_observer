"""Method evaluation drivers.

One shared driver for the seven CSD indicators plus the spectral-drift
driver. Both implement the identical governance pipeline
(``docs/plan/benchmark_revision_plan.md`` §4):

    preprocess -> scores on test/val x signal/null -> fixed-FPR
    calibration on val-null -> metrics -> trajectory artifact -> row

so every benchmark row is evaluated exactly like the spectral block and
the results are directly comparable (plan §6: "a single helper removes
repetition and the risk of per-method drift").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from csd_observer.benchmark.methods import (
    MethodSpec,
    indicator_fpr_target,
    resolve_params,
)
from csd_observer.models.spectral_drift import (
    SpectralDriftObserver,
    extract_mode,
    grid_search_sigma_u_q_drift,
    running_mean_center,
)
from csd_observer.utils.evaluation import (
    calibrate_threshold,
    compute_detection_time,
    compute_early_warning_auc,
    compute_null_metrics,
    compute_per_traj_dts,
)
from csd_observer.utils.io import OutputWriter

# Observation-noise standard deviation per system, matching the class
# defaults in csd_observer/data/bifurcation.py (fold 0.10, hopf 0.15,
# logistic 0.05).
_OBS_NOISE_DEFAULT = {"fold": 0.10, "hopf": 0.15, "logistic": 0.05}


@dataclass(frozen=True)
class MethodRun:
    """Metrics of one method on one (system, seed) pair."""

    method: str
    seed: int
    metrics: Dict[str, float]


def _mode_input(features: np.ndarray, system: str) -> np.ndarray:
    """Shared preprocessed input: the scalar dominant mode as a
    single-channel array (plan §2 — identical preprocessing across every
    row; for Hopf this is the radial mode ``sqrt(x1^2 + x2^2)``)."""
    return extract_mode(features, system)[:, :, None]


def _write_artifacts(
    writer: OutputWriter,
    system: str,
    method: str,
    seed: int,
    *,
    scores_test: np.ndarray,
    scores_null: np.ndarray,
    scores_val: np.ndarray,
    arrays_signal: dict,
    arrays_null: dict,
    test_idx_s: np.ndarray,
    test_idx_n: np.ndarray,
    threshold: float,
    per_traj_dts: list,
) -> None:
    writer.write_trajectory_data(
        system,
        method,
        seed,
        probs_test=scores_test,
        probs_null=scores_null,
        probs_val=scores_val,
        bifurcation_times=arrays_signal["bifurcation_times"][test_idx_s],
        bifurcation_times_null=arrays_null["bifurcation_times"][test_idx_n],
        seq_lengths=arrays_signal["seq_lengths"][test_idx_s],
        seq_lengths_null=arrays_null["seq_lengths"][test_idx_n],
        threshold=np.array([threshold]),
        detection_times=np.asarray(per_traj_dts, dtype=np.float32),
    )


def evaluate_indicator(
    spec: MethodSpec,
    arrays_signal: dict,
    arrays_null: dict,
    *,
    system: str,
    config: dict,
    writer: OutputWriter,
    seed: int,
) -> MethodRun:
    """Run one CSD indicator through the shared governance pipeline."""
    assert spec.scorer is not None, f"{spec.name} has no scorer"
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    params = resolve_params(spec, model_cfg)
    fpr_target = indicator_fpr_target(model_cfg)

    mode_sig = _mode_input(arrays_signal["features"], system)
    mode_null = _mode_input(arrays_null["features"], system)
    lens_sig = arrays_signal["seq_lengths"]
    lens_null = arrays_null["seq_lengths"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]
    val_idx_s = arrays_signal["split_indices"]["val"]
    val_idx_n = arrays_null["split_indices"]["val"]

    scores_test = spec.scorer(mode_sig[test_idx_s], lens_sig[test_idx_s], **params)
    scores_val = spec.scorer(mode_sig[val_idx_s], lens_sig[val_idx_s], **params)
    scores_null = spec.scorer(mode_null[test_idx_n], lens_null[test_idx_n], **params)
    scores_val_null = spec.scorer(mode_null[val_idx_n], lens_null[val_idx_n], **params)

    threshold = calibrate_threshold(scores_val_null, lens_null[val_idx_n], fpr_target)
    metrics = {
        "detection_time": compute_detection_time(
            scores_test,
            arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            lens_sig[test_idx_s],
            threshold,
        ),
        "ew_auc": compute_early_warning_auc(
            scores_test,
            arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            lens_sig[test_idx_s],
            scores_null,
            lens_null[test_idx_n],
        ),
        **compute_null_metrics(scores_null, threshold, lens_null[test_idx_n]),
    }
    per_traj_dts = compute_per_traj_dts(
        scores_test,
        arrays_signal["bifurcation_times"][test_idx_s],
        arrays_signal["is_positive"][test_idx_s],
        lens_sig[test_idx_s],
        threshold,
    )
    _write_artifacts(
        writer,
        system,
        spec.name,
        seed,
        scores_test=scores_test,
        scores_null=scores_null,
        scores_val=scores_val,
        arrays_signal=arrays_signal,
        arrays_null=arrays_null,
        test_idx_s=test_idx_s,
        test_idx_n=test_idx_n,
        threshold=threshold,
        per_traj_dts=per_traj_dts,
    )

    row: Dict[str, Any] = {
        "system": system,
        "seed": seed,
        "method": spec.name,
        **metrics,
        "threshold": threshold,
        "n_epochs_trained": 0,
        "generator": data_cfg.get("generator", "classic"),
        "difficulty": data_cfg.get("difficulty", "standard"),
    }
    row.update(params)
    writer.write_result_row(row)
    return MethodRun(method=spec.name, seed=seed, metrics=metrics)


def evaluate_spectral(
    spec: MethodSpec,
    arrays_signal: dict,
    arrays_null: dict,
    *,
    system: str,
    config: dict,
    writer: OutputWriter,
    seed: int,
    device: torch.device,
) -> MethodRun:
    """Run the spectral-drift observer through the shared pipeline.

    Identical governance to the indicators; the only differences are
    mode + running-mean centring preprocessing and the validation
    grid search over ``(sigma_u, Q_drift)``.
    """
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    sd_cfg = model_cfg.get("spectral_drift", {})
    n_particles = int(sd_cfg.get("n_particles", 500))
    c_min = float(sd_cfg.get("c_min", 1e-3))
    delta = float(sd_cfg.get("delta", 0.05))
    center_window = int(sd_cfg.get("center_window", 50))
    q_grid = [float(q) for q in sd_cfg.get("q_drift_grid", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])]
    sigma_u_grid = [float(s) for s in sd_cfg.get("sigma_u_grid", [0.15, 0.3, 0.6, 1.0])]
    fpr_target = float(sd_cfg.get("fpr_target", 0.05))
    obs_noise_cfg = data_cfg.get("obs_noise_scale")
    obs_noise = float(obs_noise_cfg) if obs_noise_cfg is not None else _OBS_NOISE_DEFAULT[system]
    r_var = obs_noise ** 2

    mode_sig = running_mean_center(
        extract_mode(arrays_signal["features"], system),
        arrays_signal["seq_lengths"],
        center_window,
    )
    mode_null = running_mean_center(
        extract_mode(arrays_null["features"], system),
        arrays_null["seq_lengths"],
        center_window,
    )
    lens_sig = arrays_signal["seq_lengths"]
    lens_null = arrays_null["seq_lengths"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]
    val_idx_s = arrays_signal["split_indices"]["val"]
    val_idx_n = arrays_null["split_indices"]["val"]

    best_sigma_u, best_q = grid_search_sigma_u_q_drift(
        mode_sig[val_idx_s],
        mode_null[val_idx_n],
        arrays_signal["bifurcation_times"][val_idx_s],
        lens_sig[val_idx_s],
        lens_null[val_idx_n],
        sigma_u_grid=sigma_u_grid,
        r=r_var,
        q_grid=q_grid,
        n_particles=n_particles,
        c_min=c_min,
        delta=delta,
        device=device,
    )
    observer = SpectralDriftObserver(
        sigma_u=best_sigma_u,
        r=r_var,
        q_drift=best_q,
        n_particles=n_particles,
        c_min=c_min,
        delta=delta,
    ).to(device)
    observer.eval()

    def _collapse_probs(mode: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(mode, dtype=np.float32)).to(device)
        with torch.no_grad():
            return observer(x)["collapse_prob"].cpu().numpy()

    scores_test = _collapse_probs(mode_sig[test_idx_s])
    scores_val = _collapse_probs(mode_sig[val_idx_s])
    scores_null = _collapse_probs(mode_null[test_idx_n])
    scores_val_null = _collapse_probs(mode_null[val_idx_n])

    threshold = calibrate_threshold(scores_val_null, lens_null[val_idx_n], fpr_target)
    metrics = {
        "detection_time": compute_detection_time(
            scores_test,
            arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            lens_sig[test_idx_s],
            threshold,
        ),
        "ew_auc": compute_early_warning_auc(
            scores_test,
            arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            lens_sig[test_idx_s],
            scores_null,
            lens_null[test_idx_n],
        ),
        **compute_null_metrics(scores_null, threshold, lens_null[test_idx_n]),
    }
    per_traj_dts = compute_per_traj_dts(
        scores_test,
        arrays_signal["bifurcation_times"][test_idx_s],
        arrays_signal["is_positive"][test_idx_s],
        lens_sig[test_idx_s],
        threshold,
    )
    _write_artifacts(
        writer,
        system,
        spec.name,
        seed,
        scores_test=scores_test,
        scores_null=scores_null,
        scores_val=scores_val,
        arrays_signal=arrays_signal,
        arrays_null=arrays_null,
        test_idx_s=test_idx_s,
        test_idx_n=test_idx_n,
        threshold=threshold,
        per_traj_dts=per_traj_dts,
    )

    writer.write_result_row(
        {
            "system": system,
            "seed": seed,
            "method": spec.name,
            **metrics,
            "threshold": threshold,
            "n_epochs_trained": 0,
            "q_drift": best_q,
            "sigma_u": best_sigma_u,
            "generator": data_cfg.get("generator", "classic"),
            "difficulty": data_cfg.get("difficulty", "standard"),
        }
    )
    return MethodRun(method=spec.name, seed=seed, metrics=metrics)


__all__ = ["MethodRun", "evaluate_indicator", "evaluate_spectral"]
