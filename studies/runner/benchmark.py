"""Benchmark: evaluate the Kalman-Spectral-Drift observer.

Usage:
    python studies/runner/benchmark.py [run_name ...]

    No args: runs patients_100..patients_500 and high_noise
    One or more args: runs those specific configs

Output:
    outputs/benchmark/<run_name>/<timestamp>/
        configs/resolved.yaml
        results/results.jsonl
        metrics/metrics.json
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
for p in (str(_SRC), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from csd_observer.config.load import load_config  # noqa: E402
from csd_observer.data.bifurcation import build_dataset  # noqa: E402
from csd_observer.models.spectral_drift import (  # noqa: E402
    SpectralDriftObserver,
    extract_mode,
    grid_search_sigma_u_q_drift,
    running_mean_center,
)
from csd_observer.utils.io import OutputWriter  # noqa: E402
from csd_observer.utils.metrics import (  # noqa: E402
    compute_detection_time,
    compute_early_warning_auc,
    compute_null_metrics,
)

SYSTEMS = ("fold", "hopf", "logistic")
METHODS = ("Kalman-Spectral-Drift",)


@dataclass(frozen=True)
class RunResult:
    method: str
    seed: int
    metrics: dict


@dataclass(frozen=True)
class SystemResult:
    system: str
    runs: List[RunResult]

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, list] = {m: [] for m in METHODS}
        for run in self.runs:
            if run.method in grouped:
                grouped[run.method].append(run.metrics)
        out: Dict[str, Dict[str, float]] = {}
        for method, rows in grouped.items():
            if not rows:
                continue
            merged: Dict[str, float] = {}
            for key in rows[0]:
                vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
                merged[key] = float(np.mean(vals)) if vals else float("nan")
            out[method] = merged
        return out


def _compute_per_traj_dts(
    probs: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> List[float]:
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    times: List[float] = []
    for i in range(len(probs)):
        if not is_positive[i] or np.isnan(threshold):
            times.append(float("nan"))
            continue
        tau = bifurcation_times[i]
        if tau <= 0:
            times.append(float("nan"))
            continue
        pre = probs[i, :int(tau)]
        alerts = np.where(pre >= threshold)[0]
        if len(alerts) > 0:
            times.append(float(tau - alerts[0]))
        else:
            times.append(float("nan"))
    return times


_SYSTEM_BUILDERS = {"fold": "fold", "hopf": "hopf", "logistic": "logistic"}

# Observation-noise standard deviation per system, matching the class defaults
# in csd_observer/data/bifurcation.py (fold 0.10, hopf 0.15, logistic 0.05).
_OBS_NOISE_DEFAULT = {"fold": 0.10, "hopf": 0.15, "logistic": 0.05}


def _build_dataset_for_system(
    system: str,
    *,
    null: bool,
    n_trajectories: int,
    noise_scale: float,
    obs_noise_scale: float | None = None,
    seed: int,
    max_length: int = 200,
    generator: str = "classic",
    difficulty: str = "standard",
) -> dict:
    if system not in _SYSTEM_BUILDERS:
        raise ValueError(f"Unknown system: {system!r}. Options: {list(_SYSTEM_BUILDERS)}")
    kwargs = dict(
        n_trajectories=n_trajectories,
        max_length=max_length,
        noise_scale=noise_scale,
        obs_noise_scale=obs_noise_scale,
        seed=seed,
        null=null,
    )
    return build_dataset(system, generator=generator, difficulty=difficulty, **kwargs)


def _run_synthetic_experiment(
    system: str,
    arrays_signal: dict,
    arrays_null: dict,
    *,
    config: dict,
    device: torch.device,
    writer: OutputWriter,
    enabled_methods: Optional[set[str]] = None,
) -> SystemResult:
    data_cfg = config.get("data", {})
    val_idx_s = arrays_signal["split_indices"]["val"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]
    val_idx_n = arrays_null["split_indices"]["val"]

    runs: List[RunResult] = []

    def _enabled(name: str) -> bool:
        return enabled_methods is None or name in enabled_methods

    # --- Kalman-Spectral-Drift (Rao-Blackwellised spectral-gap observer) ---
    if _enabled("Kalman-Spectral-Drift"):
        sd_cfg = config.get("model", {}).get("spectral_drift", {})
        n_particles = int(sd_cfg.get("n_particles", 500))
        c_min = float(sd_cfg.get("c_min", 1e-3))
        delta = float(sd_cfg.get("delta", 0.05))
        center_window = int(sd_cfg.get("center_window", 50))
        q_grid = [float(q) for q in sd_cfg.get("q_drift_grid", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])]
        sigma_u_grid = [float(s) for s in sd_cfg.get("sigma_u_grid", [0.15, 0.3, 0.6, 1.0])]
        obs_noise = float(data_cfg.get("obs_noise_scale") or _OBS_NOISE_DEFAULT[system])
        r_var = obs_noise ** 2

        # Extract the scalar dominant mode and centre it (running mean).
        mode_sig = extract_mode(arrays_signal["features"], system)
        mode_null = extract_mode(arrays_null["features"], system)
        mode_sig = running_mean_center(mode_sig, arrays_signal["seq_lengths"], center_window)
        mode_null = running_mean_center(mode_null, arrays_null["seq_lengths"], center_window)

        # Grid-search (sigma_u, Q_drift) on the validation split (signal +
        # null). The per-step noise scale sigma_u is a free hyperparameter:
        # fixing it to noise_scale mis-specifies the innovation variance by
        # up to ~6x on some systems, which degrades the alarm (see
        # docs/notes/spectral_drift_diagnosis.md).
        best_sigma_u_sd, best_q_sd = grid_search_sigma_u_q_drift(
            mode_sig[val_idx_s],
            mode_null[val_idx_n],
            arrays_signal["bifurcation_times"][val_idx_s],
            arrays_signal["seq_lengths"][val_idx_s],
            arrays_null["seq_lengths"][val_idx_n],
            sigma_u_grid=sigma_u_grid,
            r=r_var,
            q_grid=q_grid,
            n_particles=n_particles,
            c_min=c_min,
            delta=delta,
            device=device,
        )

        observer_sd = SpectralDriftObserver(
            sigma_u=best_sigma_u_sd,
            r=r_var,
            q_drift=best_q_sd,
            n_particles=n_particles,
            c_min=c_min,
            delta=delta,
        ).to(device)
        observer_sd.eval()

        def _collapse_probs(mode: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(np.asarray(mode, dtype=np.float32)).to(device)
            with torch.no_grad():
                return observer_sd(x)["collapse_prob"].cpu().numpy()

        probs_test_sd = _collapse_probs(mode_sig[test_idx_s])
        probs_null_sd = _collapse_probs(mode_null[test_idx_n])
        probs_val_sd = _collapse_probs(mode_sig[val_idx_s])
        probs_val_null_sd = _collapse_probs(mode_null[val_idx_n])

        # Calibrate the alarm threshold to a target false-positive rate
        # against null trajectories (formal definition, section 5: fixed-FPR
        # rule). The Youden rule degenerates for posterior-probability alarms:
        # the collapse probability carries a prior floor at t=0, so a
        # sensitivity-maximising threshold fires on most null steps.
        fpr_target = float(sd_cfg.get("fpr_target", 0.05))
        null_steps = np.concatenate(
            [
                probs_val_null_sd[i, : int(length)]
                for i, length in enumerate(arrays_null["seq_lengths"][val_idx_n])
            ]
        )
        thresh_sd = float(np.percentile(null_steps, 100.0 * (1.0 - fpr_target)))

        dt_sd = compute_detection_time(
            probs_test_sd, arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            arrays_signal["seq_lengths"][test_idx_s], thresh_sd,
        )
        ewa_sd = compute_early_warning_auc(
            probs_test_sd, arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            arrays_signal["seq_lengths"][test_idx_s],
            probs_null_sd, arrays_null["seq_lengths"][test_idx_n],
        )
        null_m_sd = compute_null_metrics(probs_null_sd, thresh_sd, arrays_null["seq_lengths"][test_idx_n])
        per_traj_dts_sd = _compute_per_traj_dts(
            probs_test_sd,
            arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            arrays_signal["seq_lengths"][test_idx_s],
            thresh_sd,
        )
        writer.write_trajectory_data(
            system, "Kalman-Spectral-Drift", 0,
            probs_test=probs_test_sd,
            probs_null=probs_null_sd,
            probs_val=probs_val_sd,
            bifurcation_times=arrays_signal["bifurcation_times"][test_idx_s],
            bifurcation_times_null=arrays_null["bifurcation_times"][test_idx_n],
            seq_lengths=arrays_signal["seq_lengths"][test_idx_s],
            seq_lengths_null=arrays_null["seq_lengths"][test_idx_n],
            threshold=np.array([thresh_sd]),
            detection_times=np.array(per_traj_dts_sd, dtype=np.float32),
        )

        sd_metrics = {
            "detection_time": dt_sd, "ew_auc": ewa_sd, **null_m_sd,
        }
        runs.append(RunResult(method="Kalman-Spectral-Drift", seed=0, metrics=sd_metrics))
        writer.write_result_row({
            "system": system, "seed": 0, "method": "Kalman-Spectral-Drift",
            **sd_metrics,
            "threshold": thresh_sd,
            "n_epochs_trained": 0,
            "q_drift": best_q_sd,
            "sigma_u": best_sigma_u_sd,
        })

    return SystemResult(system=system, runs=runs)


def _summarize_system(agg: Dict[str, Dict[str, float]], system: str) -> None:
    print(f"\nSystem: {system.capitalize()} Bifurcation")
    print("-" * 70)
    print(f"{'Method':<20s} {'DT':>10s} {'EW-AUC':>10s} {'FPR':>10s}")
    print("-" * 50)
    for method in METHODS:
        m = agg.get(method, {})
        dt = m.get("detection_time", float("nan"))
        ewa = m.get("ew_auc", float("nan"))
        fpr = m.get("fpr", float("nan"))
        dt_s = f"{dt:.1f}" if np.isfinite(dt) else "nan"
        ewa_s = f"{ewa:.3f}" if np.isfinite(ewa) else "nan"
        fpr_s = f"{fpr:.4f}" if np.isfinite(fpr) else "nan"
        print(f"{method:<20s} {dt_s:>10s} {ewa_s:>10s} {fpr_s:>10s}")


def _parse_args() -> Tuple[List[str], Optional[int], Optional[str], Optional[str], Optional[set[str]]]:
    n_seeds_override: Optional[int] = None
    generator_override: Optional[str] = None
    difficulty_override: Optional[str] = None
    methods_override: Optional[set[str]] = None
    run_names: List[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("n_seeds="):
            n_seeds_override = int(arg.split("=", 1)[1])
        elif arg.startswith("generator="):
            generator_override = arg.split("=", 1)[1]
        elif arg.startswith("difficulty="):
            difficulty_override = arg.split("=", 1)[1]
        elif arg.startswith("methods="):
            raw = arg.split("=", 1)[1]
            if raw.strip().lower() == "all":
                methods_override = None
            else:
                chosen = {m.strip() for m in raw.split(",") if m.strip()}
                valid = set(METHODS)
                invalid = chosen - valid
                if invalid:
                    raise ValueError(
                        f"Unknown method(s): {sorted(invalid)}. "
                        f"Valid methods: {', '.join(METHODS)}"
                    )
                methods_override = chosen
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            run_names.append(arg)
    if not run_names:
        run_names = ["patients_100", "patients_200", "patients_300", "patients_400", "patients_500", "high_noise"]
    return run_names, n_seeds_override, generator_override, difficulty_override, methods_override


def _run_single(
    run_name: str,
    n_seeds_override: Optional[int] = None,
    generator_override: Optional[str] = None,
    difficulty_override: Optional[str] = None,
    enabled_methods: Optional[set[str]] = None,
) -> None:
    print(f"Loading config: {run_name}")
    config = load_config(run_name)
    data_cfg = config.get("data", {})
    if n_seeds_override is not None:
        data_cfg["n_seeds"] = n_seeds_override
        print(f"  [override] n_seeds={n_seeds_override}")
    if enabled_methods is not None:
        print(f"  [override] methods={','.join(sorted(enabled_methods))}")
    if generator_override is not None:
        data_cfg["generator"] = generator_override
        print(f"  [override] generator={generator_override}")
    if difficulty_override is not None:
        data_cfg["difficulty"] = difficulty_override
        print(f"  [override] difficulty={difficulty_override}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"Settings: noise_scale={data_cfg.get('noise_scale')}, "
          f"n_patients={data_cfg.get('n_patients')}, "
          f"epochs={config.get('training', {}).get('epochs')}")
    print()

    writer = OutputWriter(experiment_name=f"{os.environ.get('CSD_EXPERIMENT_PREFIX', 'benchmark')}/{run_name}")
    writer.write_config(config)
    print(f"Output: {writer.path}\n")

    systems = data_cfg.get("systems", SYSTEMS)
    max_length = data_cfg.get("max_length", 200)
    n_patients = data_cfg.get("n_patients", 500)
    noise_scale = data_cfg.get("noise_scale", 0.15)
    obs_noise_scale = data_cfg.get("obs_noise_scale")
    seed_offset = data_cfg.get("seed_offset", 0)

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    started = time.time()

    for system in systems:
        print(f"--- Generating {system} data (Synthetic Pipeline) ---")
        generator = data_cfg.get("generator", "classic")
        difficulty = data_cfg.get("difficulty", "standard")
        data_kwargs = dict(
            n_trajectories=n_patients, noise_scale=noise_scale,
            obs_noise_scale=obs_noise_scale, max_length=max_length,
        )
        arrays_signal = _build_dataset_for_system(
            system, null=False, seed=seed_offset + 101,
            generator=generator, difficulty=difficulty, **data_kwargs,
        )
        arrays_null = _build_dataset_for_system(
            system, null=True, seed=seed_offset + 202,
            generator=generator, difficulty=difficulty, **data_kwargs,
        )

        print(f"  signal: {arrays_signal['features'].shape}, null: {arrays_null['features'].shape}")

        sys_res = _run_synthetic_experiment(
            system,
            arrays_signal, arrays_null,
            config=config, device=device, writer=writer,
            enabled_methods=enabled_methods,
        )

        agg = sys_res.aggregate()
        all_metrics[system] = agg
        _summarize_system(agg, system)

    writer.write_metrics(all_metrics)

    elapsed = time.time() - started
    print(f"\nTime: {elapsed:.1f}s")

    print(f"\nAll results saved to: {writer.path}")


def main() -> None:
    run_names, n_seeds_override, generator_override, difficulty_override, methods_override = _parse_args()
    total_started = time.time()
    failed_runs: List[str] = []
    for i, run_name in enumerate(run_names, 1):
        tag = f"[{i}/{len(run_names)}] " if len(run_names) > 1 else ""
        print(f"\n{tag}{'='*70}")
        print(f"{tag}RUN: {run_name}")
        print(f"{tag}{'='*70}")
        try:
            _run_single(run_name, n_seeds_override, generator_override, difficulty_override, methods_override)
        except Exception as e:
            import traceback
            print(f"\nERROR: {run_name} failed: {e}")
            traceback.print_exc()
            failed_runs.append(run_name)
            continue
    total_elapsed = time.time() - total_started
    print(f"\n{'='*70}")
    if failed_runs:
        print(f"WARNING: {len(failed_runs)} run(s) FAILED: {', '.join(failed_runs)}")
    print(f"All runs complete. Total time: {total_elapsed:.1f}s")
    if failed_runs:
        sys.exit(1)


if __name__ == "__main__":
    main()
