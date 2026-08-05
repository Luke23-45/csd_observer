"""Benchmark suite: one run configuration, all systems x seeds x methods.

Orchestration for ``outputs/benchmark/<run_name>/<timestamp>/``:

* loads and resolves a run config (``csd_observer.config.load``),
* cross-checks the method catalog against the model config,
* for each system and each seed builds the signal + null datasets once
  and evaluates every enabled method on them (shared data, so the
  expensive indicators are paid once per seed, not once per method),
* calibrates each method's alarm threshold on that seed's validation
  nulls (fixed-FPR rule) and reports detection time, EW-AUC and FPR,
* aggregates per-seed metrics by the mean over finite entries and
  writes ``metrics/metrics.json`` plus one ``results/results.jsonl`` row
  per (system, method, seed).

Seed schedule: signal seed ``seed_offset + s * 1000 + 101``, null seed
``seed_offset + s * 1000 + 202`` — seed 0 reproduces the historical
101/202 schedule exactly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from csd_observer.benchmark.experiments import MethodRun, evaluate_indicator, evaluate_spectral
from csd_observer.benchmark.methods import METHOD_NAMES, METHODS, get_method, validate_catalog
from csd_observer.config.load import load_config
from csd_observer.data.bifurcation import build_dataset
from csd_observer.utils.io import OutputWriter

_SYSTEM_BUILDERS = {"fold": "fold", "hopf": "hopf", "logistic": "logistic"}


def _build_dataset_for_system(
    system: str,
    *,
    null: bool,
    n_trajectories: int,
    noise_scale: float,
    obs_noise_scale: float | None,
    seed: int,
    max_length: int,
    generator: str,
    difficulty: str,
) -> dict:
    if system not in _SYSTEM_BUILDERS:
        raise ValueError(f"Unknown system: {system!r}. Options: {list(_SYSTEM_BUILDERS)}")
    return build_dataset(
        system,
        n_trajectories=n_trajectories,
        max_length=max_length,
        noise_scale=noise_scale,
        obs_noise_scale=obs_noise_scale,
        seed=seed,
        null=null,
        generator=generator,
        difficulty=difficulty,
    )


@dataclass(frozen=True)
class SystemResult:
    system: str
    runs: List[MethodRun]

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[MethodRun]] = {m: [] for m in METHOD_NAMES}
        for run in self.runs:
            grouped[run.method].append(run)
        out: Dict[str, Dict[str, float]] = {}
        for method, runs in grouped.items():
            if not runs:
                continue
            merged: Dict[str, float] = {}
            for key in runs[0].metrics:
                vals = [r.metrics[key] for r in runs if np.isfinite(r.metrics[key])]
                merged[key] = float(np.mean(vals)) if vals else float("nan")
            out[method] = merged
        return out


def _print_summary(agg: Dict[str, Dict[str, float]], system: str) -> None:
    print(f"\nSystem: {system.capitalize()} Bifurcation")
    print("-" * 70)
    print(f"{'Method':<22s} {'DT':>10s} {'EW-AUC':>10s} {'FPR':>10s}")
    print("-" * 58)
    for method in METHOD_NAMES:
        m = agg.get(method, {})
        dt = m.get("detection_time", float("nan"))
        ewa = m.get("ew_auc", float("nan"))
        fpr = m.get("fpr", float("nan"))
        dt_s = f"{dt:.1f}" if np.isfinite(dt) else "nan"
        ewa_s = f"{ewa:.3f}" if np.isfinite(ewa) else "nan"
        fpr_s = f"{fpr:.4f}" if np.isfinite(fpr) else "nan"
        print(f"{method:<22s} {dt_s:>10s} {ewa_s:>10s} {fpr_s:>10s}")


def run_config(
    run_name: str,
    *,
    config_root: str | Path | None = None,
    n_seeds: Optional[int] = None,
    generator: Optional[str] = None,
    difficulty: Optional[str] = None,
    enabled_methods: Optional[Set[str]] = None,
    data_overrides: Optional[dict] = None,
    model_overrides: Optional[dict] = None,
    device: Optional[torch.device] = None,
    writer: Optional[OutputWriter] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run one configuration and return per-system aggregated metrics.

    Args:
        run_name: run config name under ``configs/run/``.
        config_root: config directory root (defaults to the repo's).
        n_seeds: overrides ``data.n_seeds``.
        generator: overrides ``data.generator`` ("classic" | "bury").
        difficulty: overrides ``data.difficulty`` ("standard" | "hard").
        enabled_methods: method-name subset, or ``None`` for all.
        data_overrides / model_overrides: extra config overrides
            (test and diagnostic hooks; applied after the CLI ones).
        device: torch device for the spectral observer.
        writer: output sink (created under ``outputs/`` when omitted).

    Returns:
        ``{system: {method: {metric: mean-over-seeds}}}``.
    """
    config = load_config(run_name, config_root)
    data_cfg = config["data"]
    model_cfg = config["model"]

    if n_seeds is not None:
        data_cfg["n_seeds"] = n_seeds
        print(f"  [override] n_seeds={n_seeds}")
    if generator is not None:
        data_cfg["generator"] = generator
        print(f"  [override] generator={generator}")
    if difficulty is not None:
        data_cfg["difficulty"] = difficulty
        print(f"  [override] difficulty={difficulty}")
    if data_overrides:
        data_cfg.update(data_overrides)
    if model_overrides:
        model_cfg.update(model_overrides)

    validate_catalog(model_cfg)

    if enabled_methods is None:
        specs = list(METHODS)
    else:
        if not enabled_methods:
            raise ValueError("enabled_methods must not be empty")
        specs = [get_method(name) for name in sorted(enabled_methods)]
        print(f"  [override] methods={','.join(sorted(enabled_methods))}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(
        f"Settings: noise_scale={data_cfg.get('noise_scale')}, "
        f"n_patients={data_cfg.get('n_patients')}, "
        f"n_seeds={data_cfg.get('n_seeds')}"
    )
    print()

    if writer is None:
        prefix = os.environ.get("CSD_EXPERIMENT_PREFIX", "benchmark")
        writer = OutputWriter(experiment_name=f"{prefix}/{run_name}")
    writer.write_config(config)
    print(f"Output: {writer.path}\n")

    systems = data_cfg.get("systems", ["fold", "hopf", "logistic"])
    n_patients = int(data_cfg.get("n_patients", 500))
    max_length = int(data_cfg.get("max_length", 200))
    noise_scale = float(data_cfg.get("noise_scale", 0.15))
    obs_noise_scale = data_cfg.get("obs_noise_scale")
    n_seed_loop = int(data_cfg.get("n_seeds", 1))
    seed_offset = int(data_cfg.get("seed_offset", 0))
    gen = str(data_cfg.get("generator", "classic"))
    diff = str(data_cfg.get("difficulty", "standard"))

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    started = time.time()

    for system in systems:
        print(f"--- Generating {system} data (Synthetic Pipeline) ---")
        sys_runs: List[MethodRun] = []
        for s in range(n_seed_loop):
            signal_seed = seed_offset + s * 1000 + 101
            null_seed = seed_offset + s * 1000 + 202
            arrays_signal = _build_dataset_for_system(
                system,
                null=False,
                n_trajectories=n_patients,
                noise_scale=noise_scale,
                obs_noise_scale=obs_noise_scale,
                seed=signal_seed,
                max_length=max_length,
                generator=gen,
                difficulty=diff,
            )
            arrays_null = _build_dataset_for_system(
                system,
                null=True,
                n_trajectories=n_patients,
                noise_scale=noise_scale,
                obs_noise_scale=obs_noise_scale,
                seed=null_seed,
                max_length=max_length,
                generator=gen,
                difficulty=diff,
            )
            print(
                f"  seed {s}: signal {arrays_signal['features'].shape}, "
                f"null {arrays_null['features'].shape}"
            )
            for spec in specs:
                kwargs = dict(
                    arrays_signal=arrays_signal,
                    arrays_null=arrays_null,
                    system=system,
                    config=config,
                    writer=writer,
                    seed=s,
                )
                if spec.family == "indicator":
                    sys_runs.append(evaluate_indicator(spec, **kwargs))
                else:
                    sys_runs.append(
                        evaluate_spectral(spec, **kwargs, device=device)
                    )

        sys_res = SystemResult(system=system, runs=sys_runs)
        agg = sys_res.aggregate()
        all_metrics[system] = agg
        _print_summary(agg, system)

    writer.write_metrics(all_metrics)
    print(f"\nTime: {time.time() - started:.1f}s")
    print(f"All results saved to: {writer.path}")
    return all_metrics


__all__ = ["SystemResult", "run_config"]
