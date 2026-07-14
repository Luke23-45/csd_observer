"""Benchmark: LSTM head vs MLP head on Kalman observer.

Usage:
    python studies/runner/benchmark.py [run_name ...]

    No args: runs all three (default, high_noise, low_data)
    One or more args: runs those specific configs

Output:
    outputs/benchmark/<run_name>/<timestamp>/
        configs/resolved.yaml
        results/results.jsonl
        metrics/metrics.json
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
for p in (str(_SRC), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from csd_observer.config.load import load_config  # noqa: E402
from csd_observer.data.bifurcation import build_dataset  # noqa: E402
from csd_observer.training.trainer import (  # noqa: E402
    TensorizedDataset,
    build_probs,
    tensorize,
    train_kalman,
)
from csd_observer.utils.io import OutputWriter  # noqa: E402
from csd_observer.utils.metrics import (  # noqa: E402
    compute_detection_time,
    compute_early_warning_auc,
    compute_null_metrics,
    evaluate_raw_csd,
    raw_csd_indicator,
    select_threshold,
)

SYSTEMS = ("fold", "hopf", "logistic")
METHODS = ("Raw-CSD", "Kalman-BCE", "Kalman-LSTM", "Kalman-LSTM-Spec")
_REAL_SYSTEMS: Dict[str, Callable] = {}


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


def _mean_metric(agg: Dict[str, Dict[str, float]], method: str, metric: str) -> float:
    return float(agg.get(method, {}).get(metric, float("nan")))


_SYSTEM_BUILDERS = {"fold": "fold", "hopf": "hopf", "logistic": "logistic"}


def _build_dataset_for_system(
    system: str,
    *,
    null: bool,
    n_trajectories: int,
    noise_scale: float,
    obs_noise_scale: float | None = None,
    seed: int,
    max_length: int = 200,
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
    return build_dataset(system, **kwargs)



def _run_empirical_experiment(
    system: str,
    arrays_signal: dict,
    arrays_null: dict,
    *,
    config: dict,
    device: torch.device,
    writer: OutputWriter,
) -> SystemResult:
    # 1. Merge the arrays for training on both
    B_sig = len(arrays_signal["is_positive"])
    features = np.concatenate([arrays_signal["features"], arrays_null["features"]], axis=0)
    seq_lengths = np.concatenate([arrays_signal["seq_lengths"], arrays_null["seq_lengths"]], axis=0)
    bifurcation_times = np.concatenate([arrays_signal["bifurcation_times"], arrays_null["bifurcation_times"]], axis=0)
    is_positive = np.concatenate([arrays_signal["is_positive"], arrays_null["is_positive"]], axis=0)
    
    split_indices = {}
    for split in ("train", "val", "test"):
        idx_sig = arrays_signal["split_indices"][split]
        idx_null = arrays_null["split_indices"][split] + B_sig
        split_indices[split] = np.concatenate([idx_sig, idx_null])
        
    merged_dataset = {
        "features": features,
        "seq_lengths": seq_lengths,
        "bifurcation_times": bifurcation_times,
        "is_positive": is_positive,
        "split_indices": split_indices,
        "augment_features": arrays_signal.get("augment_features", False)
    }
    
    tensors_merged = tensorize(merged_dataset, device)
    
    train_idx = merged_dataset["split_indices"]["train"]
    val_idx = merged_dataset["split_indices"]["val"]
    
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]
    
    tensors_signal = tensorize(arrays_signal, device)
    tensors_null = tensorize(arrays_null, device)

    data_cfg = config.get("data", {})
    n_seeds = data_cfg.get("n_seeds", 5)
    seed_offset = data_cfg.get("seed_offset", 0)
    seeds = [seed_offset + 101 + 101 * i for i in range(n_seeds)]

    runs: List[RunResult] = []

    csd_scores_test = raw_csd_indicator(arrays_signal["features"][test_idx_s], arrays_signal["seq_lengths"][test_idx_s], 30)
    csd_scores_null_test = raw_csd_indicator(arrays_null["features"][test_idx_n], arrays_null["seq_lengths"][test_idx_n], 30)
    raw_metrics = evaluate_raw_csd(
        csd_scores_test,
        arrays_signal["bifurcation_times"][test_idx_s],
        arrays_signal["is_positive"][test_idx_s],
        arrays_signal["seq_lengths"][test_idx_s],
        csd_scores_null_test,
        arrays_null["seq_lengths"][test_idx_n],
        threshold=0.6,
    )
    runs.append(RunResult(method="Raw-CSD", seed=0, metrics=raw_metrics))
    writer.write_result_row({"system": system, "seed": 0, "method": "Raw-CSD", **raw_metrics})

    methods_list = [
        ("Kalman-BCE", "bce"),
        ("Kalman-LSTM", "lstm"),
        ("Kalman-LSTM-Spec", "lstm_spec"),
    ]
    total = len(seeds) * len(methods_list)
    pbar = tqdm(total=total, desc=f"{system}", unit="run", leave=False)
    for seed in seeds:
        for method_name, loss_type in methods_list:
            pbar.set_description(f"{system} {method_name}")
            
            # TRAIN on merged dataset
            model = train_kalman(
                tensors_merged, train_idx, val_idx,
                loss_type=loss_type, seed=seed, config=config, device=device,
            )

            # EVALUATE on separate signal/null (just like synthetic!)
            probs_test = build_probs(model, tensors_signal, test_idx_s)
            probs_null = build_probs(model, tensors_null, test_idx_n)
            
            # Use merged val for threshold selection
            probs_val = build_probs(model, tensors_merged, val_idx)
            thresh = select_threshold(
                probs_val,
                merged_dataset["bifurcation_times"][val_idx],
                merged_dataset["is_positive"][val_idx],
                merged_dataset["seq_lengths"][val_idx],
            )

            dt = compute_detection_time(
                probs_test,
                arrays_signal["bifurcation_times"][test_idx_s],
                arrays_signal["is_positive"][test_idx_s],
                arrays_signal["seq_lengths"][test_idx_s],
                thresh,
            )
            ewa = compute_early_warning_auc(
                probs_test,
                arrays_signal["bifurcation_times"][test_idx_s],
                arrays_signal["is_positive"][test_idx_s],
                arrays_signal["seq_lengths"][test_idx_s],
                probs_null,
                arrays_null["seq_lengths"][test_idx_n],
            )
            null_met = compute_null_metrics(probs_null, thresh, arrays_null["seq_lengths"][test_idx_n])

            metrics = {"detection_time": dt, "ew_auc": ewa, **null_met}
            runs.append(RunResult(method=method_name, seed=seed, metrics=metrics))
            writer.write_result_row({"system": system, "seed": seed, "method": method_name, **metrics})
            pbar.update(1)
    pbar.close()

    return SystemResult(system=system, runs=runs)


def _run_synthetic_experiment(
    system: str,
    tensors_signal: TensorizedDataset,
    tensors_null: TensorizedDataset,
    arrays_signal: dict,
    arrays_null: dict,
    *,
    config: dict,
    device: torch.device,
    writer: OutputWriter,
) -> SystemResult:
    data_cfg = config.get("data", {})
    train_idx_s = arrays_signal["split_indices"]["train"]
    val_idx_s = arrays_signal["split_indices"]["val"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]

    n_seeds = data_cfg.get("n_seeds", 5)
    seed_offset = data_cfg.get("seed_offset", 0)
    seeds = [seed_offset + 101 + 101 * i for i in range(n_seeds)]

    runs: List[RunResult] = []

    csd_scores_test = raw_csd_indicator(arrays_signal["features"][test_idx_s], arrays_signal["seq_lengths"][test_idx_s], 30)
    csd_scores_null_test = raw_csd_indicator(arrays_null["features"][test_idx_n], arrays_null["seq_lengths"][test_idx_n], 30)
    raw_metrics = evaluate_raw_csd(
        csd_scores_test,
        arrays_signal["bifurcation_times"][test_idx_s],
        arrays_signal["is_positive"][test_idx_s],
        arrays_signal["seq_lengths"][test_idx_s],
        csd_scores_null_test,
        arrays_null["seq_lengths"][test_idx_n],
        threshold=0.6,
    )
    runs.append(RunResult(method="Raw-CSD", seed=0, metrics=raw_metrics))
    writer.write_result_row({"system": system, "seed": 0, "method": "Raw-CSD", **raw_metrics})

    methods_list = [
        ("Kalman-BCE", "bce"),
        ("Kalman-LSTM", "lstm"),
        ("Kalman-LSTM-Spec", "lstm_spec"),
    ]
    total = len(seeds) * len(methods_list)
    pbar = tqdm(total=total, desc=f"{system}", unit="run", leave=False)
    for seed in seeds:
        for method_name, loss_type in methods_list:
            pbar.set_description(f"{system} {method_name}")
            model = train_kalman(
                tensors_signal, train_idx_s, val_idx_s,
                loss_type=loss_type, seed=seed, config=config, device=device,
            )

            probs_test = build_probs(model, tensors_signal, test_idx_s)
            probs_null = build_probs(model, tensors_null, test_idx_n)
            probs_val = build_probs(model, tensors_signal, val_idx_s)

            thresh = select_threshold(
                probs_val,
                arrays_signal["bifurcation_times"][val_idx_s],
                arrays_signal["is_positive"][val_idx_s],
                arrays_signal["seq_lengths"][val_idx_s],
            )

            dt = compute_detection_time(
                probs_test,
                arrays_signal["bifurcation_times"][test_idx_s],
                arrays_signal["is_positive"][test_idx_s],
                arrays_signal["seq_lengths"][test_idx_s],
                thresh,
            )
            ewa = compute_early_warning_auc(
                probs_test,
                arrays_signal["bifurcation_times"][test_idx_s],
                arrays_signal["is_positive"][test_idx_s],
                arrays_signal["seq_lengths"][test_idx_s],
                probs_null,
                arrays_null["seq_lengths"][test_idx_n],
            )
            null_met = compute_null_metrics(probs_null, thresh, arrays_null["seq_lengths"][test_idx_n])

            metrics = {"detection_time": dt, "ew_auc": ewa, **null_met}
            runs.append(RunResult(method=method_name, seed=seed, metrics=metrics))
            writer.write_result_row({"system": system, "seed": seed, "method": method_name, **metrics})
            pbar.update(1)
    pbar.close()

    return SystemResult(system=system, runs=runs)


def _verdict_system(agg: Dict[str, Dict[str, float]], system: str) -> Tuple[bool, str]:
    reasons = []

    dt_bce = _mean_metric(agg, "Kalman-BCE", "detection_time")
    dt_lstm = _mean_metric(agg, "Kalman-LSTM", "detection_time")
    dt_spec = _mean_metric(agg, "Kalman-LSTM-Spec", "detection_time")
    dt_raw = _mean_metric(agg, "Raw-CSD", "detection_time")

    ewa_bce = _mean_metric(agg, "Kalman-BCE", "ew_auc")
    ewa_lstm = _mean_metric(agg, "Kalman-LSTM", "ew_auc")

    fpr_lstm = _mean_metric(agg, "Kalman-LSTM", "fpr")
    fpr_bce = _mean_metric(agg, "Kalman-BCE", "fpr")

    dt_gain = dt_bce - dt_lstm if (np.isfinite(dt_lstm) and np.isfinite(dt_bce)) else float("nan")
    ewa_gain = ewa_lstm - ewa_bce if (np.isfinite(ewa_lstm) and np.isfinite(ewa_bce)) else float("nan")

    def safe(v: float) -> str:
        return f"{v:.3f}" if np.isfinite(v) else "nan"
    reasons.append(f"  Raw-CSD detection time:            {safe(dt_raw)}")
    reasons.append(f"  Kalman-BCE detection time:         {safe(dt_bce)}")
    reasons.append(f"  Kalman-LSTM detection time:        {safe(dt_lstm)}")
    reasons.append(f"  Kalman-LSTM-Spec detection time:   {safe(dt_spec)}")
    reasons.append(f"  DT gain (LSTM vs BCE):             {safe(dt_gain)}")
    reasons.append(f"  EW-AUC gain (LSTM vs BCE):         {safe(ewa_gain)}")
    reasons.append(f"  FPR ratio (LSTM/BCE null):         {safe(fpr_lstm / max(fpr_bce, 1e-8))}")

    passed_dt = np.isfinite(dt_lstm) and dt_lstm >= 15.0
    
    if system == "chick_heart":
        passed_ewa = np.isfinite(ewa_lstm) and ewa_lstm >= 0.70
    else:
        passed_ewa = np.isfinite(ewa_gain) and ewa_gain >= 0.05
        
    passed_null = not (np.isfinite(fpr_lstm) and np.isfinite(fpr_bce) and fpr_lstm > 1.5 * fpr_bce + 0.05)

    reasons.append(f"  DT PASS: LSTM DT={safe(dt_lstm)} >= 15.0" if passed_dt else f"  DT FAIL: LSTM DT={safe(dt_lstm)} < 15.0")
    
    if system == "chick_heart":
        reasons.append(f"  EW-AUC PASS: LSTM AUC={safe(ewa_lstm)} >= 0.70" if passed_ewa else f"  EW-AUC FAIL: LSTM AUC={safe(ewa_lstm)} < 0.70")
    else:
        reasons.append(f"  EW-AUC PASS: gain={safe(ewa_gain)} >= 0.05" if passed_ewa else f"  EW-AUC FAIL: gain={safe(ewa_gain)}")
        
    reasons.append(f"  NULL PASS: FPR ratio={safe(fpr_lstm / max(fpr_bce, 1e-8))}" if passed_null else f"  NULL FAIL: FPR ratio={safe(fpr_lstm / max(fpr_bce, 1e-8))}")

    return passed_dt and passed_ewa and passed_null, "\n".join(reasons)


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


def _parse_args() -> Tuple[List[str], Optional[int]]:
    n_seeds_override: Optional[int] = None
    run_names: List[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("n_seeds="):
            n_seeds_override = int(arg.split("=", 1)[1])
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            run_names.append(arg)
    if not run_names:
        run_names = ["default", "high_noise", "low_data"]
    return run_names, n_seeds_override


def _run_single(run_name: str, n_seeds_override: Optional[int] = None) -> None:
    print(f"Loading config: {run_name}")
    config = load_config(run_name)
    data_cfg = config.get("data", {})
    if n_seeds_override is not None:
        data_cfg["n_seeds"] = n_seeds_override
        print(f"  [override] n_seeds={n_seeds_override}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"Settings: noise_scale={data_cfg.get('noise_scale')}, "
          f"n_patients={data_cfg.get('n_patients')}, "
          f"epochs={config.get('training', {}).get('epochs')}")
    print()

    writer = OutputWriter(experiment_name=f"benchmark/{run_name}")
    writer.write_config(config)
    print(f"Output: {writer.path}\n")

    systems = data_cfg.get("systems", SYSTEMS)
    max_length = data_cfg.get("max_length", 200)
    n_patients = data_cfg.get("n_patients", 500)
    noise_scale = data_cfg.get("noise_scale", 0.15)
    obs_noise_scale = data_cfg.get("obs_noise_scale")
    seed_offset = data_cfg.get("seed_offset", 0)

    system_results: Dict[str, Tuple[bool, str]] = {}
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    started = time.time()

    # Lazy-register real dataset builders
    if "chick_heart" in systems and "chick_heart" not in _REAL_SYSTEMS:
        from csd_observer.data.chick_heart import build_benchmark_datasets as _ch
        _REAL_SYSTEMS["chick_heart"] = _ch

    for system in systems:
        if system in _REAL_SYSTEMS:
            print(f"--- Loading {system} dataset (Empirical Pipeline) ---")
            arrays_signal, arrays_null = _REAL_SYSTEMS[system]()
            
            print(f"  signal: {arrays_signal['features'].shape}, null: {arrays_null['features'].shape}")
            
            sys_res = _run_empirical_experiment(
                system, arrays_signal, arrays_null,
                config=config, device=device, writer=writer,
            )
        else:
            print(f"--- Generating {system} data (Synthetic Pipeline) ---")
            data_kwargs = dict(
                n_trajectories=n_patients, noise_scale=noise_scale,
                obs_noise_scale=obs_noise_scale, max_length=max_length,
            )
            arrays_signal = _build_dataset_for_system(
                system, null=False, seed=seed_offset + 101, **data_kwargs,
            )
            arrays_null = _build_dataset_for_system(
                system, null=True, seed=seed_offset + 202, **data_kwargs,
            )
            
            print(f"  signal: {arrays_signal['features'].shape}, null: {arrays_null['features'].shape}")
            
            tensors_signal = tensorize(arrays_signal, device)
            tensors_null = tensorize(arrays_null, device)
            
            sys_res = _run_synthetic_experiment(
                system, tensors_signal, tensors_null,
                arrays_signal, arrays_null,
                config=config, device=device, writer=writer,
            )

        agg = sys_res.aggregate()
        all_metrics[system] = agg
        _summarize_system(agg, system)

        sys_pass, sys_reason = _verdict_system(agg, system)
        system_results[system] = (sys_pass, sys_reason)

    writer.write_metrics(all_metrics)

    elapsed = time.time() - started
    print(f"\nTime: {elapsed:.1f}s")
    print("\nSystem Verdicts:")
    for sys_name, (passed, _) in system_results.items():
        print(f"  {sys_name.capitalize():>12s}: {'PASS' if passed else 'FAIL'}")

    n_pass = sum(1 for p, _ in system_results.values() if p)
    n_total = len(system_results)
    if n_pass == n_total or n_pass >= 2:
        print(f"\nVERDICT: GO ({n_pass}/{n_total})")
    else:
        print(f"\nVERDICT: NO-GO ({n_pass}/{n_total})")

    print(f"\nAll results saved to: {writer.path}")


def main() -> None:
    run_names, n_seeds_override = _parse_args()
    total_started = time.time()
    for i, run_name in enumerate(run_names, 1):
        tag = f"[{i}/{len(run_names)}] " if len(run_names) > 1 else ""
        print(f"\n{tag}{'='*70}")
        print(f"{tag}RUN: {run_name}")
        print(f"{tag}{'='*70}")
        try:
            _run_single(run_name, n_seeds_override)
        except Exception as e:
            print(f"\nERROR: {run_name} failed: {e}")
            continue
    total_elapsed = time.time() - total_started
    print(f"\n{'='*70}")
    print(f"All runs complete. Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
