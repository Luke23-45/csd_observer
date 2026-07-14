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

import copy
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
    build_probs_kalman_lag2,
    tensorize,
    train_kalman,
    train_kalman_lag2,
)
from csd_observer.models.kalman_lag2 import ClassicalKalmanLag2, grid_search_q  # noqa: E402
from csd_observer.utils.io import OutputWriter  # noqa: E402
from csd_observer.utils.metrics import (  # noqa: E402
    compute_detection_time,
    compute_early_warning_auc,
    compute_null_metrics,
    evaluate_raw_csd,
    evaluate_raw_lag2,
    evaluate_raw_var,
    raw_csd_indicator,
    raw_lag2_indicator,
    raw_lag2_indicator_detrended,
    raw_var_indicator,
    select_threshold,
)

SYSTEMS = ("fold", "hopf", "logistic")
METHODS = (
    "Raw-CSD", "RunningVar",
    "Lag2-CSD", "Lag2-CSD-detrended",
    "Kalman-Lag2", "Kalman-BCE",
    "Kalman-LSTM", "Kalman-LSTM-Spec", "Kalman-LSTM-Aux",
    "Kalman-Lag2-Net",
    "Kalman-ACKO",
)
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
    enabled_methods: Optional[set[str]] = None,
) -> SystemResult:
    B_sig = len(arrays_signal["is_positive"])
    features = np.concatenate([arrays_signal["features"], arrays_null["features"]], axis=0)
    seq_lengths = np.concatenate([arrays_signal["seq_lengths"], arrays_null["seq_lengths"]], axis=0)
    bifurcation_times = np.concatenate([arrays_signal["bifurcation_times"], arrays_null["bifurcation_times"]], axis=0)
    is_positive = np.concatenate([arrays_signal["is_positive"], arrays_null["is_positive"]], axis=0)

    all_idx = np.arange(len(is_positive))
    sig_idx = np.where(is_positive)[0]
    null_idx = np.where(~is_positive)[0]
    n_folds = 5
    cv_folds = []
    for k in range(n_folds):
        rng_fold = np.random.default_rng(42 + k * 101)
        sig_shuffled = sig_idx.copy()
        null_shuffled = null_idx.copy()
        rng_fold.shuffle(sig_shuffled)
        rng_fold.shuffle(null_shuffled)
        sig_test = sig_shuffled[k::n_folds]
        null_test = null_shuffled[k::n_folds]
        test_mask = np.zeros(len(is_positive), dtype=bool)
        test_mask[sig_test] = True
        test_mask[null_test] = True
        train_val_idx = np.where(~test_mask)[0]
        val_size = max(1, len(train_val_idx) // 5)
        rng_val = np.random.default_rng(k + 999)
        rng_val.shuffle(train_val_idx)
        train_idx = train_val_idx[val_size:]
        val_idx = train_val_idx[:val_size]
        cv_folds.append({"train": train_idx, "val": val_idx, "test": np.where(test_mask)[0]})

    merged_dataset = {
        "features": features,
        "seq_lengths": seq_lengths,
        "bifurcation_times": bifurcation_times,
        "is_positive": is_positive,
        "split_indices": cv_folds[0],
        "augment_features": arrays_signal.get("augment_features", False),
        "phase_features": arrays_signal.get("phase_features", False),
    }

    tensors_signal = tensorize(arrays_signal, device)
    tensors_null = tensorize(arrays_null, device)

    data_cfg = config.get("data", {})
    n_seeds = data_cfg.get("n_seeds", 5)
    seed_offset = data_cfg.get("seed_offset", 0)
    seeds = [seed_offset + 101 + 101 * i for i in range(n_seeds)]

    runs: List[RunResult] = []

    def _enabled(name: str) -> bool:
        return enabled_methods is None or name in enabled_methods

    def _enabled(name: str) -> bool:
        return enabled_methods is None or name in enabled_methods

    if _enabled("Raw-CSD"):
        csd_scores_test = raw_csd_indicator(arrays_signal["features"], arrays_signal["seq_lengths"], 30)
        csd_scores_null_test = raw_csd_indicator(arrays_null["features"], arrays_null["seq_lengths"], 30)
        raw_metrics = evaluate_raw_csd(
            csd_scores_test,
            arrays_signal["bifurcation_times"],
            arrays_signal["is_positive"],
            arrays_signal["seq_lengths"],
            csd_scores_null_test,
            arrays_null["seq_lengths"],
            threshold=0.6,
        )
        runs.append(RunResult(method="Raw-CSD", seed=0, metrics=raw_metrics))
        writer.write_result_row({"system": system, "seed": 0, "method": "Raw-CSD", **raw_metrics})

    if _enabled("RunningVar"):
        var_scores_test = raw_var_indicator(arrays_signal["features"], arrays_signal["seq_lengths"], 30)
        var_scores_null_test = raw_var_indicator(arrays_null["features"], arrays_null["seq_lengths"], 30)
        var_metrics = evaluate_raw_var(
            var_scores_test,
            arrays_signal["bifurcation_times"],
            arrays_signal["is_positive"],
            arrays_signal["seq_lengths"],
            var_scores_null_test,
            arrays_null["seq_lengths"],
        )
        runs.append(RunResult(method="RunningVar", seed=0, metrics=var_metrics))
        writer.write_result_row({"system": system, "seed": 0, "method": "RunningVar", **var_metrics})

    if _enabled("Lag2-CSD"):
        lag2_scores_test = raw_lag2_indicator(arrays_signal["features"], arrays_signal["seq_lengths"], 30)
        lag2_scores_null_test = raw_lag2_indicator(arrays_null["features"], arrays_null["seq_lengths"], 30)
        lag2_metrics = evaluate_raw_lag2(
            lag2_scores_test,
            arrays_signal["bifurcation_times"],
            arrays_signal["is_positive"],
            arrays_signal["seq_lengths"],
            lag2_scores_null_test,
            arrays_null["seq_lengths"],
            threshold=0.5,
        )
        runs.append(RunResult(method="Lag2-CSD", seed=0, metrics=lag2_metrics))
        writer.write_result_row({"system": system, "seed": 0, "method": "Lag2-CSD", **lag2_metrics})

    if _enabled("Lag2-CSD-detrended"):
        lag2_det_scores_test = raw_lag2_indicator_detrended(arrays_signal["features"], arrays_signal["seq_lengths"], 30)
        lag2_det_scores_null_test = raw_lag2_indicator_detrended(arrays_null["features"], arrays_null["seq_lengths"], 30)
        lag2_det_metrics = evaluate_raw_lag2(
            lag2_det_scores_test,
            arrays_signal["bifurcation_times"],
            arrays_signal["is_positive"],
            arrays_signal["seq_lengths"],
            lag2_det_scores_null_test,
            arrays_null["seq_lengths"],
            threshold=0.5,
        )
        runs.append(RunResult(method="Lag2-CSD-detrended", seed=0, metrics=lag2_det_metrics))
        writer.write_result_row({"system": system, "seed": 0, "method": "Lag2-CSD-detrended", **lag2_det_metrics})

    methods_list = [
        ("Kalman-BCE", "bce"),
        ("Kalman-LSTM", "lstm"),
        ("Kalman-LSTM-Spec", "lstm_spec"),
        ("Kalman-ACKO", "parity"),
    ]
    if system == "chick_heart":
        methods_list.append(("Kalman-LSTM-Aux", "lstm_aux"))

    methods_list = [(n, lt) for n, lt in methods_list if _enabled(n)]

    for method_name, loss_type in methods_list:
        for seed in seeds:
            cv_metrics = {"detection_time": [], "ew_auc": [], "fpr": []}
            for fold_idx, split in enumerate(cv_folds):
                merged_dataset_fold = dict(merged_dataset)
                merged_dataset_fold["split_indices"] = split
                tensors_merged = tensorize(merged_dataset_fold, device)

                train_idx = split["train"]
                val_idx = split["val"]
                test_idx = split["test"]
                test_mask_sig = test_idx < B_sig
                test_idx_s = test_idx[test_mask_sig]
                test_idx_n = test_idx[~test_mask_sig] - B_sig

                rng = np.random.default_rng(seed + fold_idx * 101)
                model = train_kalman(
                    tensors_merged, train_idx, val_idx,
                    loss_type=loss_type, seed=seed + fold_idx * 101,
                    config=config, device=device,
                )

                if len(test_idx_s) > 0:
                    probs_test = build_probs(model, tensors_signal, test_idx_s)
                else:
                    probs_test = np.zeros((0, tensors_merged.features.shape[1]))
                if len(test_idx_n) > 0:
                    probs_null = build_probs(model, tensors_null, test_idx_n)
                else:
                    probs_null = np.zeros((0, tensors_merged.features.shape[1]))

                probs_val_all = build_probs(model, tensors_merged, val_idx)
                val_null_mask = val_idx >= B_sig
                if val_null_mask.any():
                    null_val_probs = probs_val_all[val_null_mask]
                    null_val_lens = seq_lengths[val_idx[val_null_mask]]
                else:
                    null_val_probs = None
                    null_val_lens = None
                thresh = select_threshold(
                    probs_val_all,
                    bifurcation_times[val_idx],
                    is_positive[val_idx],
                    seq_lengths[val_idx],
                    null_probs=null_val_probs,
                    null_seq_lengths=null_val_lens,
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

                cv_metrics["detection_time"].append(dt)
                cv_metrics["ew_auc"].append(ewa)
                cv_metrics["fpr"].append(null_met.get("fpr", float("nan")))

            metrics = {
                "detection_time": float(np.nanmean(cv_metrics["detection_time"])),
                "ew_auc": float(np.nanmean(cv_metrics["ew_auc"])),
                "fpr": float(np.nanmean(cv_metrics["fpr"])),
                "detection_time_std": float(np.nanstd(cv_metrics["detection_time"])),
                "ew_auc_std": float(np.nanstd(cv_metrics["ew_auc"])),
                "fpr_std": float(np.nanstd(cv_metrics["fpr"])),
            }
            runs.append(RunResult(method=method_name, seed=seed, metrics=metrics))
            writer.write_result_row({"system": system, "seed": seed, "method": method_name, **metrics})

    # --- Kalman-Lag2 (classical, non-learned) ---
    lag2_det_all = np.concatenate([
        raw_lag2_indicator_detrended(arrays_signal["features"], arrays_signal["seq_lengths"], 30),
        raw_lag2_indicator_detrended(arrays_null["features"], arrays_null["seq_lengths"], 30),
    ], axis=0)
    if _enabled("Kalman-Lag2"):
        kalman_lag2_cv = {"detection_time": [], "ew_auc": [], "fpr": []}
        for fold_idx, split in enumerate(cv_folds):
            train_idx = split["train"]
            val_idx = split["val"]
            test_idx = split["test"]
            test_mask_sig = test_idx < B_sig
            test_idx_s = test_idx[test_mask_sig]
            test_idx_n = test_idx[~test_mask_sig] - B_sig

            lag2_val = lag2_det_all[val_idx]
            lag2_test_s = lag2_det_all[test_idx_s]
            lag2_test_n = lag2_det_all[test_idx_n]

            best_q = grid_search_q(
                lag2_val,
                bifurcation_times[val_idx],
                is_positive[val_idx],
                seq_lengths[val_idx],
            )

            kalman = ClassicalKalmanLag2(q=best_q, r=1.0).to(device)
            kalman.eval()
            with torch.no_grad():
                mu_val = kalman(torch.from_numpy(lag2_val.astype(np.float32)).to(device))["mu_hat"].cpu().numpy()
                mu_test_s = kalman(torch.from_numpy(lag2_test_s.astype(np.float32)).to(device))["mu_hat"].cpu().numpy()
                if len(test_idx_n) > 0:
                    mu_test_n = kalman(torch.from_numpy(lag2_test_n.astype(np.float32)).to(device))["mu_hat"].cpu().numpy()

            sig_mask_val = is_positive[val_idx] & (bifurcation_times[val_idx] > 0)
            null_mask_val = ~is_positive[val_idx]
            val_scores = []
            val_labels = []
            for i in np.where(sig_mask_val)[0]:
                tau = bifurcation_times[val_idx][i]
                t_start = max(0, int(tau - 50.0))
                t_end = max(0, int(tau - 5.0))
                if t_end > t_start:
                    val_scores.append(float(np.max(mu_val[i, t_start:t_end])))
                    val_labels.append(1)
            for i in np.where(null_mask_val)[0]:
                T_len = int(seq_lengths[val_idx][i])
                t_start = max(0, int(T_len - 50.0))
                t_end = max(0, int(T_len - 5.0))
                if t_end > t_start:
                    val_scores.append(float(np.max(mu_val[i, t_start:t_end])))
                    val_labels.append(0)

            if len(set(val_labels)) >= 2:
                from sklearn.metrics import roc_curve as _roc
                fpr_v, tpr_v, thr_v = _roc(val_labels, val_scores)
                youden = tpr_v - fpr_v
                best_thr = float(thr_v[int(np.argmax(youden))])
            else:
                best_thr = 0.5

            dt = compute_detection_time(mu_test_s, arrays_signal["bifurcation_times"][test_idx_s], arrays_signal["is_positive"][test_idx_s], arrays_signal["seq_lengths"][test_idx_s], best_thr)
            ewa = compute_early_warning_auc(mu_test_s, arrays_signal["bifurcation_times"][test_idx_s], arrays_signal["is_positive"][test_idx_s], arrays_signal["seq_lengths"][test_idx_s], mu_test_n if len(test_idx_n) > 0 else np.zeros((0, mu_test_s.shape[1])), arrays_null["seq_lengths"][test_idx_n] if len(test_idx_n) > 0 else np.zeros(0))
            null_m = compute_null_metrics(mu_test_n if len(test_idx_n) > 0 else np.zeros((0, mu_test_s.shape[1])), best_thr, arrays_null["seq_lengths"][test_idx_n] if len(test_idx_n) > 0 else np.zeros(0))
            kalman_lag2_cv["detection_time"].append(dt)
            kalman_lag2_cv["ew_auc"].append(ewa)
            kalman_lag2_cv["fpr"].append(null_m.get("fpr", float("nan")))

        kl2_metrics = {
            "detection_time": float(np.nanmean(kalman_lag2_cv["detection_time"])),
            "ew_auc": float(np.nanmean(kalman_lag2_cv["ew_auc"])),
            "fpr": float(np.nanmean(kalman_lag2_cv["fpr"])),
            "detection_time_std": float(np.nanstd(kalman_lag2_cv["detection_time"])),
            "ew_auc_std": float(np.nanstd(kalman_lag2_cv["ew_auc"])),
            "fpr_std": float(np.nanstd(kalman_lag2_cv["fpr"])),
        }
        runs.append(RunResult(method="Kalman-Lag2", seed=0, metrics=kl2_metrics))
        writer.write_result_row({"system": system, "seed": 0, "method": "Kalman-Lag2", **kl2_metrics})

    # --- Kalman-Lag2-Net (learned MLP head on top of Kalman) ---
    if _enabled("Kalman-Lag2-Net"):
        for seed in seeds:
            cv_metrics = {"detection_time": [], "ew_auc": [], "fpr": []}
            for fold_idx, split in enumerate(cv_folds):
                train_idx = split["train"]
                val_idx = split["val"]
                test_idx = split["test"]
                test_mask_sig = test_idx < B_sig
                test_idx_s = test_idx[test_mask_sig]
                test_idx_n = test_idx[~test_mask_sig] - B_sig

                lag2_train = lag2_det_all[train_idx]
                lag2_val = lag2_det_all[val_idx]
                lag2_test_s = lag2_det_all[test_idx_s]
                lag2_test_n = lag2_det_all[test_idx_n]

                best_q = grid_search_q(
                    lag2_val,
                    bifurcation_times[val_idx],
                    is_positive[val_idx],
                    seq_lengths[val_idx],
                )

                cfg_kl2 = copy.deepcopy(config)
                cfg_kl2.setdefault("training", {})["seed_override"] = seed
                model_kl2 = train_kalman_lag2(
                    lag2_det_all, train_idx, val_idx,
                    bifurcation_times, is_positive, seq_lengths,
                    q=best_q, config=cfg_kl2, device=device,
                )

                if len(test_idx_s) > 0:
                    probs_test = build_probs_kalman_lag2(model_kl2, lag2_det_all, test_idx_s)
                else:
                    probs_test = np.zeros((0, lag2_det_all.shape[1]))
                if len(test_idx_n) > 0:
                    probs_null = build_probs_kalman_lag2(model_kl2, lag2_det_all, test_idx_n)
                else:
                    probs_null = np.zeros((0, lag2_det_all.shape[1]))

                probs_val = build_probs_kalman_lag2(model_kl2, lag2_det_all, val_idx)
                val_null_mask = val_idx >= B_sig
                null_val_probs = probs_val[val_null_mask] if val_null_mask.any() else None
                null_val_lens = seq_lengths[val_idx[val_null_mask]] if val_null_mask.any() else None

                thresh = select_threshold(
                    probs_val, bifurcation_times[val_idx],
                    is_positive[val_idx], seq_lengths[val_idx],
                    null_probs=null_val_probs, null_seq_lengths=null_val_lens,
                )

                dt = compute_detection_time(
                    probs_test, arrays_signal["bifurcation_times"][test_idx_s],
                    arrays_signal["is_positive"][test_idx_s],
                    arrays_signal["seq_lengths"][test_idx_s], thresh,
                )
                ewa = compute_early_warning_auc(
                    probs_test, arrays_signal["bifurcation_times"][test_idx_s],
                    arrays_signal["is_positive"][test_idx_s],
                    arrays_signal["seq_lengths"][test_idx_s],
                    probs_null, arrays_null["seq_lengths"][test_idx_n],
                )
                null_m = compute_null_metrics(probs_null, thresh, arrays_null["seq_lengths"][test_idx_n])

                cv_metrics["detection_time"].append(dt)
                cv_metrics["ew_auc"].append(ewa)
                cv_metrics["fpr"].append(null_m.get("fpr", float("nan")))

            metrics = {
                "detection_time": float(np.nanmean(cv_metrics["detection_time"])),
                "ew_auc": float(np.nanmean(cv_metrics["ew_auc"])),
                "fpr": float(np.nanmean(cv_metrics["fpr"])),
                "detection_time_std": float(np.nanstd(cv_metrics["detection_time"])),
                "ew_auc_std": float(np.nanstd(cv_metrics["ew_auc"])),
                "fpr_std": float(np.nanstd(cv_metrics["fpr"])),
            }
            runs.append(RunResult(method="Kalman-Lag2-Net", seed=seed, metrics=metrics))
            writer.write_result_row({"system": system, "seed": seed, "method": "Kalman-Lag2-Net", **metrics})

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
    enabled_methods: Optional[set[str]] = None,
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

    def _enabled(name: str) -> bool:
        return enabled_methods is None or name in enabled_methods

    if _enabled("Raw-CSD"):
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

    if _enabled("RunningVar"):
        var_scores_test = raw_var_indicator(arrays_signal["features"][test_idx_s], arrays_signal["seq_lengths"][test_idx_s], 30)
        var_scores_null_test = raw_var_indicator(arrays_null["features"][test_idx_n], arrays_null["seq_lengths"][test_idx_n], 30)
        var_metrics = evaluate_raw_var(
            var_scores_test,
            arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            arrays_signal["seq_lengths"][test_idx_s],
            var_scores_null_test,
            arrays_null["seq_lengths"][test_idx_n],
        )
        runs.append(RunResult(method="RunningVar", seed=0, metrics=var_metrics))
        writer.write_result_row({"system": system, "seed": 0, "method": "RunningVar", **var_metrics})

    if _enabled("Lag2-CSD"):
        lag2_scores_test = raw_lag2_indicator(arrays_signal["features"][test_idx_s], arrays_signal["seq_lengths"][test_idx_s], 30)
        lag2_scores_null_test = raw_lag2_indicator(arrays_null["features"][test_idx_n], arrays_null["seq_lengths"][test_idx_n], 30)
        lag2_metrics = evaluate_raw_lag2(
            lag2_scores_test,
            arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            arrays_signal["seq_lengths"][test_idx_s],
            lag2_scores_null_test,
            arrays_null["seq_lengths"][test_idx_n],
            threshold=0.5,
        )
        runs.append(RunResult(method="Lag2-CSD", seed=0, metrics=lag2_metrics))
        writer.write_result_row({"system": system, "seed": 0, "method": "Lag2-CSD", **lag2_metrics})

    methods_list = [
        ("Kalman-BCE", "bce"),
        ("Kalman-LSTM", "lstm"),
        ("Kalman-LSTM-Spec", "lstm_spec"),
        ("Kalman-ACKO", "parity"),
    ]
    methods_list = [(n, lt) for n, lt in methods_list if _enabled(n)]
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
    dt_aux = _mean_metric(agg, "Kalman-LSTM-Aux", "detection_time")
    dt_spec = _mean_metric(agg, "Kalman-LSTM-Spec", "detection_time")
    dt_raw = _mean_metric(agg, "Raw-CSD", "detection_time")
    dt_var = _mean_metric(agg, "RunningVar", "detection_time")
    dt_lag2 = _mean_metric(agg, "Lag2-CSD", "detection_time")

    ewa_bce = _mean_metric(agg, "Kalman-BCE", "ew_auc")
    ewa_lstm = _mean_metric(agg, "Kalman-LSTM", "ew_auc")
    ewa_aux = _mean_metric(agg, "Kalman-LSTM-Aux", "ew_auc")
    ewa_raw = _mean_metric(agg, "Raw-CSD", "ew_auc")
    ewa_var = _mean_metric(agg, "RunningVar", "ew_auc")
    ewa_lag2 = _mean_metric(agg, "Lag2-CSD", "ew_auc")

    fpr_lstm = _mean_metric(agg, "Kalman-LSTM", "fpr")
    fpr_aux = _mean_metric(agg, "Kalman-LSTM-Aux", "fpr")
    fpr_bce = _mean_metric(agg, "Kalman-BCE", "fpr")
    fpr_lag2 = _mean_metric(agg, "Lag2-CSD", "fpr")

    primary_method = "Kalman-LSTM-Aux" if system == "chick_heart" and np.isfinite(dt_aux) else "Kalman-LSTM"
    dt_primary = dt_aux if primary_method == "Kalman-LSTM-Aux" else dt_lstm
    ewa_primary = ewa_aux if primary_method == "Kalman-LSTM-Aux" else ewa_lstm
    fpr_primary = fpr_aux if primary_method == "Kalman-LSTM-Aux" else fpr_lstm

    dt_gain = dt_bce - dt_primary if (np.isfinite(dt_primary) and np.isfinite(dt_bce)) else float("nan")
    ewa_gain = ewa_primary - ewa_bce if (np.isfinite(ewa_primary) and np.isfinite(ewa_bce)) else float("nan")

    def safe(v: float) -> str:
        return f"{v:.3f}" if np.isfinite(v) else "nan"
    reasons.append(f"  Raw-CSD detection time:            {safe(dt_raw)}")
    reasons.append(f"  RunningVar detection time:         {safe(dt_var)}")
    reasons.append(f"  Lag2-CSD detection time:           {safe(dt_lag2)}")
    reasons.append(f"  Kalman-BCE detection time:         {safe(dt_bce)}")
    reasons.append(f"  Kalman-LSTM detection time:        {safe(dt_lstm)}")
    reasons.append(f"  Kalman-LSTM-Aux detection time:    {safe(dt_aux)}")
    reasons.append(f"  Kalman-LSTM-Spec detection time:   {safe(dt_spec)}")
    reasons.append(f"  Raw-CSD EW-AUC:                    {safe(ewa_raw)}")
    reasons.append(f"  RunningVar EW-AUC:                 {safe(ewa_var)}")
    reasons.append(f"  Lag2-CSD EW-AUC:                   {safe(ewa_lag2)}")
    reasons.append(f"  DT gain (primary vs BCE):          {safe(dt_gain)}")
    reasons.append(f"  EW-AUC gain (primary vs BCE):      {safe(ewa_gain)}")
    reasons.append(f"  FPR ratio (primary/BCE null):      {safe(fpr_primary / max(fpr_bce, 1e-8))}")

    passed_dt = np.isfinite(dt_gain) and dt_gain >= 15.0
    passed_ewa = np.isfinite(ewa_gain) and ewa_gain >= 0.05
    passed_null = not (np.isfinite(fpr_primary) and np.isfinite(fpr_bce) and fpr_primary > 1.5 * fpr_bce + 0.05)

    reasons.append(f"  DT PASS: gain={safe(dt_gain)} >= 15.0" if passed_dt else f"  DT FAIL: gain={safe(dt_gain)} < 15.0")
    reasons.append(f"  EW-AUC PASS: gain={safe(ewa_gain)} >= 0.05" if passed_ewa else f"  EW-AUC FAIL: gain={safe(ewa_gain)}")
    reasons.append(f"  NULL PASS: FPR ratio={safe(fpr_primary / max(fpr_bce, 1e-8))}" if passed_null else f"  NULL FAIL: FPR ratio={safe(fpr_primary / max(fpr_bce, 1e-8))}")

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


def _parse_args() -> Tuple[List[str], Optional[int], Optional[set[str]]]:
    n_seeds_override: Optional[int] = None
    methods_override: Optional[set[str]] = None
    run_names: List[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("n_seeds="):
            n_seeds_override = int(arg.split("=", 1)[1])
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
        run_names = ["default", "high_noise", "low_data"]
    return run_names, n_seeds_override, methods_override


def _run_single(run_name: str, n_seeds_override: Optional[int] = None, enabled_methods: Optional[set[str]] = None) -> None:
    print(f"Loading config: {run_name}")
    config = load_config(run_name)
    data_cfg = config.get("data", {})
    if n_seeds_override is not None:
        data_cfg["n_seeds"] = n_seeds_override
        print(f"  [override] n_seeds={n_seeds_override}")
    if enabled_methods is not None:
        print(f"  [override] methods={','.join(sorted(enabled_methods))}")

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
                enabled_methods=enabled_methods,
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
                enabled_methods=enabled_methods,
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
    run_names, n_seeds_override, methods_override = _parse_args()
    total_started = time.time()
    for i, run_name in enumerate(run_names, 1):
        tag = f"[{i}/{len(run_names)}] " if len(run_names) > 1 else ""
        print(f"\n{tag}{'='*70}")
        print(f"{tag}RUN: {run_name}")
        print(f"{tag}{'='*70}")
        try:
            _run_single(run_name, n_seeds_override, methods_override)
        except Exception as e:
            print(f"\nERROR: {run_name} failed: {e}")
            continue
    total_elapsed = time.time() - total_started
    print(f"\n{'='*70}")
    print(f"All runs complete. Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
