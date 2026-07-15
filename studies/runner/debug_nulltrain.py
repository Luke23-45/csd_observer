"""
Debug script: test H4 — train Kalman-LSTM-Spec with null data.

Compares baseline (no null training) vs null-trained on Hopf.
If successful, the null-trained model should have lower FPR
without sacrificing DT or AUC.
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from csd_observer.data.bifurcation import build_dataset
from csd_observer.training.trainer import tensorize, train_kalman, build_probs, TensorizedDataset
from csd_observer.utils.metrics import (
    compute_detection_time, compute_early_warning_auc,
    compute_false_positive_rate, select_threshold,
)
from csd_observer.config.load import load_config


VERY_LARGE = 100000

def run_baseline(system: str, n_patients: int = 200, seed: int = 0):
    """Baseline LSTM-Spec — trained on signal only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("default")
    data_cfg = config.get("data", {})
    noise_scale = data_cfg.get("noise_scale", 0.15)
    max_length = data_cfg.get("max_length", 200)
    seed_offset = data_cfg.get("seed_offset", 0)

    kwargs = dict(n_trajectories=n_patients, noise_scale=noise_scale,
                  obs_noise_scale=None, max_length=max_length)
    arrays_signal = build_dataset(system, null=False, seed=seed_offset + 101 + seed, **kwargs)
    arrays_null = build_dataset(system, null=True, seed=seed_offset + 202 + seed, **kwargs)

    tensors = tensorize(arrays_signal, device)
    train_idx = arrays_signal["split_indices"]["train"]
    val_idx = arrays_signal["split_indices"]["val"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]

    model = train_kalman(tensors, train_idx, val_idx,
                         loss_type="lstm_spec", seed=seed, config=config, device=device)

    probs_test = build_probs(model, tensors, test_idx_s)
    probs_null = build_probs(model, tensorize(arrays_null, device), test_idx_n)
    probs_val = build_probs(model, tensors, val_idx)

    bif = arrays_signal["bifurcation_times"]
    is_pos = arrays_signal["is_positive"]
    seq_lens = arrays_signal["seq_lengths"]
    null_seq_lens = arrays_null["seq_lengths"]

    th = select_threshold(probs_val, bif[val_idx], is_pos[val_idx], seq_lens[val_idx])
    dt = compute_detection_time(probs_test, bif[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], th)
    auc = compute_early_warning_auc(probs_test, bif[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], probs_null, null_seq_lens[test_idx_n])
    fpr = compute_false_positive_rate(probs_null, null_seq_lens[test_idx_n], th)
    return {"dt": dt, "auc": auc, "fpr": fpr, "thresh": th}


def run_null_trained(system: str, n_patients: int = 200, seed: int = 0, null_ratio: float = 1.0):
    """Null-trained LSTM-Spec — signal + null in training set.

    null_ratio controls how many null trajectories to include:
    null_ratio = 1.0 means equal null and signal trajectories.
    null_ratio = 0.5 means half as many null as signal.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("default")
    data_cfg = config.get("data", {})
    noise_scale = data_cfg.get("noise_scale", 0.15)
    max_length = data_cfg.get("max_length", 200)
    seed_offset = data_cfg.get("seed_offset", 0)

    kwargs = dict(n_trajectories=n_patients, noise_scale=noise_scale,
                  obs_noise_scale=None, max_length=max_length)
    arrays_signal = build_dataset(system, null=False, seed=seed_offset + 101 + seed, **kwargs)
    arrays_null = build_dataset(system, null=True, seed=seed_offset + 202 + seed, **kwargs)

    tensors_signal = tensorize(arrays_signal, device)
    tensors_null = tensorize(arrays_null, device)

    # Fix seed for reproducible null subset selection
    rng = np.random.RandomState(seed)

    n_sig = tensors_signal.features.shape[0]
    n_null = tensors_null.features.shape[0]

    train_idx_s = arrays_signal["split_indices"]["train"]
    val_idx_s = arrays_signal["split_indices"]["val"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]
    train_idx_n_all = arrays_null["split_indices"]["train"]
    val_idx_n = arrays_null["split_indices"]["val"]

    # Subset null train indices based on null_ratio
    n_null_train = max(1, int(len(train_idx_s) * null_ratio))
    n_null_train = min(n_null_train, len(train_idx_n_all))
    train_idx_n = rng.choice(train_idx_n_all, size=n_null_train, replace=False)

    n_sig_total = n_sig
    n_null_used = len(train_idx_n) + len(val_idx_n)

    features = torch.cat([tensors_signal.features, tensors_null.features], dim=0)
    masks = torch.cat([tensors_signal.masks, tensors_null.masks], dim=0)
    seq_lengths = torch.cat([tensors_signal.seq_lengths, tensors_null.seq_lengths])

    bif_times = torch.cat([
        tensors_signal.bifurcation_times,
        torch.full((n_null,), VERY_LARGE, dtype=torch.float32, device=device),
    ])
    is_positive = torch.cat([
        tensors_signal.is_positive,
        torch.zeros(n_null, dtype=torch.bool, device=device),
    ])

    combined = TensorizedDataset(features, masks, seq_lengths, bif_times, is_positive)

    train_idx = np.concatenate([train_idx_s, train_idx_n + n_sig_total])
    val_idx = np.concatenate([val_idx_s, val_idx_n + n_sig_total])

    model = train_kalman(combined, train_idx, val_idx,
                         loss_type="lstm_spec", seed=seed, config=config, device=device)

    probs_test = build_probs(model, combined, test_idx_s)
    probs_null = build_probs(model, combined, test_idx_n + n_sig_total)
    probs_val = build_probs(model, combined, val_idx_s)

    bif = arrays_signal["bifurcation_times"]
    is_pos = arrays_signal["is_positive"]
    seq_lens = arrays_signal["seq_lengths"]
    null_seq_lens = arrays_null["seq_lengths"]

    th = select_threshold(probs_val, bif[val_idx_s], is_pos[val_idx_s], seq_lens[val_idx_s])
    dt = compute_detection_time(probs_test, bif[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], th)
    auc = compute_early_warning_auc(probs_test, bif[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], probs_null, null_seq_lens[test_idx_n])
    fpr = compute_false_positive_rate(probs_null, null_seq_lens[test_idx_n], th)
    return {"dt": dt, "auc": auc, "fpr": fpr, "thresh": th, "n_null_used": n_null_used}


def main():
    n_patients = 200
    n_seeds = 2
    target_ratio = 0.50  # best ratio from Hopf sweep

    for system in ("fold", "logistic", "hopf"):
        print(f"\n{'=' * 90}")
        print(f"  System: {system.capitalize()} — null_ratio={target_ratio}")
        print(f"{'=' * 90}")

        bl_results = {"dt": [], "auc": [], "fpr": []}
        nt_results = {"dt": [], "auc": [], "fpr": []}

        for seed in range(n_seeds):
            print(f"\n  Seed {seed}: ")
            bl = run_baseline(system, n_patients=n_patients, seed=seed)
            bl_results["dt"].append(bl["dt"])
            bl_results["auc"].append(bl["auc"])
            bl_results["fpr"].append(bl["fpr"])
            print(f"    Baseline:     DT={bl['dt']:.1f} AUC={bl['auc']:.3f} FPR={bl['fpr']:.4f}")

            nt = run_null_trained(system, n_patients=n_patients, seed=seed, null_ratio=target_ratio)
            nt_results["dt"].append(nt["dt"])
            nt_results["auc"].append(nt["auc"])
            nt_results["fpr"].append(nt["fpr"])
            print(f"    Null r={target_ratio:.2f}: DT={nt['dt']:.1f} AUC={nt['auc']:.3f} FPR={nt['fpr']:.4f}")

        def _row(name, m):
            dt_s = f"{np.nanmean(m['dt']):.1f}" if any(np.isfinite(m['dt'])) else "nan"
            auc_s = f"{np.nanmean(m['auc']):.3f}" if any(np.isfinite(m['auc'])) else "nan"
            fpr_s = f"{np.nanmean(m['fpr']):.4f}" if any(np.isfinite(m['fpr'])) else "nan"
            return f"{name:<20s} {dt_s:>10s} {auc_s:>10s} {fpr_s:>10s}"

        print(f"\n{'=' * 60}")
        print(f"{'Config':<20s} {'DT':>10s} {'EW-AUC':>10s} {'FPR':>10s}")
        print(f"{'-' * 60}")
        print(_row("Baseline", bl_results))
        print(_row(f"Null r={target_ratio:.2f}", nt_results))

        bl_fpr = np.nanmean(bl_results["fpr"])
        nt_fpr = np.nanmean(nt_results["fpr"])
        bl_dt = np.nanmean(bl_results["dt"])
        nt_dt = np.nanmean(nt_results["dt"])
        bl_auc = np.nanmean(bl_results["auc"])
        nt_auc = np.nanmean(nt_results["auc"])
        print(f"\n  ΔDT={nt_dt - bl_dt:+.1f}  ΔAUC={nt_auc - bl_auc:+.3f}  ΔFPR={nt_fpr - bl_fpr:+.4f}")


if __name__ == "__main__":
    main()
