"""
Debug script: test threshold selection strategies for Kalman-LSTM-Aug.

Tests how different threshold configurations affect DT, EW-AUC, and FPR
for a SINGLE system with a SINGLE trained model.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from csd_observer.data.bifurcation import build_dataset
from csd_observer.training.trainer import tensorize, train_kalman, build_probs
from csd_observer.utils.metrics import (
    compute_detection_time,
    compute_early_warning_auc,
    compute_false_positive_rate,
    select_threshold,
)
from csd_observer.config.load import load_config


def run_system(system: str, n_patients: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("default")

    data_cfg = config.get("data", {})
    noise_scale = data_cfg.get("noise_scale", 0.15)
    max_length = data_cfg.get("max_length", 200)
    seed_offset = data_cfg.get("seed_offset", 0)

    print(f"System: {system}, n_patients={n_patients}, device={device}")
    print(f"noise_scale={noise_scale}, max_length={max_length}")

    data_kwargs = dict(
        n_trajectories=n_patients, noise_scale=noise_scale,
        obs_noise_scale=None, max_length=max_length,
    )

    arrays_signal = build_dataset(system, null=False, seed=seed_offset + 101, **data_kwargs)
    arrays_null = build_dataset(system, null=True, seed=seed_offset + 202, **data_kwargs)

    arrays_signal_aug = {**arrays_signal, "augment_features": True}
    arrays_null_aug = {**arrays_null, "augment_features": True}

    tensors_signal_aug = tensorize(arrays_signal_aug, device)
    tensors_null_aug = tensorize(arrays_null_aug, device)

    train_idx = arrays_signal["split_indices"]["train"]
    val_idx = arrays_signal["split_indices"]["val"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]

    print(f"  Signal: {arrays_signal['features'].shape}")
    print(f"  Null: {arrays_null['features'].shape}")
    print(f"  Aug signal shape: {tensors_signal_aug.features.shape}")
    print(f"  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx_s)}")

    print("\nTraining Kalman-LSTM-Aug...")
    model = train_kalman(
        tensors_signal_aug, train_idx, val_idx,
        loss_type="lstm_spec", seed=0, config=config, device=device,
    )

    probs_test = build_probs(model, tensors_signal_aug, test_idx_s)
    null_total_idx = np.arange(len(arrays_null["features"]))
    probs_null_full = build_probs(model, tensors_null_aug, null_total_idx)
    probs_val = build_probs(model, tensors_signal_aug, val_idx)

    bif_times = arrays_signal["bifurcation_times"]
    is_pos = arrays_signal["is_positive"]
    seq_lens = arrays_signal["seq_lengths"]

    n_null = len(arrays_null["features"])
    n_null_val = max(1, n_null // 5)
    null_val_idx = np.arange(n_null_val)
    null_test_idx = np.arange(n_null_val, n_null)

    probs_null_test = probs_null_full[null_test_idx]
    probs_null_val = probs_null_full[null_val_idx]
    null_seq_lens = arrays_null["seq_lengths"]

    strategies = []

    # Strategy 1: No null data (current Aug behavior)
    th1 = select_threshold(
        probs_val, bif_times[val_idx], is_pos[val_idx], seq_lens[val_idx],
    )
    dt1 = compute_detection_time(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], th1)
    ewa1 = compute_early_warning_auc(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], probs_null_test, null_seq_lens[null_test_idx])
    fpr1 = compute_false_positive_rate(probs_null_test, null_seq_lens[null_test_idx], th1)
    strategies.append(("Current (no null)", th1, dt1, ewa1, fpr1))

    # Strategy 2: With all null data
    th2 = select_threshold(
        probs_val, bif_times[val_idx], is_pos[val_idx], seq_lens[val_idx],
        null_probs=probs_null_full, null_seq_lengths=null_seq_lens,
    )
    dt2 = compute_detection_time(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], th2)
    ewa2 = compute_early_warning_auc(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], probs_null_test, null_seq_lens[null_test_idx])
    fpr2 = compute_false_positive_rate(probs_null_test, null_seq_lens[null_test_idx], th2)
    strategies.append(("With all null", th2, dt2, ewa2, fpr2))

    # Strategy 3: With null val split (no leakage to test null)
    th3 = select_threshold(
        probs_val, bif_times[val_idx], is_pos[val_idx], seq_lens[val_idx],
        null_probs=probs_null_val, null_seq_lengths=null_seq_lens[null_val_idx],
    )
    dt3 = compute_detection_time(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], th3)
    ewa3 = compute_early_warning_auc(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], probs_null_test, null_seq_lens[null_test_idx])
    fpr3 = compute_false_positive_rate(probs_null_test, null_seq_lens[null_test_idx], th3)
    strategies.append(("With null val split", th3, dt3, ewa3, fpr3))

    # Strategy 4: Fixed threshold at 0.5
    th4 = 0.5
    dt4 = compute_detection_time(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], th4)
    ewa4 = compute_early_warning_auc(probs_test, bif_times[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], probs_null_test, null_seq_lens[null_test_idx])
    fpr4 = compute_false_positive_rate(probs_null_test, null_seq_lens[null_test_idx], th4)
    strategies.append(("Fixed thresh=0.5", th4, dt4, ewa4, fpr4))

    print(f"\n{'=' * 70}")
    print(f"  Threshold Strategies for {system.capitalize()} (Kalman-LSTM-Aug)")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<25s} {'Thresh':>8s} {'DT':>10s} {'EW-AUC':>10s} {'FPR':>10s}")
    print(f"{'-' * 65}")
    for name, th, dt, ewa, fpr in strategies:
        th_s = f"{th:.4f}"
        dt_s = f"{dt:.1f}" if np.isfinite(dt) else "nan"
        ewa_s = f"{ewa:.3f}" if np.isfinite(ewa) else "nan"
        fpr_s = f"{fpr:.4f}" if np.isfinite(fpr) else "nan"
        print(f"{name:<25s} {th_s:>8s} {dt_s:>10s} {ewa_s:>10s} {fpr_s:>10s}")
    print()

    # Score distribution diagnostics
    sig_scores = []
    for i in range(len(probs_test)):
        if is_pos[test_idx_s][i] and bif_times[test_idx_s][i] > 0:
            tau = int(bif_times[test_idx_s][i])
            pre = probs_test[i, :tau]
            if len(pre) > 0:
                sig_scores.append(float(np.max(pre)))

    null_scores = []
    for i in range(len(probs_null_test)):
        T = int(null_seq_lens[null_test_idx][i])
        if T > 0:
            null_scores.extend(probs_null_test[i, :T].tolist())

    print(f"  Score Diagnostics:")
    if sig_scores:
        print(f"    Signal max scores:  min={np.min(sig_scores):.3f}, median={np.median(sig_scores):.3f}, max={np.max(sig_scores):.3f}")
    print(f"    Null scores:        min={np.min(null_scores):.3f}, median={np.median(null_scores):.3f}, max={np.max(null_scores):.3f}")
    print(f"    Null p90/p95/p99:   {np.percentile(null_scores, 90):.3f}/{np.percentile(null_scores, 95):.3f}/{np.percentile(null_scores, 99):.3f}")
    print()


if __name__ == "__main__":
    for system in ("fold", "hopf", "logistic"):
        run_system(system)
        print()
