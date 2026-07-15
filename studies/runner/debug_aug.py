"""
Debug script: systematically test Kalman-LSTM-Aug configurations.

Tests multiple variations on a single system to isolate what causes high FPR.
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from csd_observer.data.bifurcation import build_dataset
from csd_observer.training.trainer import tensorize, train_kalman, build_probs
from csd_observer.utils.metrics import (
    compute_detection_time, compute_early_warning_auc,
    compute_false_positive_rate, select_threshold,
)
from csd_observer.config.load import load_config


def _compute_ews(features, seq_lengths, window_size=60):
    B, T, D = features.shape
    csd = np.zeros((B, T), dtype=np.float32)
    rvar = np.zeros((B, T), dtype=np.float32)
    lag2 = np.zeros((B, T), dtype=np.float32)
    alternans = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        L = int(seq_lengths[b])
        if L <= 0:
            continue
        seq = features[b, :L, 0]
        for t in range(min(window_size, L), L):
            w = seq[t - window_size : t]
            csd[b, t] = np.var(w)
            rvar[b, t] = np.std(w) / max(np.mean(np.abs(w)), 1e-8)
            if len(w) >= 3:
                lag2[b, t] = np.corrcoef(w[:-2], w[2:])[0, 1] if np.std(w[:-2]) > 1e-8 and np.std(w[2:]) > 1e-8 else 0.0
            alt_count = int(((w[2:] - w[:-2]) > 0).sum()) if len(w) > 2 else 0
            alternans[b, t] = alt_count / max(len(w) - 2, 1)
        if L > window_size:
            csd[b, :window_size] = csd[b, window_size]
            rvar[b, :window_size] = rvar[b, window_size]
            lag2[b, :window_size] = lag2[b, window_size]
            alternans[b, :window_size] = alternans[b, window_size]
    return csd, rvar, lag2, alternans


def run_config(system: str, config_overrides: dict, n_patients: int = 200, seed: int = 0):
    """Run Kalman-LSTM-Aug with given config overrides. Returns (dt, auc, fpr, thresh)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_config = load_config("default")

    data_cfg = main_config.get("data", {})
    noise_scale = data_cfg.get("noise_scale", 0.15)
    max_length = data_cfg.get("max_length", 200)
    seed_offset = data_cfg.get("seed_offset", 0)

    data_kwargs = dict(
        n_trajectories=n_patients, noise_scale=noise_scale,
        obs_noise_scale=None, max_length=max_length,
    )

    arrays_signal = build_dataset(system, null=False, seed=seed_offset + 101 + seed, **data_kwargs)
    arrays_null = build_dataset(system, null=True, seed=seed_offset + 202 + seed, **data_kwargs)

    # Apply feature modifications based on config
    feature_mask = config_overrides.get("feature_mask", slice(None))
    standardize = config_overrides.get("standardize", False)

    # Build augmented signal tensors
    arrays_signal_aug = {**arrays_signal, "augment_features": True}
    tensors_signal = tensorize(arrays_signal_aug, device)

    # Build augmented null tensors
    arrays_null_aug = {**arrays_null, "augment_features": True}
    tensors_null = tensorize(arrays_null_aug, device)

    # Apply feature masking if needed
    if feature_mask != slice(None):
        n_features = tensors_signal.features.shape[-1]
        orig_dim = arrays_signal["features"].shape[-1]
        full_mask = np.zeros(n_features, dtype=bool)
        full_mask[:orig_dim] = True
        if isinstance(feature_mask, int):
            full_mask[orig_dim + feature_mask] = True
        elif isinstance(feature_mask, (list, tuple, np.ndarray)):
            for f in feature_mask:
                full_mask[orig_dim + f] = True
        elif feature_mask == "all":
            full_mask[orig_dim:] = True
        tensors_signal.features = tensors_signal.features[:, :, full_mask]
        tensors_signal.masks = tensors_signal.masks[:, :, full_mask]
        tensors_null.features = tensors_null.features[:, :, full_mask]
        tensors_null.masks = tensors_null.masks[:, :, full_mask]

    # Standardization: compute stats from TRAINING null data
    if standardize:
        n_total_null = len(arrays_null["features"])
        null_train_idx = arrays_null["split_indices"]["train"]
        null_train = tensors_null.features[null_train_idx]

        # For each feature channel, compute mean/std over all non-padded steps
        feat_mean = []
        feat_std = []
        for c in range(null_train.shape[-1]):
            vals = []
            for b in range(len(null_train)):
                L = int(arrays_null["seq_lengths"][null_train_idx[b]])
                if L > 0:
                    vals.append(null_train[b, :L, c].cpu().numpy())
            if vals:
                all_vals = np.concatenate(vals)
                feat_mean.append(float(np.mean(all_vals)))
                feat_std.append(max(float(np.std(all_vals)), 1e-8))
            else:
                feat_mean.append(0.0)
                feat_std.append(1.0)

        # Apply standardization
        for name in ["features"]:
            t = getattr(tensors_signal, name)
            for c in range(t.shape[-1]):
                t[:, :, c] = (t[:, :, c] - feat_mean[c]) / feat_std[c]

            t = getattr(tensors_null, name)
            for c in range(t.shape[-1]):
                t[:, :, c] = (t[:, :, c] - feat_mean[c]) / feat_std[c]

    train_idx_s = arrays_signal["split_indices"]["train"]
    val_idx_s = arrays_signal["split_indices"]["val"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]

    model = train_kalman(
        tensors_signal, train_idx_s, val_idx_s,
        loss_type="lstm_spec", seed=seed, config=main_config, device=device,
    )

    probs_test = build_probs(model, tensors_signal, test_idx_s)
    probs_null = build_probs(model, tensors_null, test_idx_n)
    probs_val = build_probs(model, tensors_signal, val_idx_s)

    bif = arrays_signal["bifurcation_times"]
    is_pos = arrays_signal["is_positive"]
    seq_lens = arrays_signal["seq_lengths"]
    null_seq_lens = arrays_null["seq_lengths"]

    thresh = select_threshold(
        probs_val, bif[val_idx_s], is_pos[val_idx_s], seq_lens[val_idx_s],
    )
    dt = compute_detection_time(probs_test, bif[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], thresh)
    auc = compute_early_warning_auc(probs_test, bif[test_idx_s], is_pos[test_idx_s], seq_lens[test_idx_s], probs_null, null_seq_lens[test_idx_n])
    fpr = compute_false_positive_rate(probs_null, null_seq_lens[test_idx_n], thresh)

    # Score distribution
    null_scores = []
    for i in range(len(probs_null)):
        T = int(null_seq_lens[test_idx_n][i])
        if T > 0:
            null_scores.extend(probs_null[i, :T].tolist())

    sig_scores = []
    for i in range(len(probs_test)):
        if is_pos[test_idx_s][i]:
            tau = int(bif[test_idx_s][i])
            if tau > 0:
                pre = probs_test[i, :tau]
                if len(pre) > 0:
                    sig_scores.append(float(np.max(pre)))

    return {
        "dt": dt, "auc": auc, "fpr": fpr, "thresh": thresh,
        "null_p95": float(np.percentile(null_scores, 95)) if null_scores else 0,
        "null_max": float(np.max(null_scores)) if null_scores else 0,
        "sig_min": float(np.min(sig_scores)) if sig_scores else 0,
    }


def main():
    configs = [
        ("Baseline (no aug)", {"feature_mask": []}),
        ("Aug (csd only)", {"feature_mask": [0]}),
        ("Aug (rvar only)", {"feature_mask": [1]}),
        ("Aug (lag2 only)", {"feature_mask": [2]}),
        ("Aug (alternans only)", {"feature_mask": [3]}),
    ]

    n_patients = 200
    n_seeds = 2

    for system in ("fold", "logistic", "hopf"):
        print(f"\n{'=' * 80}")
        print(f"  System: {system.capitalize()} Bifurcation")
        print(f"{'=' * 80}")

        results = {name: {"dt": [], "auc": [], "fpr": []} for name, _ in configs}
        for name, cfg in configs:
            print(f"\n  --- {name} --- ", end="", flush=True)
            for seed in range(n_seeds):
                print(f"{seed}", end="", flush=True)
                try:
                    r = run_config(system, cfg, n_patients=n_patients, seed=seed)
                    results[name]["dt"].append(r["dt"])
                    results[name]["auc"].append(r["auc"])
                    results[name]["fpr"].append(r["fpr"])
                except Exception as e:
                    print(f"!", end="")
                    results[name]["dt"].append(float("nan"))
                    results[name]["auc"].append(float("nan"))
                    results[name]["fpr"].append(float("nan"))
            print()

        print(f"\n{'=' * 80}")
        print(f"{'Config':<30s} {'DT':>10s} {'EW-AUC':>10s} {'FPR':>10s}")
        print(f"{'-' * 60}")
        for name, _ in configs:
            dts = results[name]["dt"]
            aucs = results[name]["auc"]
            fprs = results[name]["fpr"]
            dt_m = np.nanmean(dts) if any(np.isfinite(dts)) else float("nan")
            auc_m = np.nanmean(aucs) if any(np.isfinite(aucs)) else float("nan")
            fpr_m = np.nanmean(fprs) if any(np.isfinite(fprs)) else float("nan")
            dt_s = f"{dt_m:.1f}" if np.isfinite(dt_m) else "nan"
            auc_s = f"{auc_m:.3f}" if np.isfinite(auc_m) else "nan"
            fpr_s = f"{fpr_m:.4f}" if np.isfinite(fpr_m) else "nan"
            print(f"{name:<30s} {dt_s:>10s} {auc_s:>10s} {fpr_s:>10s}")


if __name__ == "__main__":
    main()
