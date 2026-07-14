from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

W_LABEL = 60


def raw_csd_indicator(features: np.ndarray, seq_lengths: np.ndarray, window_size: int = 30) -> np.ndarray:
    B, T, C = features.shape
    W = min(window_size, T)
    scores = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        L = int(seq_lengths[b])
        for c in range(C):
            seq = features[b, :, c]
            for t in range(W, L):
                seg = seq[t - W : t]
                seg_c = seg - seg.mean()
                num = np.sum(seg_c[:-1] * seg_c[1:])
                denom = np.sum(seg_c ** 2) + 1e-8
                rho = num / denom
                scores[b, t] = max(scores[b, t], rho)
    return scores


def compute_bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, float]:
    if len(set(labels)) < 2:
        return {"mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(42)
    aucs = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(set(labels[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(labels[idx], scores[idx]))
        except Exception:
            continue
    if not aucs:
        return {"mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    auc_arr = np.array(aucs)
    return {
        "mean": float(np.mean(auc_arr)),
        "std": float(np.std(auc_arr)),
        "ci95_low": float(np.percentile(auc_arr, 2.5)),
        "ci95_high": float(np.percentile(auc_arr, 97.5)),
    }


def evaluate_raw_var(
    var_scores_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    var_scores_null: np.ndarray,
    seq_lens_null: np.ndarray,
    threshold: float = 0.0,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> Dict[str, float]:
    detection_times: List[float] = []
    early_probs: List[float] = []
    early_labels: List[int] = []
    for i in range(len(var_scores_signal)):
        tau = bif_times_signal[i]
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0:
            per_sample_thresh = np.percentile(var_scores_signal[i, :int(tau)], 80)
            alert_idx = np.where(var_scores_signal[i, :int(tau)] >= per_sample_thresh)[0]
            if len(alert_idx) > 0:
                detection_times.append(tau - alert_idx[0])
            ep_start = max(0, int(tau - early_start_delta))
            ep_end = max(0, int(tau - early_end_delta))
            if ep_end > ep_start:
                early_probs.append(float(np.max(var_scores_signal[i, ep_start:ep_end])))
                early_labels.append(1)
    for i in range(len(var_scores_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            ep_start = max(0, int(T - early_start_delta))
            ep_end = max(0, int(T - early_end_delta))
            if ep_end > ep_start:
                early_probs.append(float(np.max(var_scores_null[i, ep_start:ep_end])))
                early_labels.append(0)
    dt = float(np.mean(detection_times)) if detection_times else float("nan")
    ewa = float(roc_auc_score(early_labels, early_probs)) if len(set(early_labels)) >= 2 else float("nan")
    return {"detection_time": dt, "ew_auc": ewa}


def evaluate_raw_csd(
    csd_scores_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    csd_scores_null: np.ndarray,
    seq_lens_null: np.ndarray,
    threshold: float = 0.6,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> Dict[str, float]:
    detection_times: List[float] = []
    early_probs: List[float] = []
    early_labels: List[int] = []

    for i in range(len(csd_scores_signal)):
        tau = bif_times_signal[i]
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0:
            alert_idx = np.where(csd_scores_signal[i, :int(tau)] >= threshold)[0]
            if len(alert_idx) > 0:
                detection_times.append(tau - alert_idx[0])
            ep_start = max(0, int(tau - early_start_delta))
            ep_end = max(0, int(tau - early_end_delta))
            if ep_end > ep_start:
                early_probs.append(float(np.max(csd_scores_signal[i, ep_start:ep_end])))
                early_labels.append(1)

    for i in range(len(csd_scores_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            ep_start = max(0, int(T - early_start_delta))
            ep_end = max(0, int(T - early_end_delta))
            if ep_end > ep_start:
                early_probs.append(float(np.max(csd_scores_null[i, ep_start:ep_end])))
                early_labels.append(0)

    dt = float(np.mean(detection_times)) if detection_times else float("nan")
    ewa = float(roc_auc_score(early_labels, early_probs)) if len(set(early_labels)) >= 2 else float("nan")
    return {"detection_time": dt, "ew_auc": ewa}


def select_threshold(
    val_probs: np.ndarray,
    val_bif_times: np.ndarray,
    val_is_positive: np.ndarray,
    val_seq_lengths: np.ndarray,
    *,
    target_sensitivity: float = 0.80,
    null_probs: Optional[np.ndarray] = None,
    null_seq_lengths: Optional[np.ndarray] = None,
) -> float:
    positives = val_is_positive & (val_bif_times > 0)
    if not positives.any():
        return 0.5
    scores: List[float] = []
    labels: List[int] = []
    for i in np.where(positives)[0]:
        tau = val_bif_times[i]
        T = int(val_seq_lengths[i])
        window = max(0, int(tau - W_LABEL))
        for t in range(window, min(T, int(tau))):
            scores.append(val_probs[i, t])
            labels.append(1)
        far_window = max(0, int(tau - 2 * W_LABEL))
        for t in range(0, max(far_window - 1, 0)):
            scores.append(val_probs[i, t])
            labels.append(0)
    if null_probs is not None and null_seq_lengths is not None:
        for i in range(len(null_probs)):
            T = int(null_seq_lengths[i])
            if T > 0:
                scores.extend(float(null_probs[i, t]) for t in range(min(T, W_LABEL * 2)))
                labels.extend([0] * min(T, W_LABEL * 2))
    if len(set(labels)) < 2:
        return 0.5
    score_arr = np.array(scores)
    if np.std(score_arr) < 1e-6:
        return 0.5
    if np.ptp(score_arr) < 0.01:
        return float(np.median(score_arr))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    tpr_at_best = tpr[best_idx]
    if tpr_at_best >= target_sensitivity:
        return float(thresholds[best_idx])
    best_idx = np.argmin(np.abs(tpr - target_sensitivity))
    return float(thresholds[best_idx])


def compute_detection_time(
    probs: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> float:
    times: List[float] = []
    for i in range(len(probs)):
        if not is_positive[i]:
            continue
        tau = bifurcation_times[i]
        if tau <= 0:
            continue
        pre = probs[i, :int(tau)]
        alerts = np.where(pre >= threshold)[0]
        if len(alerts) > 0:
            times.append(tau - alerts[0])
    return float(np.mean(times)) if times else float("nan")


def compute_early_warning_auc(
    probs_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    probs_null: np.ndarray,
    seq_lens_null: np.ndarray,
    *,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> float:
    scores: List[float] = []
    labels: List[int] = []
    for i in range(len(probs_signal)):
        tau = bif_times_signal[i]
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0:
            t_start = max(0, int(tau - early_start_delta))
            t_end = max(0, int(tau - early_end_delta))
            window = probs_signal[i, t_start:t_end]
            if len(window) > 0:
                scores.append(float(np.max(window)))
                labels.append(1)
    for i in range(len(probs_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            t_start = max(0, int(T - early_start_delta))
            t_end = max(0, int(T - early_end_delta))
            window = probs_null[i, t_start:t_end]
            if len(window) > 0:
                scores.append(float(np.max(window)))
                labels.append(0)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def raw_var_indicator(features: np.ndarray, seq_lengths: np.ndarray, window_size: int = 30) -> np.ndarray:
    B, T, C = features.shape
    W = min(window_size, T)
    scores = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        L = int(seq_lengths[b])
        for c in range(C):
            seq = features[b, :, c]
            for t in range(W, L):
                seg = seq[t - W : t]
                scores[b, t] = max(scores[b, t], float(np.var(seg)))
    if W < T:
        scores[:, :W] = np.maximum(scores[:, :W], scores[:, [W]])
    return scores


def compute_false_positive_rate(
    probs: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> float:
    total_steps = 0
    alert_steps = 0
    for i in range(len(probs)):
        T = int(seq_lengths[i])
        total_steps += T
        alert_steps += int((probs[i, :T] >= threshold).sum())
    return alert_steps / max(total_steps, 1)


def compute_null_metrics(
    probs: np.ndarray,
    threshold: float,
    seq_lengths: np.ndarray,
) -> Dict[str, float]:
    return {"fpr": compute_false_positive_rate(probs, seq_lengths, threshold)}
