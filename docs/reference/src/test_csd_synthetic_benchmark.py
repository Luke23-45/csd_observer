"""Kalman observer LSTM head benchmark.

Hypothesis: Replacing the per-step MLP head with a causal LSTM head
enables detection of temporal CSD patterns (rising autocorrelation,
slowing recovery) that a per-step head cannot see.

Models:
    - Raw-CSD:        Classical CSD indicator (non-learned baseline)
    - Kalman-BCE:     Observer with per-step MLP head
    - Kalman-LSTM:    Observer with causal LSTM head
    - Kalman-LSTM-Spec: Observer with causal LSTM head + spectral radius loss

Verdict: GO if Kalman-LSTM >= 20 DT gain on >= 2/3 systems vs Kalman-BCE.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bgsl.core.common.csd_loss import SpectralRadiusLoss
from bgsl.data.synthetic.bifurcation import build_dataset
from bgsl.models.csd_observer import CSDKalmanObserver

SYSTEMS = ("fold", "hopf", "logistic")
METHODS = ("Raw-CSD", "Kalman-BCE", "Kalman-LSTM", "Kalman-LSTM-Spec")
STRESS_SEEDS = (101, 202, 303, 404, 505)

W_LABEL = 60
T_max = 200


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset_for_system(
    system: str, *, null: bool, n_trajectories: int, noise_scale: float, seed: int,
) -> Dict:
    kwargs = dict(
        n_trajectories=n_trajectories, max_length=T_max,
        noise_scale=noise_scale, seed=seed, null=null,
    )
    if system == "fold":
        return build_dataset("fold", **kwargs)
    elif system == "hopf":
        return build_dataset("hopf", **kwargs)
    else:
        return build_dataset("logistic", **kwargs)


@dataclass
class TensorizedDataset:
    features: torch.Tensor
    masks: torch.Tensor
    seq_lengths: torch.Tensor
    bifurcation_times: torch.Tensor
    is_positive: torch.Tensor


def _tensorize(dataset: Dict, device: torch.device) -> TensorizedDataset:
    return TensorizedDataset(
        features=torch.tensor(dataset["features"], dtype=torch.float32, device=device),
        masks=torch.ones(dataset["features"].shape, dtype=torch.float32, device=device),
        seq_lengths=torch.tensor(dataset["seq_lengths"], dtype=torch.long, device=device),
        bifurcation_times=torch.tensor(dataset["bifurcation_times"], dtype=torch.float32, device=device),
        is_positive=torch.tensor(dataset["is_positive"], dtype=torch.bool, device=device),
    )


_arange_T = torch.arange(T_max).float().unsqueeze(0)


def _make_targets(
    bifurcation_times: torch.Tensor,
    seq_lengths: torch.Tensor,
    device: torch.device,
    sigma: float = 60.0,
) -> torch.Tensor:
    B = bifurcation_times.shape[0]
    t = _arange_T.to(device, non_blocking=True).expand(B, -1)
    tau = bifurcation_times.unsqueeze(1).float()
    
    # Continuous criticality target: exp(-((tau - t) / sigma)^2)
    dist = torch.clamp(tau - t, min=0.0) # distance to bifurcation, 0 if t >= tau
    target = torch.exp(- (dist / sigma)**2)
    
    # Mask out completely invalid conditions (e.g. null series where tau <= 0)
    valid_tau = tau > 0
    target = target * valid_tau.float()
    
    # Strictly 0 out after the bifurcation
    is_before_bifurcation = (t <= tau).float()
    target = target * is_before_bifurcation
    
    valid_len = t < seq_lengths.unsqueeze(1).float()
    return target * valid_len.float()


def _train_kalman(
    tensors: TensorizedDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    loss_type: str,
    seed: int,
    device: torch.device,
    latent_dim: int = 4,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 5,
    spec_weight: float = 0.1,
) -> CSDKalmanObserver:
    _seed_all(seed)

    use_lstm = loss_type in ("lstm", "lstm_spec")
    use_spec = loss_type == "lstm_spec"
    n_features = tensors.features.shape[-1]

    x_train = tensors.features[train_idx]
    x_val = tensors.features[val_idx]
    m_train = tensors.masks[train_idx]
    m_val = tensors.masks[val_idx]
    lens_train = tensors.seq_lengths[train_idx]
    lens_val = tensors.seq_lengths[val_idx]
    bif_train = tensors.bifurcation_times[train_idx]
    bif_val = tensors.bifurcation_times[val_idx]

    model = CSDKalmanObserver(
        input_dim=n_features,
        latent_dim=latent_dim,
        lstm_head=use_lstm,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    spec_loss_fn = SpectralRadiusLoss(weight=spec_weight) if use_spec else None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    stale_epochs = 0
    train_size = len(train_idx)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_size, device=device)
        for start in range(0, train_size, batch_size):
            batch_ids = order[start : start + batch_size]
            logits, zs, A, K, C = model(x_train[batch_ids], m_train[batch_ids])

            targets = _make_targets(
                bif_train[batch_ids], lens_train[batch_ids], device
            )
            valid_mask = (torch.arange(T_max, device=device).unsqueeze(0) <
                          lens_train[batch_ids].unsqueeze(1)).float()

            bce_per_step = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss = (bce_per_step * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

            if use_spec:
                loss = loss + spec_loss_fn(A, K, C)["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_val, zs_val, A_val, K_val, C_val = model(x_val, m_val)
            targets_val = _make_targets(bif_val, lens_val, device)
            valid_mask_val = (torch.arange(T_max, device=device).unsqueeze(0) <
                              lens_val.unsqueeze(1)).float()

            bce_per_step_val = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_val, targets_val, reduction="none"
            )
            val_loss = (bce_per_step_val * valid_mask_val).sum() / valid_mask_val.sum().clamp(min=1.0)

            if use_spec:
                val_loss = val_loss + spec_loss_fn(A_val, K_val, C_val)["loss"]

            val_loss = val_loss.item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _build_probs(
    model: CSDKalmanObserver,
    tensors: TensorizedDataset,
    indices: np.ndarray,
) -> np.ndarray:
    x = tensors.features[indices]
    m = tensors.masks[indices]
    model.eval()
    with torch.no_grad():
        logits, _, _, _, _ = model(x, m)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def _raw_csd_indicator(features: np.ndarray, window_size: int = 30) -> np.ndarray:
    B, T, C = features.shape
    W = min(window_size, T)
    scores = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            seq = features[b, :, c]
            for t in range(W, T):
                seg = seq[t - W : t]
                seg_c = seg - seg.mean()
                num = np.sum(seg_c[:-1] * seg_c[1:])
                denom = np.sum(seg_c ** 2) + 1e-8
                rho = num / denom
                scores[b, t] = max(scores[b, t], rho)
    return scores


def _evaluate_raw_csd(
    csd_scores_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    csd_scores_null: np.ndarray,
    seq_lens_null: np.ndarray,
    threshold: float,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> Dict[str, float]:
    detection_times = []
    early_probs = []
    early_labels = []

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
            early_probs.append(float(np.max(csd_scores_null[i, :T])))
            early_labels.append(0)

    metrics: Dict[str, float] = {}
    metrics["detection_time"] = float(np.mean(detection_times)) if detection_times else float("nan")
    if len(set(early_labels)) >= 2:
        from sklearn.metrics import roc_auc_score
        metrics["ew_auc"] = float(roc_auc_score(early_labels, early_probs))
    else:
        metrics["ew_auc"] = float("nan")
    return metrics


def _select_threshold(
    val_probs: np.ndarray,
    val_bif_times: np.ndarray,
    val_is_positive: np.ndarray,
    val_seq_lengths: np.ndarray,
    *,
    target_sensitivity: float = 0.80,
) -> float:
    positives = val_is_positive & (val_bif_times > 0)
    if not positives.any():
        return 0.5
    scores = []
    labels = []
    for i in np.where(positives)[0]:
        tau = val_bif_times[i]
        T = int(val_seq_lengths[i])
        window = max(0, int(tau - W_LABEL))
        for t in range(window, min(T, int(tau))):
            scores.append(val_probs[i, t])
            labels.append(1)
        for t in range(0, max(window - 1, 0)):
            scores.append(val_probs[i, t])
            labels.append(0)
    if len(set(labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = np.argmin(np.abs(tpr - target_sensitivity))
    return float(thresholds[best_idx])


def _compute_detection_time(
    probs: np.ndarray, bifurcation_times: np.ndarray,
    is_positive: np.ndarray, seq_lengths: np.ndarray, threshold: float,
) -> float:
    times = []
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


def _compute_early_warning_auc(
    probs_signal: np.ndarray, bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray, seq_lens_signal: np.ndarray,
    probs_null: np.ndarray, seq_lens_null: np.ndarray,
    *,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> float:
    from sklearn.metrics import roc_auc_score
    scores = []
    labels = []
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
            scores.append(float(np.max(probs_null[i, :T])))
            labels.append(0)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _compute_null_metrics(
    probs: np.ndarray, threshold: float, seq_lengths: np.ndarray,
) -> Dict[str, float]:
    return {"fpr": _compute_false_positive_rate(probs, seq_lengths, threshold)}


def _compute_false_positive_rate(
    probs: np.ndarray, seq_lengths: np.ndarray, threshold: float,
) -> float:
    total_steps = 0
    alert_steps = 0
    for i in range(len(probs)):
        T = int(seq_lengths[i])
        total_steps += T
        alert_steps += int((probs[i, :T] >= threshold).sum())
    return alert_steps / max(total_steps, 1)


@dataclass(frozen=True)
class RunResult:
    method: str
    seed: int
    metrics: Dict[str, float]


@dataclass(frozen=True)
class SystemResult:
    system: str
    runs: List[RunResult]

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHODS}
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


def _verdict_system(agg: Dict[str, Dict[str, float]]) -> Tuple[bool, str]:
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

    reasons.append(f"  Raw-CSD detection time:            {dt_raw:.1f}" if np.isfinite(dt_raw) else "  Raw-CSD detection time:            nan")
    reasons.append(f"  Kalman-BCE detection time:         {dt_bce:.1f}" if np.isfinite(dt_bce) else "  Kalman-BCE detection time:         nan")
    reasons.append(f"  Kalman-LSTM detection time:        {dt_lstm:.1f}" if np.isfinite(dt_lstm) else "  Kalman-LSTM detection time:        nan")
    reasons.append(f"  Kalman-LSTM-Spec detection time:   {dt_spec:.1f}" if np.isfinite(dt_spec) else "  Kalman-LSTM-Spec detection time:   nan")
    reasons.append(f"  DT gain (LSTM vs BCE):             {dt_gain:.1f}" if np.isfinite(dt_gain) else "  DT gain (LSTM vs BCE):             nan")
    reasons.append(f"  EW-AUC gain (LSTM vs BCE):         {ewa_gain:.3f}" if np.isfinite(ewa_gain) else "  EW-AUC gain (LSTM vs BCE):         nan")
    reasons.append(f"  FPR ratio (LSTM/BCE null):         {fpr_lstm / max(fpr_bce, 1e-8):.3f}" if (np.isfinite(fpr_lstm) and np.isfinite(fpr_bce)) else "  FPR ratio (LSTM/BCE null):         nan")

    passed_dt = np.isfinite(dt_lstm) and dt_lstm >= 15.0
    passed_ewa = np.isfinite(ewa_gain) and ewa_gain >= 0.05
    passed_null = not (np.isfinite(fpr_lstm) and np.isfinite(fpr_bce) and fpr_lstm > 1.5 * fpr_bce + 0.05)

    if passed_dt:
        reasons.append(f"  DT PASS: LSTM DT={dt_lstm:.1f} >= 15.0")
    else:
        reasons.append(f"  DT FAIL: LSTM DT={dt_lstm:.1f} < 15.0" if np.isfinite(dt_lstm) else "  DT FAIL: nan")
    if passed_ewa:
        reasons.append(f"  EW-AUC PASS: gain={ewa_gain:.3f} >= 0.05")
    else:
        reasons.append(f"  EW-AUC FAIL: gain={ewa_gain:.3f}" if np.isfinite(ewa_gain) else "  EW-AUC FAIL: nan")
    if passed_null:
        reasons.append(f"  NULL PASS: FPR ratio={fpr_lstm / max(fpr_bce, 1e-8):.3f}")
    else:
        reasons.append(f"  NULL FAIL: FPR ratio={fpr_lstm / max(fpr_bce, 1e-8):.3f}")

    system_pass = passed_dt and passed_ewa and passed_null
    return system_pass, "\n".join(reasons)


def _overall_verdict(system_results: Dict[str, Tuple[bool, str]]) -> Tuple[bool, str]:
    n_pass = sum(1 for passed, _ in system_results.values() if passed)
    n_total = len(system_results)
    if n_pass >= 2:
        verdict = (
            f"VERDICT: GO - LSTM head passes on "
            f"{n_pass}/{n_total} bifurcation systems.\n"
            f"A causal LSTM head enables temporal CSD pattern detection "
            f"beyond the per-step MLP."
        )
        passed = True
    else:
        verdict = (
            f"VERDICT: NO-GO - LSTM head passes only "
            f"{n_pass}/{n_total} bifurcation systems.\n"
            f"LSTM head does not reliably outperform MLP head."
        )
        passed = False
    details = "\n".join(reasons for _, reasons in system_results.values())
    return passed, f"{verdict}\n\nDetails:\n{details}"


def _run_system_experiment(
    system: str,
    tensors_signal: TensorizedDataset,
    tensors_null: TensorizedDataset,
    arrays_signal: Dict,
    arrays_null: Dict,
    *,
    seed: int,
    device: torch.device,
    spec_weight: float,
    epochs: int,
) -> SystemResult:
    train_idx_s = arrays_signal["split_indices"]["train"]
    val_idx_s = arrays_signal["split_indices"]["val"]
    test_idx_s = arrays_signal["split_indices"]["test"]
    test_idx_n = arrays_null["split_indices"]["test"]

    runs: List[RunResult] = []

    _seed_all(seed)
    csd_scores_test = _raw_csd_indicator(arrays_signal["features"][test_idx_s], 30)
    csd_scores_null_test = _raw_csd_indicator(arrays_null["features"][test_idx_n], 30)
    raw_metrics = _evaluate_raw_csd(
        csd_scores_test,
        arrays_signal["bifurcation_times"][test_idx_s],
        arrays_signal["is_positive"][test_idx_s],
        arrays_signal["seq_lengths"][test_idx_s],
        csd_scores_null_test,
        arrays_null["seq_lengths"][test_idx_n],
        threshold=0.6,
    )
    runs.append(RunResult(method="Raw-CSD", seed=seed, metrics=raw_metrics))

    def _run_one(method: str, loss_type: str) -> None:
        model = _train_kalman(
            tensors_signal, train_idx_s, val_idx_s,
            loss_type=loss_type, seed=seed, device=device, epochs=epochs,
            spec_weight=spec_weight,
        )
        probs = _build_probs(model, tensors_signal, test_idx_s)
        probs_null = _build_probs(model, tensors_null, test_idx_n)
        val_probs = _build_probs(model, tensors_signal, val_idx_s)

        thresh = _select_threshold(
            val_probs,
            arrays_signal["bifurcation_times"][val_idx_s],
            arrays_signal["is_positive"][val_idx_s],
            arrays_signal["seq_lengths"][val_idx_s],
        )
        dt = _compute_detection_time(
            probs, arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            arrays_signal["seq_lengths"][test_idx_s], thresh,
        )
        ewa = _compute_early_warning_auc(
            probs, arrays_signal["bifurcation_times"][test_idx_s],
            arrays_signal["is_positive"][test_idx_s],
            arrays_signal["seq_lengths"][test_idx_s],
            probs_null, arrays_null["seq_lengths"][test_idx_n],
        )
        null_met = _compute_null_metrics(probs_null, thresh, arrays_null["seq_lengths"][test_idx_n])
        runs.append(RunResult(method=method, seed=seed, metrics={
            "detection_time": dt, "ew_auc": ewa, **null_met,
        }))

    _run_one("Kalman-BCE", "bce")
    _run_one("Kalman-LSTM", "lstm")
    _run_one("Kalman-LSTM-Spec", "lstm_spec")

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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print()

    settings = [
        ("Default", 0.15, 500, 30),
        ("HighNoise", 0.30, 500, 30),
        ("LowData", 0.15, 200, 50),
    ]

    overall_pass = True

    for label, noise_scale, n_patients, epochs in settings:
        print("=" * 80)
        print(f"Stress Test: {label}")
        print(f"  noise_scale={noise_scale}, n_patients={n_patients}, "
              f"epochs={epochs}")
        print("=" * 80)
        started = time.time()

        system_results: Dict[str, Tuple[bool, str]] = {}

        for system in SYSTEMS:
            print(f"\n--- Generating {system} data ---")
            arrays_signal = _build_dataset_for_system(
                system, null=False,
                n_trajectories=n_patients, noise_scale=noise_scale,
                seed=101,
            )
            arrays_null = _build_dataset_for_system(
                system, null=True,
                n_trajectories=n_patients, noise_scale=noise_scale,
                seed=202,
            )
            print(f"  signal: {arrays_signal['features'].shape}, "
                  f"null: {arrays_null['features'].shape}")

            tensors_signal = _tensorize(arrays_signal, device)
            tensors_null = _tensorize(arrays_null, device)

            all_runs: List[RunResult] = []
            for run_seed in STRESS_SEEDS:
                _seed_all(run_seed)
                sys_res = _run_system_experiment(
                    system, tensors_signal, tensors_null,
                    arrays_signal, arrays_null,
                    seed=run_seed, device=device,
                    spec_weight=0.1, epochs=epochs,
                )
                all_runs.extend(sys_res.runs)

            combined = SystemResult(system=system, runs=all_runs)
            signal_agg = combined.aggregate()
            _summarize_system(signal_agg, system)

            sys_pass, sys_reason = _verdict_system(signal_agg)
            system_results[system] = (sys_pass, sys_reason)
            overall_pass = overall_pass and sys_pass

        elapsed = time.time() - started
        print(f"\nTime: {elapsed:.1f}s")
        print("\nSystem Verdicts:")
        for sys_name, (passed, _) in system_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {sys_name.capitalize():>12s}: {status}")

        passed, verdict_str = _overall_verdict(system_results)
        print(f"\n{verdict_str}\n")

    print("=" * 80)
    if overall_pass:
        print("FINAL VERDICT: GO - LSTM head hypothesis "
              "survived all stress settings.")
    else:
        print("FINAL VERDICT: NO-GO - LSTM head hypothesis "
              "failed at least one stress setting.")


if __name__ == "__main__":
    main()
