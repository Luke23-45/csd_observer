"""Classical early-warning baselines on the same metric pipeline (v2, reviewed).

Regenerates the exact trajectory data used by the benchmark runs
(noise_scale=0.15 for all systems, default per-system observation noise,
data seed 101 for signal / 202 for null, split seed 1101) and evaluates
four classical indicators on the observations the observers also see:

  * Var-30       rolling variance (window 30)
  * AC1-30       lag-1 autocorrelation (window 30)
  * AC2-30       lag-2 autocorrelation (window 30)
  * AC2d-30      lag-2 autocorrelation with linear detrending (window 30)

Evaluation uses the exact functions and windows of the observer pipeline:
  * EW-AUC over max scores in [tau-50, tau-5) vs null terminal windows;
  * threshold selection on the validation split via select_threshold
    (Youden's statistic with the same 0.5 fallback);
  * detection time (mean over alerting trajectories),
  * per-step false-positive rate on null trajectories.

The indicators are deterministic (no training, no seed averaging): for a
given (system, patient count) there is one score sequence per trajectory.
Uncertainty for EW-AUC is reported as a bootstrap 95% CI over trajectories,
using the same windowed max scores that define the AUC.

Outputs JSON under analysis/supplementary/outputs/classical_baselines.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from csd_observer.data.bifurcation import build_dataset  # noqa: E402
from csd_observer.utils.metrics import (  # noqa: E402
    compute_detection_time,
    compute_early_warning_auc,
    compute_false_positive_rate,
    raw_csd_indicator,
    raw_lag2_indicator,
    raw_lag2_indicator_detrended,
    raw_var_indicator,
    select_threshold,
)

OUT_DIR = _ROOT / "analysis" / "supplementary" / "outputs"
PATIENT_COUNTS = [100, 200, 300, 400, 500]
SYSTEMS = ["fold", "hopf", "logistic"]
WINDOW = 30
SIGNAL_DATA_SEED = 101
NULL_DATA_SEED = 202

INDICATORS = {
    "Var-30": raw_var_indicator,
    "AC1-30": raw_csd_indicator,
    "AC2-30": raw_lag2_indicator,
    "AC2d-30": raw_lag2_indicator_detrended,
}


def _per_traj_dts(scores: np.ndarray, tau: np.ndarray, is_pos: np.ndarray, threshold: float) -> np.ndarray:
    """Per-trajectory detection time; nan when no pre-bifurcation alert exists."""
    dts = np.full(len(scores), np.nan, dtype=np.float64)
    if not np.isfinite(threshold):
        return dts
    for i in range(len(scores)):
        if not bool(is_pos[i]):
            continue
        t = int(tau[i])
        if t <= 0:
            continue
        alerts = np.where(scores[i, :t] >= threshold)[0]
        if len(alerts) > 0:
            dts[i] = float(tau[i] - alerts[0])
    return dts


def _window_max(scores: np.ndarray, ends: np.ndarray, start_delta: float, end_delta: float) -> np.ndarray:
    """Per-trajectory max score in [max(0, end-start_delta), max(0, end-end_delta))."""
    out = np.full(len(scores), np.nan, dtype=np.float64)
    for i in range(len(scores)):
        start = max(0, int(ends[i] - start_delta))
        stop = max(0, int(ends[i] - end_delta))
        if stop > start:
            out[i] = float(np.max(scores[i, start:stop]))
    return out


def _bootstrap_ci(signal_max: np.ndarray, null_max: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple:
    scores = np.concatenate([signal_max, null_max])
    labels = np.concatenate([np.ones(len(signal_max)), np.zeros(len(null_max))])
    finite = np.isfinite(scores) & (labels == (labels == labels))  # labels always finite
    scores, labels = scores[finite], labels[finite]
    if len(set(labels.tolist())) < 2:
        return (float("nan"), float("nan"))
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), size=len(scores))
        if len(set(labels[idx].tolist())) < 2:
            continue
        aucs.append(float(roc_auc_score(labels[idx], scores[idx])))
    if not aucs:
        return (float("nan"), float("nan"))
    arr = np.array(aucs)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def main() -> None:
    out: dict = {}
    for pc in PATIENT_COUNTS:
        out[str(pc)] = {}
        for system in SYSTEMS:
            sig = build_dataset(
                system, n_trajectories=pc, max_length=200,
                noise_scale=0.15, seed=SIGNAL_DATA_SEED, null=False,
            )
            nul = build_dataset(
                system, n_trajectories=pc, max_length=200,
                noise_scale=0.15, seed=NULL_DATA_SEED, null=True,
            )
            tr, va, te = (sig["split_indices"][k] for k in ("train", "val", "test"))
            te_n = nul["split_indices"]["test"]

            feats_sig, lens_sig = sig["features"], sig["seq_lengths"]
            feats_nul, lens_nul = nul["features"], nul["seq_lengths"]
            tau_sig, tau_nul = sig["bifurcation_times"], nul["bifurcation_times"]
            is_pos_sig = sig["is_positive"]

            cell = {}
            for name, fn in INDICATORS.items():
                s_test = fn(feats_sig[te], lens_sig[te], WINDOW)
                s_null = fn(feats_nul[te_n], lens_nul[te_n], WINDOW)
                s_val = fn(feats_sig[va], lens_sig[va], WINDOW)

                sig_max = _window_max(s_test, tau_sig[te], 50.0, 5.0)
                null_max = _window_max(s_null, tau_nul[te_n], 50.0, 5.0)

                ewa = compute_early_warning_auc(
                    s_test, tau_sig[te], is_pos_sig[te], lens_sig[te],
                    s_null, lens_nul[te_n],
                )
                ci_lo, ci_hi = _bootstrap_ci(sig_max, null_max)
                thr = float(select_threshold(s_val, tau_sig[va], is_pos_sig[va], lens_sig[va]))
                dts = _per_traj_dts(s_test, tau_sig[te], is_pos_sig[te], thr)
                coverage = float(np.isfinite(dts).mean())
                dt = float(np.nanmean(dts)) if coverage > 0 else float("nan")
                fpr = float(compute_false_positive_rate(s_null, lens_nul[te_n], thr))

                cell[name] = {
                    "ew_auc": float(ewa),
                    "auc_ci95": [ci_lo, ci_hi],
                    "threshold": thr,
                    "detection_time": dt,
                    "fpr": fpr,
                    "coverage": coverage,
                    "n_test_signal": int(len(te)),
                    "n_test_null": int(len(te_n)),
                }
            out[str(pc)][system] = cell
            for name, v in cell.items():
                print(
                    f"patients={pc} {system:>8s} {name:>8s} AUC={v['ew_auc']:.3f} "
                    f"CI=[{v['auc_ci95'][0]:.3f},{v['auc_ci95'][1]:.3f}] thr={v['threshold']:.3f} "
                    f"DT={v['detection_time']:.2f} FPR={v['fpr']:.3f} cov={v['coverage']:.3f}"
                )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "classical_baselines.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
