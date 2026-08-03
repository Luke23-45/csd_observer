"""Matched-threshold robustness and achieved spectral radius.

Part 1: for every (patient count, system), recompute detection time,
false-positive rate, and alert coverage for BOTH observer variants at a
common threshold:
  * theta = 0.5 (the fallback operating point for Hopf and logistic);
  * theta = the BCE-selected threshold and the LSTM-Spec-selected threshold
    (each applied to both methods).

This isolates decision-rule effects from score-calibration effects.

Part 2: achieved spectral radius rho(M) from the saved epoch logs of the
LSTM-Spec runs (reported at the final logged epoch, averaged over seeds).

Outputs JSON under analysis/supplementary/outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.visualize.common.constants import METHODS, PATIENT_COUNTS, SYSTEMS, SEED_ORDER  # noqa: E402
from analysis.visualize.benchmark.data import load_benchmark_store  # noqa: E402

DATA_ROOT = _ROOT / "analysis" / "experiment_data"
OUT_DIR = _ROOT / "analysis" / "supplementary" / "outputs"

METHODS2 = ["Kalman-BCE", "Kalman-LSTM-Spec"]


def _dt_fpr_coverage(probs_test, probs_null, tau, seq_len_null, threshold):
    probs_test = np.nan_to_num(probs_test, nan=0.5, posinf=1.0, neginf=0.0)
    probs_null = np.nan_to_num(probs_null, nan=0.5, posinf=1.0, neginf=0.0)
    dts = []
    for i in range(probs_test.shape[0]):
        pre = probs_test[i, : int(tau[i])]
        alerts = np.where(pre >= threshold)[0]
        dts.append(float(tau[i] - alerts[0]) if len(alerts) > 0 else float("nan"))
    dts = np.array(dts)
    coverage = float(np.isfinite(dts).mean())
    dt_mean = float(np.nanmean(dts)) if coverage > 0 else float("nan")
    total_steps = int(seq_len_null.sum())
    alert_steps = int((probs_null >= threshold).sum()) if total_steps > 0 else 0
    fpr = alert_steps / max(total_steps, 1)
    return {"detection_time": dt_mean, "fpr": float(fpr), "coverage": coverage}


def _best_epoch_spectral_radius(log: pd.DataFrame) -> Optional[float]:
    """Spectral radius at the epoch whose val_metric (incl. spec penalty) is best."""
    if "spectral_radius" not in log.columns or "val_metric" not in log.columns:
        return None
    sub = log.dropna(subset=["spectral_radius"])
    if sub.empty:
        return None
    best = sub.loc[sub["val_metric"].idxmin()]
    return float(best["spectral_radius"])


def main() -> None:
    store = load_benchmark_store(DATA_ROOT)
    out: dict = {"matched_threshold": {}, "spectral_radius": {}}
    for pc in PATIENT_COUNTS:
        df = store.batch_records(pc)
        out["matched_threshold"][str(pc)] = {}
        for system in SYSTEMS:
            cell = {}
            scores = {}
            selected = {}
            for method in METHODS2:
                vals = []
                for seed in SEED_ORDER:
                    traj = store.load_trajectory(pc, system, method, seed)
                    vals.append(
                        {
                            "test": traj["probs_test"],
                            "null": traj["probs_null"],
                            "tau": traj["bifurcation_times"],
                            "seq_len_null": traj["seq_lengths_null"],
                            "thr": float(traj["threshold"][0]),
                        }
                    )
                scores[method] = vals
                thr = np.array([v["thr"] for v in vals])
                selected[method] = float(np.nanmean(thr))
            for th_name, thr in [("theta=0.5", 0.5), ("theta_BCE", selected["Kalman-BCE"]), ("theta_LSTM", selected["Kalman-LSTM-Spec"])]:
                for method in METHODS2:
                    agg = []
                    for v in scores[method]:
                        agg.append(_dt_fpr_coverage(v["test"], v["null"], v["tau"], v["seq_len_null"], thr))
                    cell[f"{th_name}_{method}"] = {
                        "detection_time": float(np.nanmean([a["detection_time"] for a in agg if np.isfinite(a["detection_time"])]))
                        if any(np.isfinite(a["detection_time"]) for a in agg)
                        else float("nan"),
                        "fpr": float(np.mean([a["fpr"] for a in agg])),
                        "coverage": float(np.mean([a["coverage"] for a in agg])),
                    }
            out["matched_threshold"][str(pc)][system] = cell

    for pc in PATIENT_COUNTS:
        out["spectral_radius"][str(pc)] = {}
        for system in SYSTEMS:
            final = []
            for seed in SEED_ORDER:
                path = store.epoch_log_path(pc, system, "Kalman-LSTM-Spec", seed)
                if not path.exists():
                    continue
                log = pd.read_csv(path)
                if "spectral_radius" not in log.columns or log["spectral_radius"].dropna().empty:
                    continue
                final.append(float(log["spectral_radius"].dropna().iloc[-1]))
            arr = np.array(final)
            out["spectral_radius"][str(pc)][system] = (
                {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))} if len(arr) else None
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "matched_threshold.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for pc in PATIENT_COUNTS:
        for system in SYSTEMS:
            print(f"patients={pc} system={system}")
            for key, val in out["matched_threshold"][str(pc)][system].items():
                print(f"  {key:>28s} DT={val['detection_time']:.2f} FPR={val['fpr']:.3f} cov={val['coverage']:.3f}")
    print()
    for pc in PATIENT_COUNTS:
        for system in SYSTEMS:
            v = out["spectral_radius"][str(pc)][system]
            if v:
                print(f"  spectral_radius patients={pc} {system}: mean={v['mean']:.4f} std={v['std']:.4f} n={v['n']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
