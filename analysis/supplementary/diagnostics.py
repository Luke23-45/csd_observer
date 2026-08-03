"""CPU-only diagnostics on the saved benchmark artifacts.

1. Window sensitivity: recompute EW-AUC for both observer variants with
   early-window variants (40,5), (60,5), (50,10) and the full pre-bifurcation
   window (120,5) against the reported (50,5).  Supports the metric-window
   discussion (review item on justifying the [tau-50, tau-5) window).

2. Null-score analysis: for both variants, the distribution of per-trajectory
   max scores of null test trajectories inside the early-warning window and
   inside the full pre-bifurcation window (fractions exceeding 0.5 and 0.8),
   aggregated over all ten seeds.

3. fold@200 per-seed view: DT / EW-AUC / FPR / threshold / coverage for both
   variants, seed by seed (explains the large cross-seed spread in Table 2).

4. hopf@100 LSTM-Spec per-seed view: DT and coverage per seed (paired-test
   sample size and mean-DT interpretation).

Outputs JSON under analysis/supplementary/outputs/diagnostics.json
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

from analysis.visualize.common.constants import PATIENT_COUNTS, SYSTEMS, SEED_ORDER  # noqa: E402
from analysis.visualize.benchmark.data import load_benchmark_store  # noqa: E402
from csd_observer.utils.metrics import compute_early_warning_auc  # noqa: E402

DATA_ROOT = _ROOT / "analysis" / "experiment_data"
OUT_DIR = _ROOT / "analysis" / "supplementary" / "outputs"

METHODS2 = ["Kalman-BCE", "Kalman-LSTM-Spec"]
WINDOW_VARIANTS = [
    ("w50_5", 50.0, 5.0),
    ("w40_5", 40.0, 5.0),
    ("w60_5", 60.0, 5.0),
    ("w50_10", 50.0, 10.0),
    ("w120_5", 120.0, 5.0),
]


def _auc_window(probs, tau, lens, probs_null, lens_null, start_delta, end_delta) -> float:
    is_pos = tau > 0
    return compute_early_warning_auc(
        probs, tau, is_pos, lens,
        probs_null, lens_null,
        early_start_delta=start_delta, early_end_delta=end_delta,
    )


def _window_max_by_traj(probs, lens, start_delta, end_delta) -> np.ndarray:
    out = np.full(len(probs), np.nan)
    for i in range(len(probs)):
        T = int(lens[i])
        start = max(0, int(T - start_delta))
        stop = max(0, int(T - end_delta))
        if stop > start:
            out[i] = float(np.max(probs[i, start:stop]))
    return out


def main() -> None:
    store = load_benchmark_store(DATA_ROOT)
    out: dict = {"window_sensitivity": {}, "null_scores": {}, "per_seed": {}}

    for pc in PATIENT_COUNTS:
        out["window_sensitivity"][str(pc)] = {}
        out["null_scores"][str(pc)] = {}
        out["per_seed"][str(pc)] = {}
        for system in SYSTEMS:
            ws_cell = {}
            ns_cell = {}
            for method in METHODS2:
                aucs = {name: [] for name, _, _ in WINDOW_VARIANTS}
                null_max_ew = []
                null_max_full = []
                null_seed_max_ew = []
                for seed in SEED_ORDER:
                    traj = store.load_trajectory(pc, system, method, seed)
                    for name, sd, ed in WINDOW_VARIANTS:
                        aucs[name].append(
                            _auc_window(
                                traj["probs_test"], traj["bifurcation_times"],
                                traj["seq_lengths"], traj["probs_null"],
                                traj["seq_lengths_null"], sd, ed,
                            )
                        )
                    ew = _window_max_by_traj(
                        traj["probs_null"], traj["seq_lengths_null"], 50.0, 5.0,
                    )
                    full = _window_max_by_traj(
                        traj["probs_null"], traj["seq_lengths_null"], 120.0, 5.0,
                    )
                    null_max_ew.extend(ew[~np.isnan(ew)].tolist())
                    null_max_full.extend(full[~np.isnan(full)].tolist())
                    null_seed_max_ew.append(float(np.nanmax(ew)) if np.any(np.isfinite(ew)) else float("nan"))

                ref = np.array(aucs["w50_5"])
                ws_cell[method] = {
                    "mean_auc": {name: float(np.mean(v)) for name, v in aucs.items()},
                    "mean_abs_delta_vs_50_5": {
                        name: float(np.mean(np.abs(np.array(v) - ref))) for name, v in aucs.items() if name != "w50_5"
                    },
                }
                ew_arr = np.array(null_max_ew)
                full_arr = np.array(null_max_full)
                ns_cell[method] = {
                    "n_null_trajectories": int(len(ew_arr)),
                    "frac_max_gt_0_5_ew_window": float(np.mean(ew_arr > 0.5)),
                    "frac_max_gt_0_8_ew_window": float(np.mean(ew_arr > 0.8)),
                    "frac_max_gt_0_5_full_window": float(np.mean(full_arr > 0.5)),
                    "frac_max_gt_0_8_full_window": float(np.mean(full_arr > 0.8)),
                    "mean_seedwise_max_ew": float(np.nanmean(null_seed_max_ew)),
                }

            out["window_sensitivity"][str(pc)][system] = ws_cell
            out["null_scores"][str(pc)][system] = ns_cell

    for pc, system in ((200, "fold"), (100, "hopf")):
        df = store.batch_records(pc)
        rows = []
        for method in METHODS2:
            for seed in SEED_ORDER:
                rec = df[(df["system"] == system) & (df["method"] == method) & (df["seed"] == seed)]
                if rec.empty:
                    continue
                r = rec.iloc[0]
                traj = store.load_trajectory(pc, system, method, seed)
                cov = float(np.isfinite(traj["detection_times"]).mean())
                rows.append({
                    "seed": int(seed),
                    "method": method,
                    "detection_time": float(r["detection_time"]),
                    "ew_auc": float(r["ew_auc"]),
                    "fpr": float(r["fpr"]),
                    "threshold": float(r["threshold"]),
                    "coverage": cov,
                })
        out["per_seed"][str(pc)][system] = rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== Window sensitivity (mean EW-AUC over seeds) ===")
    for pc in PATIENT_COUNTS:
        for system in SYSTEMS:
            for method in METHODS2:
                c = out["window_sensitivity"][str(pc)][system][method]
                print(
                    f"patients={pc} {system:>8s} {method:>16s} "
                    + " ".join(f"{k}={v:.3f}" for k, v in c["mean_auc"].items())
                )
    print("\n=== Null max-score fractions ===")
    for pc in PATIENT_COUNTS:
        for system in SYSTEMS:
            for method in METHODS2:
                c = out["null_scores"][str(pc)][system][method]
                print(
                    f"patients={pc} {system:>8s} {method:>16s} "
                    f"ew>0.5={c['frac_max_gt_0_5_ew_window']:.3f} ew>0.8={c['frac_max_gt_0_8_ew_window']:.3f} "
                    f"full>0.5={c['frac_max_gt_0_5_full_window']:.3f} full>0.8={c['frac_max_gt_0_8_full_window']:.3f}"
                )
    print("\n=== Per-seed (fold@200, hopf@100) ===")
    for pc, system in ((200, "fold"), (100, "hopf")):
        print(f"patients={pc} system={system}")
        for r in out["per_seed"][str(pc)][system]:
            print(
                f"  seed={r['seed']:>4d} {r['method']:>16s} DT={r['detection_time']:>8.2f} "
                f"AUC={r['ew_auc']:.3f} FPR={r['fpr']:.3f} thr={r['threshold']:.3f} cov={r['coverage']:.3f}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
