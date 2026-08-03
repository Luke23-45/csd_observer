"""Ablation analysis: spectral-penalty ablation of the learned observer.

The spectral penalty (rho(M) regularizer, weight 0.01, threshold 0.95) is
currently part of the LSTM-Spec variant.  The ablation grid to be run on GPU
produces the two missing cells:

  * Kalman-LSTM     - LSTM head, BCE objective, NO spectral penalty
  * Kalman-BCE-Spec - MLP head, BCE objective, WITH spectral penalty

This script compares those runs against the main benchmark:

  * Kalman-LSTM-Spec (main store)  vs  Kalman-LSTM (ablation store)
  * Kalman-BCE (main store)        vs  Kalman-BCE-Spec (ablation store)

For every (patient count, system) it reports, paired across the ten seeds:
mean +/- std of detection time / EW-AUC / FPR / threshold / epochs trained,
alert coverage from the saved per-trajectory detection times, and paired
Wilcoxon p-values for the three headline metrics.

Usage:
    python analysis/supplementary/ablation_analysis.py [main_store] [ablation_store]

Outputs JSON under analysis/supplementary/outputs/ablation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.visualize.common.constants import PATIENT_COUNTS, SYSTEMS, SEED_ORDER  # noqa: E402
from analysis.visualize.benchmark.data import load_benchmark_store  # noqa: E402

OUT_DIR = _ROOT / "analysis" / "supplementary" / "outputs"

MAIN_STORE_DEFAULT = _ROOT / "analysis" / "experiment_data"
ABLATION_STORE_DEFAULT = _ROOT / "outputs" / "ablation"

# (ablation method, main-store method)
PAIRS = [("Kalman-LSTM", "Kalman-LSTM-Spec"), ("Kalman-BCE-Spec", "Kalman-BCE")]
METRICS = ["detection_time", "ew_auc", "fpr", "threshold", "n_epochs_trained"]
HEADLINE = ["detection_time", "ew_auc", "fpr"]


def _paired(df_new: pd.DataFrame, df_old: pd.DataFrame, system: str, method_new: str, method_old: str) -> pd.DataFrame:
    a = df_new[(df_new["system"] == system) & (df_new["method"] == method_new)].set_index("seed")
    b = df_old[(df_old["system"] == system) & (df_old["method"] == method_old)].set_index("seed")
    common = a.index.intersection(b.index)
    rows = {"seed": [int(s) for s in common]}
    for metric in METRICS:
        rows[f"{method_new}_{metric}"] = pd.to_numeric(a.loc[common, metric], errors="coerce").to_numpy(dtype=float)
        rows[f"{method_old}_{metric}"] = pd.to_numeric(b.loc[common, metric], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(rows)


def _coverage(store, pc: int, system: str, method: str) -> dict:
    covs = []
    for seed in SEED_ORDER:
        traj = store.load_trajectory(pc, system, method, seed)
        covs.append(float(np.isfinite(traj["detection_times"]).mean()))
    arr = np.array(covs)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def main() -> None:
    main_root = Path(sys.argv[1]) if len(sys.argv) > 1 else MAIN_STORE_DEFAULT
    ablation_root = Path(sys.argv[2]) if len(sys.argv) > 2 else ABLATION_STORE_DEFAULT
    store_main = load_benchmark_store(main_root)
    store_abl = load_benchmark_store(ablation_root)

    out: dict = {}
    for pc in PATIENT_COUNTS:
        df_main = store_main.batch_records(pc)
        df_abl = store_abl.batch_records(pc)
        out[str(pc)] = {}
        for system in SYSTEMS:
            cell = {}
            for method_abl, method_main in PAIRS:
                pair = _paired(df_abl, df_main, system, method_abl, method_main)
                entry = {
                    "method_ablation": method_abl,
                    "method_main": method_main,
                    "n_paired_seeds": int(len(pair)),
                }
                for metric in METRICS:
                    va = pair[f"{method_abl}_{metric}"].to_numpy(dtype=float)
                    vb = pair[f"{method_main}_{metric}"].to_numpy(dtype=float)
                    finite = np.isfinite(va) & np.isfinite(vb)
                    entry[f"{metric}_ablation_mean"] = float(np.nanmean(va)) if np.any(np.isfinite(va)) else float("nan")
                    entry[f"{metric}_ablation_std"] = float(np.nanstd(va)) if np.any(np.isfinite(va)) else float("nan")
                    entry[f"{metric}_main_mean"] = float(np.nanmean(vb)) if np.any(np.isfinite(vb)) else float("nan")
                    entry[f"{metric}_main_std"] = float(np.nanstd(vb)) if np.any(np.isfinite(vb)) else float("nan")
                    if metric in HEADLINE:
                        va_f, vb_f = va[finite], vb[finite]
                        if len(va_f) >= 2:
                            with np.errstate(invalid="ignore"):
                                w_p = stats.wilcoxon(vb_f, va_f, alternative="two-sided").pvalue if np.any(vb_f != va_f) else 1.0
                            entry[f"{metric}_wilcoxon_p"] = float(w_p)
                            entry[f"{metric}_mean_diff"] = float(np.mean(vb_f - va_f))
                        else:
                            entry[f"{metric}_wilcoxon_p"] = float("nan")
                            entry[f"{metric}_mean_diff"] = float("nan")
                cov_abl = _coverage(store_abl, pc, system, method_abl)
                cov_main = _coverage(store_main, pc, system, method_main)
                entry["coverage_ablation"] = cov_abl
                entry["coverage_main"] = cov_main
                cell[f"{method_abl}__vs__{method_main}"] = entry
            out[str(pc)][system] = cell

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ablation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for pc in PATIENT_COUNTS:
        for system in SYSTEMS:
            for key, e in out[str(pc)][system].items():
                print(
                    f"patients={pc} {system:>8s} {key:>38s} "
                    f"DT={e['detection_time_ablation_mean']:.2f}/{e['detection_time_main_mean']:.2f} "
                    f"AUC={e['ew_auc_ablation_mean']:.3f}/{e['ew_auc_main_mean']:.3f} "
                    f"FPR={e['fpr_ablation_mean']:.3f}/{e['fpr_main_mean']:.3f} "
                    f"cov={e['coverage_ablation']['mean']:.3f}/{e['coverage_main']['mean']:.3f} "
                    f"n={e['n_paired_seeds']}"
                )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
