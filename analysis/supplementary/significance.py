"""Paired significance tests, medians, and alert coverage for the benchmark.

Reads the selected benchmark batches (same selection rule as the visualization
pipeline) and computes, for every (patient count, system):

  * paired Wilcoxon signed-rank and paired t-test p-values comparing
    Kalman-BCE vs Kalman-LSTM-Spec on ew_auc, detection_time, fpr;
  * medians (across the ten seeds) for all three metrics;
  * alert coverage: the fraction of test signal trajectories with at least
    one pre-bifurcation threshold crossing (from the saved per-trajectory
    detection times), which complements the mean detection time.

Outputs JSON under analysis/supplementary/outputs/.
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

from analysis.visualize.common.constants import METHODS, PATIENT_COUNTS, SYSTEMS, SEED_ORDER  # noqa: E402
from analysis.visualize.benchmark.data import load_benchmark_store  # noqa: E402

DATA_ROOT = _ROOT / "analysis" / "experiment_data"
OUT_DIR = _ROOT / "analysis" / "supplementary" / "outputs"

PAIRS = [("Kalman-BCE", "Kalman-LSTM-Spec")]
METRICS = ["ew_auc", "detection_time", "fpr"]


def _pair_rows(df: pd.DataFrame, system: str, method_a: str, method_b: str) -> pd.DataFrame:
    a = df[(df["system"] == system) & (df["method"] == method_a)].set_index("seed")
    b = df[(df["system"] == system) & (df["method"] == method_b)].set_index("seed")
    common = a.index.intersection(b.index)
    out = pd.DataFrame(index=common)
    for metric in METRICS:
        va = pd.to_numeric(a.loc[common, metric], errors="coerce")
        vb = pd.to_numeric(b.loc[common, metric], errors="coerce")
        out[f"{method_a}_{metric}"] = va
        out[f"{method_b}_{metric}"] = vb
    out["seed"] = [int(s) for s in common]
    return out


def _test_pair(pair: pd.DataFrame, metric: str) -> dict:
    col_a = [c for c in pair.columns if c.startswith("Kalman-BCE") and c.endswith(metric)]
    col_b = [c for c in pair.columns if c.startswith("Kalman-LSTM-Spec") and c.endswith(metric)]
    va = pair[col_a[0]].to_numpy(dtype=float)
    vb = pair[col_b[0]].to_numpy(dtype=float)
    finite = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[finite], vb[finite]
    n = int(finite.sum())
    if n < 2:
        return {
            "n": n, "wilcoxon_p": float("nan"), "ttest_p": float("nan"),
            "mean_diff": float("nan"), "median_a": float("nan"), "median_b": float("nan"),
        }
    diff = vb - va
    with np.errstate(invalid="ignore"):
        if np.any(diff != 0):
            w_p = stats.wilcoxon(vb, va, alternative="two-sided").pvalue
        else:
            w_p = 1.0
        t_p = stats.ttest_rel(vb, va).pvalue
    return {
        "n": n,
        "wilcoxon_p": float(w_p),
        "ttest_p": float(t_p),
        "mean_diff": float(np.mean(diff)),
        "median_a": float(np.median(va)),
        "median_b": float(np.median(vb)),
    }


def _coverage(store, patient_count: int, system: str, method: str) -> dict:
    coverage = []
    for seed in SEED_ORDER:
        traj = store.load_trajectory(patient_count, system, method, seed)
        dts = traj["detection_times"]
        finite = np.isfinite(dts)
        coverage.append(float(finite.mean()))
    cov = np.array(coverage)
    return {
        "mean": float(cov.mean()),
        "std": float(cov.std()),
        "min": float(cov.min()),
        "max": float(cov.max()),
    }


def main() -> None:
    store = load_benchmark_store(DATA_ROOT)
    results: dict = {}
    for pc in PATIENT_COUNTS:
        df = store.batch_records(pc)
        for system in SYSTEMS:
            pair = _pair_rows(df, system, *PAIRS[0])
            tests = {m: _test_pair(pair, m) for m in METRICS}
            cov = {m: _coverage(store, pc, system, m) for m in ("Kalman-BCE", "Kalman-LSTM-Spec")}
            results[str(pc)] = results.get(str(pc), {})
            results[str(pc)][system] = {"paired": tests, "coverage": cov}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "significance.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for pc in PATIENT_COUNTS:
        for system in SYSTEMS:
            entry = results[str(pc)][system]
            print(f"patients={pc} system={system}")
            for metric in METRICS:
                t = entry["paired"][metric]
                print(
                    f"  {metric:>15s} wilcoxon_p={t['wilcoxon_p']:.4g} ttest_p={t['ttest_p']:.4g} "
                    f"mean_diff(LSTM-BCE)={t['mean_diff']:+.4f} med_B={t['median_b']:.4f} med_A={t['median_a']:.4f} n={t['n']}"
                )
            for m in ("Kalman-BCE", "Kalman-LSTM-Spec"):
                c = entry["coverage"][m]
                print(f"  coverage {m:>16s} mean={c['mean']:.3f} std={c['std']:.3f} range=[{c['min']:.3f},{c['max']:.3f}]")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
