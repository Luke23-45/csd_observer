"""Aggregation, bootstrap CIs, and paired comparisons.

Paper-ready CSVs produced from a run's ``results.jsonl``:

* ``tables/aggregates.csv``         — mean±std per (system, method) over seeds
* ``tables/bootstrap_ci.csv``       — bootstrap 95% CIs per (system, method, metric)
* ``tables/paired_wilcoxon.csv``    — pairwise P-DT comparisons (same data)

All numeric handling is NaN-safe; bootstraps use a fixed seed for
reproducibility (the caller may override).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csd_observer.outputs.schema import ResultRow

_BOOT_SEED = 20240811
_BOOT_ITERS = 1000


@dataclass
class Aggregate:
    system: str
    method: str
    family: str
    is_learned: bool
    bif_type: str
    n_seeds: int
    detection_rate_mean: float
    detection_rate_std: float
    detection_time_mean: float
    detection_time_std: float
    detection_time_median: float
    ew_auc_mean: float
    ew_auc_std: float
    fpr_mean: float
    persistent_fpr_mean: float
    k_persist: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def aggregate_rows(rows: Sequence[ResultRow]) -> list[Aggregate]:
    grouped: dict[tuple[str, str], list[ResultRow]] = {}
    for r in rows:
        grouped.setdefault((r.system, r.method), []).append(r)
    out: list[Aggregate] = []
    for (system, method), group in grouped.items():
        det = np.array([r.detection_time_mean for r in group], dtype=float)
        ew = np.array([r.ew_auc for r in group], dtype=float)
        fpr = np.array([r.fpr for r in group], dtype=float)
        pfpr = np.array([r.persistent_fpr for r in group], dtype=float)
        rate = np.array([r.detection_rate for r in group], dtype=float)
        if len(group) == 0:
            continue
        head = group[0]
        out.append(
            Aggregate(
                system=system,
                method=method,
                family=head.family,
                is_learned=head.is_learned,
                bif_type=head.bif_type,
                n_seeds=len(group),
                detection_rate_mean=_nanmean(rate),
                detection_rate_std=_nanstd(rate),
                detection_time_mean=_nanmean(det),
                detection_time_std=_nanstd(det),
                detection_time_median=float(np.nanmedian(det)) if det.size else float("nan"),
                ew_auc_mean=_nanmean(ew),
                ew_auc_std=_nanstd(ew),
                fpr_mean=_nanmean(fpr),
                persistent_fpr_mean=_nanmean(pfpr),
                k_persist=head.k_persist,
            )
        )
    return out


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_iters: int = _BOOT_ITERS,
    seed: int = _BOOT_SEED,
) -> tuple[float, float, float]:
    """Return (mean, ci95_low, ci95_high) via percentile bootstrap."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_iters, dtype=float)
    n = arr.size
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    return (
        float(means.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bootstrap_ci_per_row(
    rows: Sequence[ResultRow],
    *,
    metric: str = "detection_time_mean",
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[ResultRow]] = {}
    for r in rows:
        grouped.setdefault((r.system, r.method), []).append(r)
    out: list[dict[str, object]] = []
    for (system, method), group in grouped.items():
        vals = [float(getattr(r, metric)) for r in group]
        m, lo, hi = bootstrap_ci(vals)
        out.append(
            {
                "system": system,
                "method": method,
                "metric": metric,
                "n": len(vals),
                "mean": m,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return out


def paired_wilcoxon(
    rows: Sequence[ResultRow],
    *,
    method_a: str,
    method_b: str,
    metric: str = "detection_time_mean",
) -> dict[str, object] | None:
    """Wilcoxon signed-rank between two methods on the same (system, seed)
    pairs (only valid when both methods share the same seeds)."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    pa: dict[tuple[str, int], float] = {}
    pb: dict[tuple[str, int], float] = {}
    for r in rows:
        key = (r.system, int(r.seed))
        v = float(getattr(r, metric))
        if math.isnan(v):
            continue
        if r.method == method_a:
            pa[key] = v
        elif r.method == method_b:
            pb[key] = v
    keys = sorted(set(pa) & set(pb))
    if len(keys) < 5:
        return None
    a = np.array([pa[k] for k in keys])
    b = np.array([pb[k] for k in keys])
    try:
        stat, p = wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
    except ValueError:
        return None
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_pairs": len(keys),
        "mean_diff": float(np.mean(a - b)),
        "median_diff": float(np.median(a - b)),
        "statistic": float(stat),
        "p_value": float(p),
    }


def write_aggregates_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    aggs = aggregate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not aggs:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(aggs[0].to_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for a in aggs:
            w.writerow(a.to_dict())
    return path


def write_bootstrap_csv(rows: Sequence[ResultRow], path: Path) -> Path:
    out: list[dict[str, object]] = []
    for metric in ("detection_time_mean", "ew_auc", "fpr", "persistent_fpr", "detection_rate"):
        out.extend(bootstrap_ci_per_row(rows, metric=metric))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not out:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(out[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return path


def write_paired_wilcoxon_csv(
    rows: Sequence[ResultRow],
    path: Path,
    *,
    baseline: str = "VAR-CSD",
) -> Path:
    methods = sorted({r.method for r in rows})
    out: list[dict[str, object]] = []
    for m in methods:
        if m == baseline:
            continue
        rec = paired_wilcoxon(rows, method_a=baseline, method_b=m)
        if rec is not None:
            out.append(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not out:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(out[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return path


def _nanmean(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.nanmean(a))


def _nanstd(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.nanstd(a))


__all__ = [
    "Aggregate",
    "aggregate_rows",
    "bootstrap_ci",
    "bootstrap_ci_per_row",
    "paired_wilcoxon",
    "write_aggregates_csv",
    "write_bootstrap_csv",
    "write_paired_wilcoxon_csv",
]
