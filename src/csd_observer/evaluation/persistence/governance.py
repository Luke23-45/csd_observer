"""Single governance driver every method runs through (§8.7 of the plan).

    preprocess -> score -> calibrate(val-null) -> persist(k_persist)
        -> metrics(P-DT, P-EW-AUC, FPRs, censoring) -> artifacts -> row

The driver imports no method implementation: it receives a handle that
satisfies :class:`MethodInterface` (duck-typed) and the array bundles
from the dataset registry. ``evaluation`` never imports ``models``.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from csd_observer.evaluation.common.calibration import calibrate_threshold
from csd_observer.evaluation.common.metrics import (
    compute_early_warning_auc,
    compute_false_positive_rate,
)
from csd_observer.evaluation.persistence.protocol import (
    compute_persistent_detection_metrics,
    compute_persistent_ew_auc,
    compute_persistent_fpr,
    compute_persistent_trajectory_fpr,
    null_anchor_upper_bound,
)
from csd_observer.outputs.writer import OutputWriter


def evaluate_method(
    method: Any,
    arrays_signal: dict[str, Any],
    arrays_null: dict[str, Any],
    *,
    system: str,
    bif_type: str,
    dataset: str,
    run_name: str,
    run_id: str,
    timestamp: str,
    seed: int,
    replicate: str = "",
    config: dict[str, Any],
    writer: OutputWriter,
    k_persist: int = 5,
    fpr_target: float = 0.05,
    git_sha: str = "",
    config_hash: str = "",
    store_trajectories: bool = False,
    device: Any = None,
    evaluation: str = "persistenceaware",
) -> dict[str, Any]:
    """Run one method through the persistence-aware governance pipeline.

    Returns the schema-valid row dict (also written to results.jsonl).
    """
    if evaluation not in {"persistenceaware", "baseline_classic"}:
        raise ValueError(
            "evaluation must be 'persistenceaware' or 'baseline_classic', "
            f"got {evaluation!r}"
        )
    if evaluation == "baseline_classic":
        k_persist = 1
    meta = method.meta

    t0 = time.time()
    # ------------------------------------------------------------- fit (train)
    method.fit(
        _train_bundle(arrays_signal),
        _train_bundle(arrays_null),
        config,
    )
    t_fit = time.time() - t0

    # --------------------------------------------------------- score (all)
    t0 = time.time()
    feats_sig = arrays_signal["features"]
    feats_null = arrays_null["features"]
    lens_sig = arrays_signal["seq_lengths"]
    lens_null = arrays_null["seq_lengths"]
    idx_s = arrays_signal["split_indices"]
    idx_n = arrays_null["split_indices"]

    scores_val = method.score(feats_sig[idx_s["val"]], lens_sig[idx_s["val"]], config)
    scores_val_null = method.score(
        feats_null[idx_n["val"]], lens_null[idx_n["val"]], config
    )
    scores_test = method.score(feats_sig[idx_s["test"]], lens_sig[idx_s["test"]], config)
    scores_null = method.score(feats_null[idx_n["test"]], lens_null[idx_n["test"]], config)
    t_score = time.time() - t0

    # ------------------------------------------------------------- calibrate
    threshold = calibrate_threshold(
        scores_val_null, lens_null[idx_n["val"]], fpr_target
    )

    # ------------------------------------------------------------- persist
    sig_metrics = compute_persistent_detection_metrics(
        scores_test,
        arrays_signal["bifurcation_times"][idx_s["test"]],
        arrays_signal["is_positive"][idx_s["test"]],
        lens_sig[idx_s["test"]],
        threshold,
        k_persist,
    )
    persistent_fpr = compute_persistent_fpr(
        scores_null, lens_null[idx_n["test"]], threshold, k_persist
    )
    persistent_trajectory_fpr = compute_persistent_trajectory_fpr(
        scores_null, lens_null[idx_n["test"]], threshold, k_persist
    )
    fpr = compute_false_positive_rate(scores_null, lens_null[idx_n["test"]], threshold)
    ew_auc = compute_early_warning_auc(
        scores_test,
        arrays_signal["bifurcation_times"][idx_s["test"]],
        arrays_signal["is_positive"][idx_s["test"]],
        lens_sig[idx_s["test"]],
        scores_null,
        lens_null[idx_n["test"]],
    )
    p_ew_auc = compute_persistent_ew_auc(
        scores_test,
        arrays_signal["bifurcation_times"][idx_s["test"]],
        arrays_signal["is_positive"][idx_s["test"]],
        lens_sig[idx_s["test"]],
        scores_null,
        lens_null[idx_n["test"]],
        threshold=threshold,
        k_persist=k_persist,
    )
    t_metrics = time.time() - t0

    # ----------------------------------------------------- protocol checks
    achieved_step_fpr = fpr
    anchor = null_anchor_upper_bound(fpr_target, k_persist, 50)
    protocol_checks = {
        "k_persist": k_persist,
        "fpr_target": fpr_target,
        "achieved_step_fpr": achieved_step_fpr if math.isfinite(achieved_step_fpr) else None,
        "achieved_persistent_fpr": persistent_fpr if math.isfinite(persistent_fpr) else None,
        "achieved_persistent_trajectory_fpr": (
            persistent_trajectory_fpr if math.isfinite(persistent_trajectory_fpr) else None
        ),
        "null_anchor_upper_bound_50step": anchor,
        "anchor_brackets_empirical": (
            not math.isfinite(persistent_fpr)
            or not math.isfinite(anchor)
            or persistent_trajectory_fpr <= anchor + 1e-9
        ),
        "drift": (
            "PROTOCOL_DRIFT" if achieved_step_fpr > 2.0 * fpr_target else "ok"
        ),
    }

    # ------------------------------------------------------------- artifacts
    if store_trajectories:
        writer.write_trajectory_npz(
            system,
            meta.name,
            replicate,
            seed,
            scores_test=scores_test,
            scores_null=scores_null,
            scores_val=scores_val,
            scores_val_null=scores_val_null,
            threshold=np.array([threshold]),
            detection_times=np.asarray(sig_metrics["detection_time_mean"], dtype=np.float64),
        )
    writer.write_calibration(
        f"{system}_{meta.name}_seed{seed}",
        {
            "threshold": threshold,
            "fpr_target": fpr_target,
            "k_persist": k_persist,
            "n_val_null": int(len(scores_val_null)),
        },
    )

    # ------------------------------------------------------------------ row
    row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "run_name": run_name,
        "dataset": dataset,
        "bif_type": bif_type,
        "system": system,
        "replicate": replicate,
        "method": meta.name,
        "family": meta.family,
        "is_learned": meta.is_learned,
        "seed": seed,
        "k_persist": k_persist,
        "fpr_target": fpr_target,
        "detection_rate": sig_metrics["detection_rate"],
        "detection_time_mean": sig_metrics["detection_time_mean"],
        "detection_time_median": sig_metrics["detection_time_median"],
        "detection_time_std": sig_metrics["detection_time_std"],
        "ew_auc": ew_auc,
        "fpr": fpr,
        "persistent_fpr": persistent_fpr,
        "threshold": threshold,
        "params": _flatten_params(meta, config),
        "config_hash": config_hash,
        "git_sha": git_sha,
        "achieved_fpr_target": achieved_step_fpr,
        "extra": {
            "persistence_ew_auc": p_ew_auc,
            "n_detected": sig_metrics.get("n_detected", 0),
            "n_censored": sig_metrics.get("n_censored", 0),
            "fit_seconds": t_fit,
            "score_seconds": t_score,
            "metrics_seconds": t_metrics,
            "anchor_brackets_empirical": protocol_checks["anchor_brackets_empirical"],
            "persistent_trajectory_fpr": persistent_trajectory_fpr,
            "evaluation": evaluation,
        },
    }
    writer.write_result_row(row)
    writer.write_protocol_checks(
        {
            **protocol_checks,
            "method": meta.name,
            "system": system,
            "seed": seed,
            "bif_type": bif_type,
        }
    )
    return row


def _train_bundle(arrays: dict[str, Any]) -> dict[str, Any]:
    """Expose train/val subsets of a bundle to method.fit()."""
    idx = arrays.get("split_indices", {})
    out = {k: v for k, v in arrays.items() if k != "split_indices"}
    out["split_indices"] = {"train": idx.get("train"), "val": idx.get("val")}
    return out


def _flatten_params(meta: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Pull the method's config block (defaults overridden by config)."""
    params: dict[str, Any] = dict(meta.default_params)
    node: Any = config.get("model", {})
    for key in meta.config_path:
        if key not in node:
            return params
        node = node[key]
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "fpr_target":
                continue
            params[k] = v
    return params


__all__ = ["evaluate_method"]
