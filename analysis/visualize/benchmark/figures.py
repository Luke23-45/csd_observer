"""Figure generation for the CSD observer benchmark."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.visualize.benchmark.data import BenchmarkStore
from analysis.visualize.common.constants import METHODS, METHOD_COLORS, PATIENT_COUNTS, SUPPLEMENTAL_COLORS, SYSTEMS, SYSTEM_COLORS, SYSTEM_DISPLAY_NAMES, METHOD_DISPLAY_NAMES
from analysis.visualize.common.style import apply_thesis_style, save_figure

logger = logging.getLogger(__name__)


def _aggregate(records: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return (
        records.groupby(group_cols, observed=True)
        .agg(
            detection_time_mean=("detection_time", "mean"),
            detection_time_std=("detection_time", "std"),
            ew_auc_mean=("ew_auc", "mean"),
            ew_auc_std=("ew_auc", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
        )
        .reset_index()
    )


def _method_kwargs(method: str) -> Dict[str, str]:
    color = METHOD_COLORS[method]
    if method == "Kalman-Spectral-Drift":
        return {"color": color, "marker": "o", "linestyle": "-", "linewidth": 2.0}
    return {"color": color, "marker": "s", "linestyle": "--", "linewidth": 2.0}


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _representative_seed(store: BenchmarkStore, patient_count: int, system: str) -> int:
    return store.representative_seed(patient_count, system, anchor_method="Kalman-Spectral-Drift")


def _safe_nanmedian(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def plot_patient_sweep(store: BenchmarkStore, output_dir: Path) -> List[Path]:
    """Plot mean +/- std across patient counts for all systems and metrics."""
    summary = _aggregate(store.records, ["patient_count", "system", "method"])
    metrics = [
        ("detection_time", "Detection Time"),
        ("ew_auc", "EW-AUC"),
        ("fpr", "FPR"),
    ]

    with apply_thesis_style():
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
        plt.rcParams["axes.linewidth"] = 1.0
        plt.rcParams["axes.edgecolor"] = "black"
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5), sharex=True, constrained_layout=False)

        for row_idx, system in enumerate(SYSTEMS):
            system_name = SYSTEM_DISPLAY_NAMES.get(system, system)
            df_sys = summary[summary["system"].astype(str) == system]
            for col_idx, (metric, metric_label) in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                for method in METHODS:
                    df = df_sys[df_sys["method"].astype(str) == method].sort_values("patient_count")
                    if df.empty:
                        continue
                    y = df[f"{metric}_mean"].to_numpy(dtype=float)
                    yerr = df[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
                    x = df["patient_count"].to_numpy(dtype=float)
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        capsize=3,
                        **_method_kwargs(method),
                        label=METHOD_DISPLAY_NAMES.get(method, method) if (row_idx == 0 and col_idx == 0) else None,
                    )
                if row_idx == 0:
                    ax.set_title(metric_label, fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(system_name, fontsize=11, fontweight="bold", labelpad=22)
                if row_idx == 2:
                    ax.set_xlabel("Patients", fontsize=10)
                ax.set_xticks(list(PATIENT_COUNTS))
                _clean_axis(ax)
                if metric == "ew_auc":
                    ax.set_ylim(0.0, 1.05)
                elif metric == "fpr":
                    ax.set_ylim(bottom=0.0)
                else:
                    ax.set_ylim(bottom=0.0)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 1.01),
            )
        fig.suptitle("Patient-Depth Response Across Systems", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        fig_path = output_dir / "figure1_patient_sweep"
        save_figure(fig, fig_path)
        return [fig_path.with_suffix(".pdf"), fig_path.with_suffix(".png")]


def plot_trajectory_panels(store: BenchmarkStore, output_dir: Path, patient_count: int = 500) -> List[Path]:
    """Plot representative trajectory summaries for each system and method."""
    fig, axes = None, None
    n_methods = len(METHODS)
    n_cols = int(np.ceil(n_methods / len(SYSTEMS)))
    with apply_thesis_style():
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
        plt.rcParams["axes.linewidth"] = 1.0
        plt.rcParams["axes.edgecolor"] = "black"
        fig, axes = plt.subplots(len(SYSTEMS), n_cols, figsize=(5.0 * n_cols, 9.0), sharex=True, sharey=True, constrained_layout=False)

        for row_idx, system in enumerate(SYSTEMS):
            seed = _representative_seed(store, patient_count, system)
            for col_idx, method in enumerate(METHODS):
                ax = axes[row_idx, col_idx]
                traj = store.load_trajectory(patient_count, system, method, seed)
                probs = np.asarray(traj["probs_test"], dtype=float)
                med = np.nanmedian(probs, axis=0)
                q1 = np.nanpercentile(probs, 25, axis=0)
                q3 = np.nanpercentile(probs, 75, axis=0)
                threshold = float(np.asarray(traj["threshold"]).reshape(-1)[0])
                bif = _safe_nanmedian(np.asarray(traj["bifurcation_times"], dtype=float))
                det = _safe_nanmedian(np.asarray(traj["detection_times"], dtype=float))

                x = np.arange(med.shape[0], dtype=float)
                ax.axvspan(0, bif, color=SYSTEM_COLORS[system], alpha=0.04, zorder=0)
                ax.fill_between(x, q1, q3, color=METHOD_COLORS[method], alpha=0.18, linewidth=0)
                ax.plot(x, med, color=METHOD_COLORS[method], linewidth=2.0, label="Median score")
                ax.axhline(threshold, color=SUPPLEMENTAL_COLORS["dark"], linestyle="--", linewidth=1.0, alpha=0.75, label="Threshold")
                ax.axvline(bif, color=SYSTEM_COLORS[system], linestyle=":", linewidth=1.2, alpha=0.9, label="Bifurcation")
                if np.isfinite(det):
                    ax.axvline(det, color=METHOD_COLORS[method], linestyle="-.", linewidth=1.0, alpha=0.75, label="Detection time")

                if row_idx == 0:
                    ax.set_title(METHOD_DISPLAY_NAMES[method], fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{SYSTEM_DISPLAY_NAMES[system]}\nProbability", fontsize=10, fontweight="bold")
                if row_idx == 2:
                    ax.set_xlabel("Time step", fontsize=10)
                ax.set_ylim(-0.02, 1.05)
                _clean_axis(ax)
                ax.text(
                    0.03,
                    0.93,
                    f"seed {seed}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color=SUPPLEMENTAL_COLORS["dark"],
                    va="top",
                )
                ax.text(
                    0.03,
                    0.08,
                    f"thr={threshold:.3f}  det={det:.1f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color=SUPPLEMENTAL_COLORS["dark"],
                    va="bottom",
                )

        for col_idx in range(len(METHODS), n_cols):
            for row_idx in range(len(SYSTEMS)):
                axes[row_idx, col_idx].set_visible(False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=4,
                frameon=False,
                bbox_to_anchor=(0.5, 1.01),
            )
        fig.suptitle(f"Representative Score Trajectories at {patient_count} Patients", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        fig_path = output_dir / "figure2_trajectory_panels"
        save_figure(fig, fig_path)
        return [fig_path.with_suffix(".pdf"), fig_path.with_suffix(".png")]


def plot_training_curves(store: BenchmarkStore, output_dir: Path, patient_count: int = 500) -> List[Path]:
    """Plot training loss and validation metric curves for representative runs.

    Only methods with epoch logs are plotted; the current benchmark suite
    has no training loop, so a missing epoch log for a method simply skips
    it, and the figure is skipped entirely when none are available.
    """
    with apply_thesis_style():
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
        plt.rcParams["axes.linewidth"] = 1.0
        plt.rcParams["axes.edgecolor"] = "black"
        fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True, constrained_layout=False)

        plotted_any = False
        for row_idx, system in enumerate(SYSTEMS):
            ax = axes[row_idx]
            seed = _representative_seed(store, patient_count, system)
            for method in METHODS:
                try:
                    log = store.load_epoch_log(patient_count, system, method, seed)
                except FileNotFoundError:
                    continue
                if log.empty:
                    continue
                plotted_any = True
                epochs = log["epoch"].to_numpy(dtype=float)
                train_loss = log["train_loss"].to_numpy(dtype=float)
                val_metric = log["val_metric"].to_numpy(dtype=float)

                ax.plot(
                    epochs,
                    train_loss,
                    color=METHOD_COLORS[method],
                    linewidth=2.0,
                    label=f"{METHOD_DISPLAY_NAMES[method]} train" if row_idx == 0 else None,
                )
                ax.plot(
                    epochs,
                    val_metric,
                    color=METHOD_COLORS[method],
                    linewidth=2.0,
                    linestyle="--",
                    label=f"{METHOD_DISPLAY_NAMES[method]} val" if row_idx == 0 else None,
                )

            ax.set_ylabel(f"{SYSTEM_DISPLAY_NAMES[system]}\nLoss / Metric", fontsize=10, fontweight="bold")
            _clean_axis(ax)

        if not plotted_any:
            plt.close(fig)
            logger.warning("No epoch logs found in the selected batches; skipping training-curve figure.")
            return []

        axes[-1].set_xlabel("Epoch", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(f"Representative Training Curves at {patient_count} Patients", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        fig_path = output_dir / "figure3_training_curves"
        save_figure(fig, fig_path)
        return [fig_path.with_suffix(".pdf"), fig_path.with_suffix(".png")]
