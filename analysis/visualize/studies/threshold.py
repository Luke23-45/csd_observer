"""
Confidence Threshold Sweep & Operational Boundary Analysis.

Generates:
- Figure 2: Confidence Threshold Sweep Curves (publication-grade single-axis plot).
- Table 2: Confidence Threshold Sweep Metrics LaTeX + Markdown tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import make_interp_spline

from studies.analysis.common.constants import MODEL_COLORS, METRIC_LABELS
from studies.analysis.common.io import read_csv_rows, load_json
from studies.analysis.common.latex import format_value, save_latex_table
from studies.analysis.common.style import apply_thesis_style, create_figure, save_figure, PALETTE

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────


def _smooth_interp(x: List[float], y: List[float], num_pts: int = 300) -> tuple:
    """Cubic B-spline interpolation for smooth curve rendering.

    Returns (x_smooth, y_smooth) arrays with `num_pts` points.
    Falls back to raw data if interpolation fails.
    """
    xa, ya = np.array(x), np.array(y)
    try:
        spl = make_interp_spline(xa, ya, k=3)
        x_sm = np.linspace(xa.min(), xa.max(), num_pts)
        y_sm = spl(x_sm)
        return x_sm, y_sm
    except Exception:
        return xa, ya


# ── Figure 2: Threshold Sweep ──────────────────────────────────────


def plot_threshold_sweep(data_dir: Path, output_dir: Path) -> List[Path]:
    """Generate Figure 2: Premium confidence threshold sweep curves."""
    sweep_file = data_dir / "baseline" / "threshold" / "threshold_sweep.csv"
    optimal_file = data_dir / "baseline" / "threshold" / "optimal_thresholds.json"

    if not sweep_file.exists():
        logger.error("Threshold sweep file missing: %s", sweep_file)
        return []

    rows = read_csv_rows(sweep_file)
    thresholds = [float(r["confidence_threshold"]) for r in rows]
    precision  = [float(r["precision"]) for r in rows]
    recall     = [float(r["recall"]) for r in rows]
    f1_score   = [float(r["f1_score"]) for r in rows]

    # ── Optimal threshold from JSON or fallback ─────────────────────
    opt_thresh = 0.10
    opt_f1 = 0.7362
    if optimal_file.exists():
        opt_data = load_json(optimal_file)
        opt_thresh = float(opt_data["optimal_f1"]["optimal_threshold"])
        opt_f1 = float(opt_data["optimal_f1"]["value"])

    # ── Find key data points ────────────────────────────────────────
    # Precision & Recall at optimal threshold
    opt_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - opt_thresh))
    opt_prec = precision[opt_idx]
    opt_rec  = recall[opt_idx]

    # Default YOLO threshold (0.25)
    def_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.25))
    def_f1 = f1_score[def_idx]

    # ── Smooth interpolation for silky curves ───────────────────────
    t_sm_p, p_sm = _smooth_interp(thresholds, precision)
    t_sm_r, r_sm = _smooth_interp(thresholds, recall)
    t_sm_f, f_sm = _smooth_interp(thresholds, f1_score)

    with apply_thesis_style():
        # Publication-ready sans-serif overrides
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0

        fig, ax = plt.subplots(figsize=(9.0, 5.0))

        # ── Cohesive colour palette (matching Figures 1 & 3) ────────
        c_prec = '#2f4f4f'   # Dark Slate Grey
        c_rec  = '#4682b4'   # Steel Blue
        c_f1   = '#2e8b57'   # Sea Green
        c_opt  = '#b22222'   # Firebrick Red (for emphasis)

        # ── Shaded operational zones ────────────────────────────────
        # Green zone: optimal operating range
        ax.axvspan(0.05, 0.20, color=c_f1, alpha=0.04, zorder=0)
        # Red zone: over-confident (high threshold, low recall)
        ax.axvspan(0.75, 0.95, color=c_opt, alpha=0.04, zorder=0)

        # ── Fill between Precision & Recall (precision-recall gap) ──
        ax.fill_between(t_sm_p, p_sm, r_sm, alpha=0.06, color=c_rec, zorder=1)

        # ── Smooth curves ───────────────────────────────────────────
        ax.plot(t_sm_p, p_sm, color=c_prec, lw=2.2, label="Precision", zorder=3)
        ax.plot(t_sm_r, r_sm, color=c_rec,  lw=2.2, label="Recall", ls="--", zorder=3)
        ax.plot(t_sm_f, f_sm, color=c_f1,   lw=2.8, label="F1-Score", zorder=4)

        # ── Raw data points (subtle markers) ────────────────────────
        ax.scatter(thresholds, precision, color=c_prec, s=20, zorder=5,
                   edgecolors='white', linewidths=0.8, alpha=0.7)
        ax.scatter(thresholds, recall, color=c_rec, s=20, zorder=5,
                   edgecolors='white', linewidths=0.8, alpha=0.7)
        ax.scatter(thresholds, f1_score, color=c_f1, s=25, zorder=5,
                   edgecolors='white', linewidths=0.8, alpha=0.8)

        # ── Optimal F1 marker + annotation ──────────────────────────
        ax.axvline(opt_thresh, color=c_f1, ls=":", lw=1.3, alpha=0.6, zorder=2)
        ax.scatter([opt_thresh], [opt_f1], color=c_f1, s=100, zorder=6,
                   edgecolors='white', linewidths=2.0, marker='D')
        ax.annotate(
            f"Optimal F1 = {opt_f1:.3f}\nThreshold = {opt_thresh:.2f}",
            xy=(opt_thresh, opt_f1),
            xytext=(opt_thresh + 0.12, opt_f1 + 0.18),
            fontsize=9,
            fontweight="bold",
            color='#333333',
            arrowprops=dict(arrowstyle="->", color='#555555', lw=1.2,
                            connectionstyle="arc3,rad=-0.15"),
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=c_f1,
                      lw=1.0, alpha=0.95),
            zorder=7,
        )

        # ── Default YOLO threshold marker ───────────────────────────
        ax.axvline(0.25, color='#888888', ls=":", lw=1.0, alpha=0.5, zorder=2)
        ax.scatter([0.25], [def_f1], color='#888888', s=60, zorder=6,
                   edgecolors='white', linewidths=1.5, marker='s')
        ax.annotate(
            f"Default (0.25)\nF1 = {def_f1:.3f}",
            xy=(0.25, def_f1),
            xytext=(0.38, def_f1 + 0.22),
            fontsize=8,
            color='#555555',
            arrowprops=dict(arrowstyle="->", color='#888888', lw=1.0,
                            connectionstyle="arc3,rad=-0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='#cccccc',
                      lw=0.8, alpha=0.9),
            zorder=7,
        )

        # ── Zone labels (subtle) ───────────────────────────────────
        ax.text(0.125, 0.03, "Optimal\nOperating Zone", ha="center", va="bottom",
                fontsize=7.5, color=c_f1, alpha=0.6, fontstyle="italic",
                transform=ax.get_xaxis_transform())
        ax.text(0.85, 0.03, "Over-Confident\nZone", ha="center", va="bottom",
                fontsize=7.5, color=c_opt, alpha=0.5, fontstyle="italic",
                transform=ax.get_xaxis_transform())

        # ── Axes formatting ─────────────────────────────────────────
        ax.set_xlabel("Confidence Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xlim(0.03, 0.97)
        ax.set_ylim(0.0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.tick_params(labelsize=9)

        # Grid
        ax.grid(True, which='major', linestyle='-', color='gray', alpha=0.15, linewidth=0.5)
        ax.grid(True, which='minor', linestyle=':', color='gray', alpha=0.08, linewidth=0.3)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── Legend (horizontal, top) ────────────────────────────────
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=10,
            columnspacing=2.5,
            handlelength=2.5,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        fig_path = output_dir / "figure2_threshold_sweep"
        save_figure(fig, fig_path)
        return [fig_path.with_suffix(".pdf"), fig_path.with_suffix(".png")]


# ── Table 2: Threshold Sweep ──────────────────────────────────────


def generate_threshold_tables(data_dir: Path, output_dir: Path) -> Dict[str, Path]:
    """Generate Table 2: Threshold sweep metrics table in LaTeX and Markdown."""
    sweep_file = data_dir / "baseline" / "threshold" / "threshold_sweep.csv"
    if not sweep_file.exists():
        logger.error("Threshold sweep CSV missing for table generation")
        return {}

    rows = read_csv_rows(sweep_file)
    
    # ── 1. Create LaTeX Table ──
    tex_rows = []
    for r in rows:
        thresh = float(r["confidence_threshold"])
        prec = float(r["precision"])
        rec = float(r["recall"])
        f1 = float(r["f1_score"])
        m50 = float(r["mAP_50"])
        m95 = float(r["mAP_50_95"])
        fit = float(r["fitness"])
        
        # Highlight threshold 0.10 (operational best)
        if abs(thresh - 0.10) < 1e-4:
            row_str = (
                f"    \\textbf{{{thresh:.2f}}} & "
                f"\\textbf{{{prec:.4f}}} & "
                f"\\textbf{{{rec:.4f}}} & "
                f"\\textbf{{{f1:.4f}}} & "
                f"\\textbf{{{m50:.4f}}} & "
                f"\\textbf{{{m95:.4f}}} & "
                f"\\textbf{{{fit:.4f}}} \\\\"
            )
        else:
            row_str = (
                f"    {thresh:.2f} & "
                f"{prec:.4f} & "
                f"{rec:.4f} & "
                f"{f1:.4f} & "
                f"{m50:.4f} & "
                f"{m95:.4f} & "
                f"{fit:.4f} \\\\"
            )
        tex_rows.append(row_str)

    tex_body = "\n".join(tex_rows)
    tex_str = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Sensitivity Sweep of Confidence Thresholds on Krishi YOLOv11 Baseline Model}\n"
        "  \\label{tab:threshold_sensitivity_sweep}\n"
        "  \\begin{tabular}{ccccccc}\n"
        "    \\toprule\n"
        "    \\textbf{Confidence Thresh} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{mAP@50} & \\textbf{mAP@50-95} & \\textbf{Fitness} \\\\\n"
        "    \\midrule\n"
        f"{tex_body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    tex_path = output_dir / "table2_threshold_sweep.tex"
    save_latex_table(tex_str, tex_path, standalone_header=True)

    # ── 2. Create Markdown Table ──
    md_rows = []
    for r in rows:
        thresh = float(r["confidence_threshold"])
        prec = float(r["precision"])
        rec = float(r["recall"])
        f1 = float(r["f1_score"])
        m50 = float(r["mAP_50"])
        m95 = float(r["mAP_50_95"])
        fit = float(r["fitness"])
        
        if abs(thresh - 0.10) < 1e-4:
            row_str = f"| **{thresh:.2f} (Optimal)** | **{prec:.4f}** | **{rec:.4f}** | **{f1:.4f}** | **{m50:.4f}** | **{m95:.4f}** | **{fit:.4f}** |"
        else:
            row_str = f"| {thresh:.2f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {m50:.4f} | {m95:.4f} | {fit:.4f} |"
        md_rows.append(row_str)
        
    md_body = "\n".join(md_rows)
    md_str = (
        "# Table 2: Confidence Threshold Sensitivity Sweep Performance\n\n"
        "| Confidence Thresh | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 | Fitness |\n"
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        f"{md_body}\n"
    )
    
    md_path = output_dir / "table2_threshold_sweep.md"
    md_path.write_text(md_str, encoding="utf-8")

    return {"tex": tex_path, "md": md_path}
