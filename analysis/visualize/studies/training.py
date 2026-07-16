"""
YOLO Training Dynamics & Best Epoch performance analysis.

Generates:
- Figure 1: Training Loss & Metric Curves over 100 epochs (2x2 premium layout).
- Table 1: Best Epoch & Training Summary LaTeX + Markdown tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from studies.analysis.common.constants import MODEL_COLORS, METRIC_LABELS
from studies.analysis.common.io import read_csv_rows, load_json
from studies.analysis.common.latex import format_value, save_latex_table
from studies.analysis.common.style import apply_thesis_style, create_figure, save_figure, PALETTE

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────


def _ema_smooth(values: List[float], alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average smoother for noisy metric curves.

    Parameters
    ----------
    values : list of float
        Raw data points.
    alpha : float
        Smoothing factor (0 = heavy smooth, 1 = no smooth).
    """
    arr = np.array(values, dtype=np.float64)
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


# ── Figure 1: 2×2 Training Dynamics Dashboard ──────────────────────


def plot_training_curves(data_dir: Path, output_dir: Path) -> List[Path]:
    """Generate Figure 1: 2×2 training dynamics dashboard.

    Layout:
        (a) Bounding-Box Regression Loss   │ (b) Precision & Recall
        ────────────────────────────────────┼────────────────────────
        (c) Classification & DFL Losses     │ (d) Mean Average Precision
    """
    metrics_file = data_dir / "thesis_data" / "metrics" / "thesis_metrics.csv"
    if not metrics_file.exists():
        logger.error("Training metrics file not found: %s", metrics_file)
        return []

    rows = read_csv_rows(metrics_file)
    epochs = [float(r["epoch"]) for r in rows]

    # ── Extract raw series ──────────────────────────────────────────
    train_box = [float(r["train/box_loss"]) for r in rows]
    val_box   = [float(r["val/box_loss"]) for r in rows]
    train_cls = [float(r["train/cls_loss"]) for r in rows]
    val_cls   = [float(r["val/cls_loss"]) for r in rows]
    train_dfl = [float(r["train/dfl_loss"]) for r in rows]
    val_dfl   = [float(r["val/dfl_loss"]) for r in rows]

    precision = [float(r["metrics/precision(B)"]) for r in rows]
    recall    = [float(r["metrics/recall(B)"]) for r in rows]
    map50     = [float(r["metrics/mAP50(B)"]) for r in rows]
    map95     = [float(r["metrics/mAP50-95(B)"]) for r in rows]

    # ── EMA-smoothed series for overlay ─────────────────────────────
    sm_train_box = _ema_smooth(train_box, alpha=0.2)
    sm_val_box   = _ema_smooth(val_box,   alpha=0.2)
    sm_train_cls = _ema_smooth(train_cls, alpha=0.2)
    sm_val_cls   = _ema_smooth(val_cls,   alpha=0.2)
    sm_train_dfl = _ema_smooth(train_dfl, alpha=0.2)
    sm_val_dfl   = _ema_smooth(val_dfl,   alpha=0.2)

    sm_prec   = _ema_smooth(precision, alpha=0.15)
    sm_recall = _ema_smooth(recall,    alpha=0.15)
    sm_map50  = _ema_smooth(map50,     alpha=0.15)
    sm_map95  = _ema_smooth(map95,     alpha=0.15)

    # ── Identify best epoch (epoch with highest mAP@50) ─────────────
    best_idx   = int(np.argmax(map50))
    best_epoch = epochs[best_idx]

    with apply_thesis_style():
        # Publication-ready sans-serif overrides (match Figures 2 & 3)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['lines.linewidth'] = 2.0

        fig, axes = plt.subplots(
            2, 2,
            figsize=(13.0, 7.5),
        )

        # ── Cohesive colour palette (matching Figures 2 & 3) ────────
        c_train   = '#2f4f4f'   # Dark Slate Grey  (primary train)
        c_val     = '#b22222'   # Firebrick Red     (primary val)
        c_train2  = '#4682b4'   # Steel Blue        (secondary train)
        c_val2    = '#d2691e'   # Chocolate Orange   (secondary val)
        c_prec    = '#2f4f4f'   # Dark Slate Grey
        c_rec     = '#4682b4'   # Steel Blue
        c_map50   = '#2e8b57'   # Sea Green
        c_map95   = '#b22222'   # Firebrick Red

        raw_alpha  = 0.25       # faint raw data
        smooth_lw  = 2.2        # smooth line width

        # ────────────────────────────────────────────────────────────
        # (a) Top-Left: Bounding-Box Regression Loss
        # ────────────────────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(epochs, train_box, color=c_train, alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, val_box,   color=c_val,   alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, sm_train_box, color=c_train, lw=smooth_lw, label="Train")
        ax.plot(epochs, sm_val_box,   color=c_val,   lw=smooth_lw, ls="--", label="Val")
        # Subtle fill between train and val
        ax.fill_between(epochs, sm_train_box, sm_val_box, color=c_train, alpha=0.06)
        ax.set_ylabel("Box Loss", fontsize=11)
        ax.set_title("(a) Bounding-Box Regression Loss", fontsize=11, fontweight="bold", loc="left")
        ax.tick_params(labelbottom=False, labelsize=9)
        ax.legend(loc="upper right", frameon=False, fontsize=9)

        # ────────────────────────────────────────────────────────────
        # (b) Top-Right: Precision & Recall
        # ────────────────────────────────────────────────────────────
        ax = axes[0, 1]
        ax.plot(epochs, precision, color=c_prec, alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, recall,    color=c_rec,  alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, sm_prec,   color=c_prec, lw=smooth_lw, label="Precision")
        ax.plot(epochs, sm_recall, color=c_rec,  lw=smooth_lw, ls="--", label="Recall")
        ax.fill_between(epochs, sm_prec, sm_recall, color=c_rec, alpha=0.06)
        # Best-epoch marker
        ax.axvline(best_epoch, color='#333333', ls=":", lw=1.0, alpha=0.5)
        ax.scatter([best_epoch], [sm_prec[best_idx]], color=c_prec, s=50, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.scatter([best_epoch], [sm_recall[best_idx]], color=c_rec, s=50, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("(b) Detection Precision & Recall", fontsize=11, fontweight="bold", loc="left")
        ax.tick_params(labelbottom=False, labelsize=9)
        ax.legend(loc="lower right", frameon=False, fontsize=9)

        # ────────────────────────────────────────────────────────────
        # (c) Bottom-Left: Classification & DFL Losses
        # ────────────────────────────────────────────────────────────
        ax = axes[1, 0]
        # Raw (faint)
        ax.plot(epochs, train_cls, color=c_train,  alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, val_cls,   color=c_val,    alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, train_dfl, color=c_train2, alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, val_dfl,   color=c_val2,   alpha=raw_alpha, lw=1.0)
        # Smooth
        ax.plot(epochs, sm_train_cls, color=c_train,  lw=smooth_lw, label="Train Cls")
        ax.plot(epochs, sm_val_cls,   color=c_val,    lw=smooth_lw, ls="--", label="Val Cls")
        ax.plot(epochs, sm_train_dfl, color=c_train2, lw=smooth_lw, label="Train DFL")
        ax.plot(epochs, sm_val_dfl,   color=c_val2,   lw=smooth_lw, ls="--", label="Val DFL")
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title("(c) Classification & DFL Losses", fontsize=11, fontweight="bold", loc="left")
        ax.tick_params(labelsize=9)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)

        # ────────────────────────────────────────────────────────────
        # (d) Bottom-Right: mAP@50 & mAP@50-95
        # ────────────────────────────────────────────────────────────
        ax = axes[1, 1]
        ax.plot(epochs, map50, color=c_map50, alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, map95, color=c_map95, alpha=raw_alpha, lw=1.0)
        ax.plot(epochs, sm_map50, color=c_map50, lw=smooth_lw, label="mAP@50")
        ax.plot(epochs, sm_map95, color=c_map95, lw=smooth_lw, ls="--", label="mAP@50-95")
        ax.fill_between(epochs, sm_map50, sm_map95, color=c_map50, alpha=0.08)
        # Best-epoch annotation
        ax.axvline(best_epoch, color='#333333', ls=":", lw=1.0, alpha=0.5)
        ax.scatter([best_epoch], [sm_map50[best_idx]], color=c_map50, s=60, zorder=5,
                   edgecolors='white', linewidths=1.5, marker='D')
        ax.annotate(
            f"Best Epoch {int(best_epoch)}",
            xy=(best_epoch, sm_map50[best_idx]),
            xytext=(-55, 18),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color='#333333',
            arrowprops=dict(arrowstyle="->", color='#333333', lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8, alpha=0.9),
        )
        ax.set_ylabel("mAP Score", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title("(d) Mean Average Precision (mAP)", fontsize=11, fontweight="bold", loc="left")
        ax.tick_params(labelsize=9)
        ax.legend(loc="lower right", frameon=False, fontsize=9)

        # ── Global formatting across all panels ─────────────────────
        for ax in axes.flat:
            ax.set_xlim(1, 100)
            ax.grid(True, linestyle='-', color='gray', alpha=0.15, linewidth=0.5)
            # Remove top & right spines for clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')

        fig.tight_layout(pad=2.0, w_pad=2.5, h_pad=2.5)

        fig_path = output_dir / "figure1_training_dynamics"
        save_figure(fig, fig_path)
        return [fig_path.with_suffix(".pdf"), fig_path.with_suffix(".png")]

def generate_training_tables(data_dir: Path, output_dir: Path) -> Dict[str, Path]:
    """Generate Table 1: Best epoch metrics LaTeX & Markdown tables."""
    best_epoch_file = data_dir / "thesis_data" / "metrics" / "best_epoch_metrics.json"
    summary_file = data_dir / "thesis_data" / "metrics" / "training_summary.json"
    
    if not best_epoch_file.exists() or not summary_file.exists():
        logger.error("Training summary files missing")
        return {}

    best_data = load_json(best_epoch_file)
    summary_data = load_json(summary_file)

    # ── 1. Create LaTeX Table ──
    tex_str = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Summary of Training Dynamics and Best-Epoch Performance for Krishi YOLOv11}\n"
        "  \\label{tab:training_dynamics_summary}\n"
        "  \\begin{tabular}{lr}\n"
        "    \\toprule\n"
        "    \\textbf{Metric / Parameter} & \\textbf{Value} \\\\\n"
        "    \\midrule\n"
        f"    Total Epochs Trained & ${int(summary_data['total_epochs_trained'])}$ \\\\\n"
        f"    Best Training Epoch & ${int(best_data['epoch'])}$ \\\\\n"
        f"    Input Resolution (pixels) & ${int(summary_data['image_size'])} \\times {int(summary_data['image_size'])}$ \\\\\n"
        f"    Total Training Time & ${format_value(summary_data['total_training_time_seconds'] / 3600.0, precision=2)}\\,\\text{{hours}}$ \\\\\n"
        "    \\midrule\n"
        "    \\multicolumn{2}{l}{\\textit{Loss Residuals at Best Epoch}} \\\\\n"
        f"    ~~Train Box Loss & {format_value(best_data['train/box_loss'], precision=4, use_math=True)} \\\\\n"
        f"    ~~Train Class Loss & {format_value(best_data['train/cls_loss'], precision=4, use_math=True)} \\\\\n"
        f"    ~~Train DFL Loss & {format_value(best_data['train/dfl_loss'], precision=4, use_math=True)} \\\\\n"
        f"    ~~Val Box Loss & {format_value(best_data['val/box_loss'], precision=4, use_math=True)} \\\\\n"
        f"    ~~Val Class Loss & {format_value(best_data['val/cls_loss'], precision=4, use_math=True)} \\\\\n"
        f"    ~~Val DFL Loss & {format_value(best_data['val/dfl_loss'], precision=4, use_math=True)} \\\\\n"
        "    \\midrule\n"
        "    \\multicolumn{2}{l}{\\textit{Detection Accuracy at Best Epoch}} \\\\\n"
        f"    ~~Precision & {format_value(best_data['metrics/precision(B)'], precision=4, use_math=True)} \\\\\n"
        f"    ~~Recall & {format_value(best_data['metrics/recall(B)'], precision=4, use_math=True)} \\\\\n"
        f"    ~~F1-Score & {format_value(summary_data['best_f1'], precision=4, use_math=True)} \\\\\n"
        f"    ~~mAP@50 & {format_value(best_data['metrics/mAP50(B)'], precision=4, use_math=True)} \\\\\n"
        f"    ~~mAP@50-95 & {format_value(best_data['metrics/mAP50-95(B)'], precision=4, use_math=True)} \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    tex_path = output_dir / "table1_training_dynamics.tex"
    save_latex_table(tex_str, tex_path, standalone_header=True)

    # ── 2. Create Markdown Table ──
    md_str = (
        "# Table 1: Krishi YOLOv11 Training Dynamics Summary\n\n"
        "| Metric / Parameter | Value |\n"
        "| :--- | :--- |\n"
        f"| **Total Epochs Trained** | {int(summary_data['total_epochs_trained'])} |\n"
        f"| **Best Training Epoch** | {int(best_data['epoch'])} |\n"
        f"| **Input Resolution (pixels)** | {int(summary_data['image_size'])}x{int(summary_data['image_size'])} |\n"
        f"| **Total Training Time** | {summary_data['total_training_time_seconds'] / 3600.0:.2f} hours ({summary_data['total_training_time_seconds']:.1f} s) |\n"
        "| **Train Box Loss** | " + f"{best_data['train/box_loss']:.4f} |\n"
        "| **Train Class Loss** | " + f"{best_data['train/cls_loss']:.4f} |\n"
        "| **Train DFL Loss** | " + f"{best_data['train/dfl_loss']:.4f} |\n"
        "| **Val Box Loss** | " + f"{best_data['val/box_loss']:.4f} |\n"
        "| **Val Class Loss** | " + f"{best_data['val/cls_loss']:.4f} |\n"
        "| **Val DFL Loss** | " + f"{best_data['val/dfl_loss']:.4f} |\n"
        "| **Precision (Best)** | " + f"{best_data['metrics/precision(B)']:.4f} |\n"
        "| **Recall (Best)** | " + f"{best_data['metrics/recall(B)']:.4f} |\n"
        "| **F1-Score (Best)** | " + f"{summary_data['best_f1']:.4f} |\n"
        "| **mAP@50 (Best)** | " + f"{best_data['metrics/mAP50(B)']:.4f} |\n"
        "| **mAP@50-95 (Best)** | " + f"{best_data['metrics/mAP50-95(B)']:.4f} |\n"
    )
    md_path = output_dir / "table1_training_dynamics.md"
    md_path.write_text(md_str, encoding="utf-8")
    
    return {"tex": tex_path, "md": md_path}
