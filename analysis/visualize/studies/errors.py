"""
Error Analysis & Robust Validation.

Generates:
- Figure 4: Premium Confusion Matrix Heatmap.
- Table 4: Error Analysis Breakdown (False Positives & False Negatives).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from studies.analysis.common.constants import CLASS_DISPLAY_NAMES
from studies.analysis.common.io import read_csv_rows
from studies.analysis.common.latex import format_value, save_latex_table
from studies.analysis.common.style import apply_thesis_style, save_figure

logger = logging.getLogger(__name__)


def _load_confusion_matrix(data_dir: Path) -> tuple[np.ndarray, List[str]]:
    """Loads confusion matrix from CSV into a 12x12 numpy array.
    Returns (matrix, class_labels) where class 0 is Background.
    """
    cm_file = data_dir / "baseline" / "robust_validation" / "confusion_matrix.csv"
    if not cm_file.exists():
        logger.error("Confusion matrix file not found: %s", cm_file)
        return np.zeros((12, 12)), []

    rows = read_csv_rows(cm_file)
    # The columns are: gt_class_id, gt_class_name, pred_background, pred_0..10
    # There are exactly 12 rows (gt=-1 to 10)
    
    # Sort rows by gt_class_id to ensure consistent order: -1, 0, 1... 10
    rows = sorted(rows, key=lambda x: int(x["gt_class_id"]))
    
    matrix = np.zeros((12, 12), dtype=int)
    labels = ["Background"]
    
    # We expect 11 foreground classes
    for i in range(11):
        # We need the key name matching CLASS_DISPLAY_NAMES, but we can just use the gt_class_name from rows
        pass

    for r_idx, row in enumerate(rows):
        gt_id = int(row["gt_class_id"])
        name = row["gt_class_name"]
        if gt_id >= 0:
            labels.append(CLASS_DISPLAY_NAMES.get(name, name.replace("_", " ").title()))
            
        matrix[r_idx, 0] = int(row["pred_background"])
        for c_idx in range(11):
            matrix[r_idx, c_idx + 1] = int(row[f"pred_{c_idx}"])
            
    return matrix, labels


import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

def plot_confusion_matrix(data_dir: Path, output_dir: Path) -> List[Path]:
    """Generate Figure 4: Premium 'Apple-style' confusion matrix with rounded tiles."""
    matrix, labels = _load_confusion_matrix(data_dir)
    if len(labels) == 0:
        return []

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums

    with apply_thesis_style():
        # Apple typography defaults
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif']
        plt.rcParams['axes.linewidth'] = 0.0  # No axes lines

        fig, ax = plt.subplots(figsize=(10.5, 9.0))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        # Custom iOS-like blue gradient for tiles
        apple_blue = LinearSegmentedColormap.from_list(
            "AppleBlue", 
            ["#F2F6FC", "#6AB0F3", "#007AFF", "#004080"]
        )

        n = len(labels)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(n - 0.4, -0.6)  # Inverted y-axis for matrix

        # Draw rounded tiles
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                norm = norm_matrix[i, j]
                
                if val == 0:
                    # Extremely subtle ghost tile for zeros
                    face_color = "#F9F9FB"
                    text_color = "none"
                else:
                    # Map the normalized value (0 to 1) to the colormap
                    face_color = apple_blue(norm)
                    # Contrast check: white text for darker backgrounds, dark gray for light
                    text_color = "#FFFFFF" if norm > 0.35 else "#1D1D1F"

                # Rounded rectangle (tile)
                # We use width/height = 0.86, leaving a 0.14 gap for beautiful whitespace
                box = mpatches.FancyBboxPatch(
                    (j - 0.43, i - 0.43), 0.86, 0.86,
                    boxstyle="round,pad=0.0,rounding_size=0.25",
                    facecolor=face_color, 
                    edgecolor="none",
                    zorder=2
                )
                ax.add_patch(box)

                if val > 0:
                    text_val = f"{val:,}" if val >= 1000 else str(val)
                    weight = "bold" if i == j else "medium"
                    ax.text(
                        j, i, text_val,
                        ha="center", va="center",
                        color=text_color, 
                        fontsize=9, 
                        fontweight=weight,
                        zorder=3
                    )

        # Minimalist Labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color="#555555")
        ax.set_yticklabels(labels, fontsize=9, color="#555555")
        
        # Remove tick marks completely
        ax.tick_params(axis='both', which='both', length=0, pad=8)
        
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", color="#1D1D1F", labelpad=15)
        ax.set_ylabel("True Class", fontsize=11, fontweight="bold", color="#1D1D1F", labelpad=15)
        ax.set_title("Figure 4: Baseline Model Confusion Matrix", fontsize=14, fontweight="bold", color="#1D1D1F", pad=25)

        # Subtle Colorbar floating on the right
        sm = plt.cm.ScalarMappable(cmap=apple_blue, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.06, shrink=0.85, aspect=35)
        
        # Style the colorbar to be minimalist
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0, labelsize=9, colors="#555555", pad=8)
        cbar.set_label("Recall (Normalized)", rotation=270, labelpad=20, fontsize=10, color="#555555")

        # Ensure no spines or grids are visible
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

        fig_path = output_dir / "figure4_confusion_matrix"
        save_figure(fig, fig_path)
        return [fig_path.with_suffix(".pdf"), fig_path.with_suffix(".png")]


def generate_error_tables(data_dir: Path, output_dir: Path) -> Dict[str, Path]:
    """Generate Table 4: Error Analysis Breakdown."""
    matrix, labels = _load_confusion_matrix(data_dir)
    if len(labels) == 0:
        return {}

    # Calculate metrics for each FOREGROUND class (idx 1 to 11)
    # TP: matrix[i, i]
    # FN: sum(matrix[i, :]) - TP
    # FP: sum(matrix[:, i]) - TP
    
    rows = []
    for i in range(1, 12):
        cls_name = labels[i]
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        total_gt = tp + fn
        total_pred = tp + fp
        
        fn_rate = (fn / total_gt) if total_gt > 0 else 0.0
        fp_rate = (fp / total_pred) if total_pred > 0 else 0.0
        
        rows.append({
            "class": cls_name,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "fn_rate": fn_rate,
            "fp_rate": fp_rate,
            "total_gt": total_gt
        })

    # Sort by total errors (FN + FP) descending to highlight worst performers
    rows.sort(key=lambda x: x["fn"] + x["fp"], reverse=True)

    # ── 1. Create LaTeX Table ──
    tex_rows = []
    for r in rows:
        fn_str = f"{r['fn']:,} ({r['fn_rate']*100:.1f}\\%)"
        fp_str = f"{r['fp']:,} ({r['fp_rate']*100:.1f}\\%)"
        tp_str = f"{r['tp']:,}"
        
        # Bold the class if it has very high errors (e.g. > 1000)
        cls_fmt = f"\\textbf{{{r['class']}}}" if (r['fn'] + r['fp']) > 1000 else r['class']
        
        tex_rows.append(
            f"    {cls_fmt} & {tp_str} & {fn_str} & {fp_str} \\\\"
        )

    tex_body = "\n".join(tex_rows)
    tex_str = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Error Analysis Breakdown: False Positives and False Negatives by Class}\n"
        "  \\label{tab:error_analysis}\n"
        "  \\begin{tabular}{lrrr}\n"
        "    \\toprule\n"
        "    \\textbf{Class} & \\textbf{True Positives} & \\textbf{False Negatives (Missed)} & \\textbf{False Positives (Ghost)} \\\\\n"
        "    \\midrule\n"
        f"{tex_body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    tex_path = output_dir / "table4_error_analysis.tex"
    save_latex_table(tex_str, tex_path, standalone_header=True)

    # ── 2. Create Markdown Table ──
    md_rows = []
    for r in rows:
        fn_str = f"{r['fn']:,} ({r['fn_rate']*100:.1f}%)"
        fp_str = f"{r['fp']:,} ({r['fp_rate']*100:.1f}%)"
        tp_str = f"{r['tp']:,}"
        
        cls_fmt = f"**{r['class']}**" if (r['fn'] + r['fp']) > 1000 else r['class']
        md_rows.append(f"| {cls_fmt} | {tp_str} | {fn_str} | {fp_str} |")

    md_body = "\n".join(md_rows)
    md_str = (
        "# Table 4: Error Analysis Breakdown\n\n"
        "| Class | True Positives | False Negatives (Missed) | False Positives (Ghost) |\n"
        "| :--- | ---: | ---: | ---: |\n"
        f"{md_body}\n"
    )

    md_path = output_dir / "table4_error_analysis.md"
    md_path.write_text(md_str, encoding="utf-8")

    return {"tex": tex_path, "md": md_path}
