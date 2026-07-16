"""
Quantization & Deployment Efficiency Analysis.

Generates:
- Figure 3: Three-panel deployment comparison (Model Size, Latency, Detection F1).
- Table 3a: Overall deployment summary (LaTeX + Markdown).
- Table 3b: Per-class F1-score breakdown across all model variants (LaTeX + Markdown).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from studies.analysis.common.constants import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_SIZES_BYTES,
    MODEL_VARIANTS,
)
from studies.analysis.common.io import load_json
from studies.analysis.common.latex import format_value, latex_escape, save_latex_table
from studies.analysis.common.style import (
    PALETTE,
    apply_thesis_style,
    create_figure,
    save_figure,
)

logger = logging.getLogger(__name__)


# ── Internal Data Loaders ───────────────────────────────────────────


def _variant_data_dir(data_root: Path, variant: str) -> Path:
    """Resolve the filesystem path for a specific model variant."""
    if variant == "baseline":
        return data_root / "baseline"
    return data_root / "quantized" / variant


def _load_latency_per_image(data_root: Path, variant: str) -> np.ndarray:
    """Load per-image latency values (ms) from CSV for a given variant."""
    csv_path = _variant_data_dir(data_root, variant) / "latency" / "latency_per_image.csv"
    if not csv_path.exists():
        logger.warning("Latency CSV missing for %s: %s", variant, csv_path)
        return np.array([])
    df = pd.read_csv(csv_path)
    return df["latency_ms"].values


def _load_latency_summary(data_root: Path, variant: str) -> Dict:
    """Load latency summary JSON for a given variant."""
    json_path = _variant_data_dir(data_root, variant) / "latency" / "latency_metrics.json"
    if not json_path.exists():
        logger.warning("Latency JSON missing for %s: %s", variant, json_path)
        return {}
    return load_json(json_path)


def _load_validation_summary(data_root: Path, variant: str) -> Dict:
    """Load robust validation summary JSON for a given variant."""
    json_path = _variant_data_dir(data_root, variant) / "robust_validation" / "summary.json"
    if not json_path.exists():
        logger.warning("Validation summary missing for %s: %s", variant, json_path)
        return {}
    return load_json(json_path)


# ── Figure 3: Three-Panel Deployment Comparison ────────────────────


def plot_quantization_comparison(data_dir: Path, output_dir: Path) -> List[Path]:
    """Generate Figure 3: 1×3 panel comparing model variants on Size, Latency, and Detection F1.

    Panel A: Horizontal bar chart of model sizes (MB) with compression ratio annotations.
    Panel B: Violin + strip overlay of per-image latency distributions.
    Panel C: Grouped bar chart of overall Precision, Recall, F1-score.
    """

    # ── Collect data ────────────────────────────────────────────────
    latency_data: Dict[str, np.ndarray] = {}
    latency_summaries: Dict[str, Dict] = {}
    validation_summaries: Dict[str, Dict] = {}

    for variant in MODEL_VARIANTS:
        latency_data[variant] = _load_latency_per_image(data_dir, variant)
        latency_summaries[variant] = _load_latency_summary(data_dir, variant)
        validation_summaries[variant] = _load_validation_summary(data_dir, variant)

    # Abort if critical data is missing
    if not all(len(v) > 0 for v in latency_data.values()):
        logger.error("Some latency data is missing; cannot generate Figure 3.")
        return []
    if not all(validation_summaries.values()):
        logger.error("Some validation summaries are missing; cannot generate Figure 3.")
        return []

    with apply_thesis_style():
        # Publication-ready sans-serif overrides (match Figures 1 & 2)
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
            1, 3,
            figsize=(15.0, 5.0),
            gridspec_kw={"width_ratios": [1.0, 1.4, 1.2]},
        )

        # ── Cohesive colour palette (matching Figures 1 & 2) ────────
        colors = ['#2f4f4f', '#4682b4', '#2e8b57', '#b22222']
        display_names = [MODEL_DISPLAY_NAMES[v] for v in MODEL_VARIANTS]
        short_names = ['Baseline', 'FP32', 'FP16', 'INT8']

        # ════════════════════════════════════════════════════════════
        # Panel A: Model Size (Horizontal Bars)
        # ════════════════════════════════════════════════════════════
        ax_a = axes[0]
        sizes_mb = [MODEL_SIZES_BYTES[v] / (1024 ** 2) for v in MODEL_VARIANTS]
        baseline_mb = sizes_mb[0]

        y_pos = np.arange(len(MODEL_VARIANTS))
        bars = ax_a.barh(
            y_pos, sizes_mb, color=colors,
            edgecolor="black", linewidth=0.8,
            height=0.55, zorder=3,
        )

        # Subtle gradient shadow effect behind bars
        for i, (bar, size) in enumerate(zip(bars, sizes_mb)):
            ax_a.barh(
                y_pos[i], size, color=colors[i],
                edgecolor="none", linewidth=0,
                height=0.62, alpha=0.12, zorder=2,
            )

        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(display_names, fontsize=10)
        ax_a.set_xlabel("Model Size (MB)", fontsize=10)
        ax_a.set_title("(a) Storage Footprint", fontsize=11, fontweight="bold", loc="left")
        ax_a.invert_yaxis()
        ax_a.set_xlim(0, max(sizes_mb) * 1.40)
        ax_a.grid(True, axis='x', linestyle='-', color='gray', alpha=0.15, linewidth=0.5, zorder=0)
        ax_a.spines['top'].set_visible(False)
        ax_a.spines['right'].set_visible(False)

        # Annotate bars with size and compression ratio
        for i, (bar, size) in enumerate(zip(bars, sizes_mb)):
            ratio = baseline_mb / size if size > 0 else 0
            label = f"{size:.2f} MB"
            if i > 0:
                label += f" ({ratio:.1f}×)"
            ax_a.text(
                bar.get_width() + 0.20,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )

        # ════════════════════════════════════════════════════════════
        # Panel B: Latency Distribution (Violin + Strip)
        # ════════════════════════════════════════════════════════════
        ax_b = axes[1]
        latency_list = [latency_data[v] for v in MODEL_VARIANTS]

        # Create violin plot
        parts = ax_b.violinplot(
            latency_list,
            positions=y_pos,
            vert=False,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        # Style violin bodies with gradient alpha
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.5)
            body.set_edgecolor(colors[i])
            body.set_linewidth(1.2)
            body.set_zorder(2)

        # Inner box-whisker overlay for each violin
        for i, variant in enumerate(MODEL_VARIANTS):
            data = latency_data[variant]
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            mean_val = np.mean(data)
            # IQR box
            ax_b.barh(
                i, q3 - q1, left=q1, height=0.12,
                color=colors[i], edgecolor="black",
                linewidth=0.6, alpha=0.9, zorder=4,
            )
            # Median line
            ax_b.plot(
                [median, median], [i - 0.08, i + 0.08],
                color="white", lw=2.0, zorder=5,
            )
            # Mean diamond
            ax_b.scatter(
                [mean_val], [i], marker='D', s=25,
                color="white", edgecolors="black",
                linewidths=0.6, zorder=6,
            )

        # Add jittered strip overlay (subsample for visual clarity)
        rng = np.random.default_rng(42)
        for i, variant in enumerate(MODEL_VARIANTS):
            data = latency_data[variant]
            n_sample = min(300, len(data))
            sample = rng.choice(data, size=n_sample, replace=False)
            jitter = rng.uniform(-0.18, 0.18, size=n_sample)
            ax_b.scatter(
                sample,
                np.full(n_sample, i) + jitter,
                s=4,
                color=colors[i],
                alpha=0.35,
                edgecolors="none",
                zorder=3,
            )

        ax_b.set_yticks(y_pos)
        hardware_labels = {
            "baseline": "Baseline\n(NVIDIA L4 GPU)",
            "fp32": "FP32\n(ARM A76 CPU)",
            "fp16": "FP16\n(ARM A76 CPU)",
            "int8": "INT8\n(ARM A76 CPU)",
        }
        b_labels = [hardware_labels[v] for v in MODEL_VARIANTS]
        ax_b.set_yticklabels(b_labels, fontsize=9)
        ax_b.set_xlabel("Inference Latency (ms)", fontsize=10)
        ax_b.set_title("(b) Latency Distribution", fontsize=11, fontweight="bold", loc="left")
        ax_b.set_ylim(len(MODEL_VARIANTS) - 0.5, -0.8)
        ax_b.grid(True, axis='x', linestyle='-', color='gray', alpha=0.15, linewidth=0.5, zorder=0)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)

        # Annotate mean latency and FPS with callout boxes
        for i, variant in enumerate(MODEL_VARIANTS):
            summary = latency_summaries[variant]
            if summary:
                mean_ms = summary["latency_ms"]["mean"]
                fps = summary["fps"]
                # Position annotation above the violin
                ax_b.annotate(
                    f"μ = {mean_ms:.1f} ms  ({fps:.1f} FPS)",
                    xy=(mean_ms, i),
                    xytext=(mean_ms + 15, i - 0.38),
                    fontsize=8,
                    fontweight="bold",
                    color=colors[i],
                    arrowprops=dict(
                        arrowstyle="->",
                        color=colors[i],
                        lw=0.8,
                        connectionstyle="arc3,rad=-0.15",
                    ),
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        fc="white", ec=colors[i],
                        lw=0.6, alpha=0.9,
                    ),
                    zorder=7,
                )

        # ════════════════════════════════════════════════════════════
        # Panel C: Detection Metrics Grouped Bars
        # ════════════════════════════════════════════════════════════
        ax_c = axes[2]
        metrics_to_plot = ["precision", "recall", "f1"]
        metric_labels = ["Precision", "Recall", "F1-Score"]
        n_metrics = len(metrics_to_plot)
        n_variants = len(MODEL_VARIANTS)
        bar_width = 0.18
        x = np.arange(n_metrics)

        for i, variant in enumerate(MODEL_VARIANTS):
            vals = validation_summaries[variant]
            metric_vals = [vals.get(m, 0.0) for m in metrics_to_plot]
            offset = (i - (n_variants - 1) / 2) * bar_width
            bars_c = ax_c.bar(
                x + offset,
                metric_vals,
                width=bar_width,
                color=colors[i],
                edgecolor="black",
                linewidth=0.8,
                label=short_names[i],
                zorder=3,
            )
            # Subtle shadow behind each bar group
            ax_c.bar(
                x + offset,
                metric_vals,
                width=bar_width + 0.02,
                color=colors[i],
                edgecolor="none",
                alpha=0.08,
                zorder=2,
            )
            # Value labels on top of bars
            for bar_rect, val in zip(bars_c, metric_vals):
                ax_c.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#333333",
                    fontweight="bold",
                    zorder=8,
                )

        ax_c.set_xticks(x)
        ax_c.set_xticklabels(metric_labels, fontsize=10)
        ax_c.set_ylabel("Score", fontsize=10)
        ax_c.set_title("(c) Detection Performance", fontsize=11, fontweight="bold", loc="left")
        ax_c.set_ylim(0, 1.08)
        ax_c.grid(True, axis='y', linestyle='-', color='gray', alpha=0.15, linewidth=0.5, zorder=0)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)

        # Legend inside Panel C
        ax_c.legend(
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        fig.tight_layout(w_pad=2.8)

        # ── Manual gap reduction between Panel B and Panel C ───────
        shift_left = 0.035
        pos_c = axes[2].get_position()
        axes[2].set_position([
            pos_c.x0 - shift_left,
            pos_c.y0,
            pos_c.width,
            pos_c.height,
        ])

        fig_path = output_dir / "figure3_quantization_comparison"
        save_figure(fig, fig_path)
        return [fig_path.with_suffix(".pdf"), fig_path.with_suffix(".png")]


# ── Table 3: Deployment Summary & Per-Class Breakdown ──────────────


def generate_quantization_tables(data_dir: Path, output_dir: Path) -> Dict[str, Path]:
    """Generate Table 3a (overall deployment summary) and Table 3b (per-class F1 breakdown)."""
    results: Dict[str, Path] = {}

    # ── Collect all data ────────────────────────────────────────────
    latency_summaries: Dict[str, Dict] = {}
    validation_summaries: Dict[str, Dict] = {}

    for variant in MODEL_VARIANTS:
        latency_summaries[variant] = _load_latency_summary(data_dir, variant)
        validation_summaries[variant] = _load_validation_summary(data_dir, variant)

    # ═══════════════════════════════════════════════════════════════
    # Table 3a: Overall Deployment Summary
    # ═══════════════════════════════════════════════════════════════

    baseline_bytes = MODEL_SIZES_BYTES["baseline"]

    tex_rows_3a = []
    md_rows_3a = []

    for variant in MODEL_VARIANTS:
        disp = MODEL_DISPLAY_NAMES[variant].replace("\n", " ")
        size_mb = MODEL_SIZES_BYTES[variant] / (1024 ** 2)
        compression = baseline_bytes / MODEL_SIZES_BYTES[variant] if MODEL_SIZES_BYTES[variant] > 0 else 0

        lat = latency_summaries[variant]
        mean_ms = lat["latency_ms"]["mean"] if lat else 0
        fps_val = lat["fps"] if lat else 0
        device = lat.get("device", "—") if lat else "—"

        val = validation_summaries[variant]
        prec = val.get("precision", 0)
        rec = val.get("recall", 0)
        f1 = val.get("f1", 0)

        # LaTeX row
        tex_row = (
            f"    {latex_escape(disp)} & "
            f"{size_mb:.2f} & "
            f"{compression:.2f}$\\times$ & "
            f"{device} & "
            f"{mean_ms:.2f} & "
            f"{fps_val:.1f} & "
            f"{prec:.4f} & "
            f"{rec:.4f} & "
            f"{f1:.4f} \\\\"
        )
        tex_rows_3a.append(tex_row)

        # Markdown row
        md_row = (
            f"| {disp} | {size_mb:.2f} | {compression:.2f}× | "
            f"{device} | {mean_ms:.2f} | {fps_val:.1f} | "
            f"{prec:.4f} | {rec:.4f} | {f1:.4f} |"
        )
        md_rows_3a.append(md_row)

    tex_body_3a = "\n".join(tex_rows_3a)
    tex_str_3a = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Deployment Efficiency Comparison Across Model Variants}\n"
        "  \\label{tab:deployment_efficiency}\n"
        "  \\small\n"
        "  \\begin{tabular}{lcccccccc}\n"
        "    \\toprule\n"
        "    \\textbf{Variant} & \\textbf{Size (MB)} & \\textbf{Compr.} & "
        "\\textbf{Device} & \\textbf{Latency (ms)} & \\textbf{FPS} & "
        "\\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\\n"
        "    \\midrule\n"
        f"{tex_body_3a}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    tex_path_3a = output_dir / "table3a_deployment_summary.tex"
    save_latex_table(tex_str_3a, tex_path_3a, standalone_header=True)
    results["tex_3a"] = tex_path_3a

    md_body_3a = "\n".join(md_rows_3a)
    md_str_3a = (
        "# Table 3a: Deployment Efficiency Comparison\n\n"
        "| Variant | Size (MB) | Compression | Device | Latency (ms) | FPS | Precision | Recall | F1-Score |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        f"{md_body_3a}\n"
    )
    md_path_3a = output_dir / "table3a_deployment_summary.md"
    md_path_3a.parent.mkdir(parents=True, exist_ok=True)
    md_path_3a.write_text(md_str_3a, encoding="utf-8")
    results["md_3a"] = md_path_3a

    # ═══════════════════════════════════════════════════════════════
    # Table 3b: Per-Class F1-Score Breakdown
    # ═══════════════════════════════════════════════════════════════

    # Build per-class F1 lookup: {variant: {class_name: f1}}
    per_class_f1: Dict[str, Dict[str, float]] = {}
    for variant in MODEL_VARIANTS:
        val = validation_summaries[variant]
        if not val:
            continue
        class_f1_map = {}
        for entry in val.get("per_class_metrics", []):
            class_f1_map[entry["class_name"]] = entry["f1"]
        per_class_f1[variant] = class_f1_map

    # LaTeX table
    n_cols = 1 + len(MODEL_VARIANTS)
    col_spec = "l" + "c" * len(MODEL_VARIANTS)
    header_vals = [MODEL_DISPLAY_NAMES[v].replace('\n', ' ') for v in MODEL_VARIANTS]
    header_cells = " & ".join([f"\\textbf{{{v}}}" for v in header_vals])

    tex_rows_3b = []
    for cls in CLASS_NAMES:
        disp_cls = CLASS_DISPLAY_NAMES[cls]
        cells = [latex_escape(disp_cls)]
        f1_vals = []
        for variant in MODEL_VARIANTS:
            f1_val = per_class_f1.get(variant, {}).get(cls, 0.0)
            f1_vals.append(f1_val)

        # Find best F1 for this class to bold it
        max_f1 = max(f1_vals) if f1_vals else 0
        for f1_val in f1_vals:
            formatted = format_value(f1_val, precision=4)
            if abs(f1_val - max_f1) < 1e-6 and max_f1 > 0:
                cells.append(f"\\textbf{{{formatted}}}")
            else:
                cells.append(formatted)

        tex_rows_3b.append("    " + " & ".join(cells) + " \\\\")

    tex_body_3b = "\n".join(tex_rows_3b)
    tex_str_3b = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Per-Class F1-Score Comparison Across Quantized Model Variants}\n"
        "  \\label{tab:per_class_f1_comparison}\n"
        "  \\small\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"    \\textbf{{Class}} & {header_cells} \\\\\n"
        "    \\midrule\n"
        f"{tex_body_3b}\n"
        "    \\bottomrule\n"
        f"  \\end{{tabular}}\n"
        "\\end{table}\n"
    )

    tex_path_3b = output_dir / "table3b_per_class_f1.tex"
    save_latex_table(tex_str_3b, tex_path_3b, standalone_header=True)
    results["tex_3b"] = tex_path_3b

    # Markdown table
    md_header_vals = [MODEL_DISPLAY_NAMES[v].replace('\n', ' ') for v in MODEL_VARIANTS]
    md_header = "| Class | " + " | ".join(md_header_vals) + " |"
    md_sep = "| :--- | " + " | ".join([":---:"] * len(MODEL_VARIANTS)) + " |"

    md_rows_3b = []
    for cls in CLASS_NAMES:
        disp_cls = CLASS_DISPLAY_NAMES[cls]
        cells = [disp_cls]
        f1_vals = []
        for variant in MODEL_VARIANTS:
            f1_val = per_class_f1.get(variant, {}).get(cls, 0.0)
            f1_vals.append(f1_val)

        max_f1 = max(f1_vals) if f1_vals else 0
        for f1_val in f1_vals:
            formatted = f"{f1_val:.4f}"
            if abs(f1_val - max_f1) < 1e-6 and max_f1 > 0:
                cells.append(f"**{formatted}**")
            else:
                cells.append(formatted)

        md_rows_3b.append("| " + " | ".join(cells) + " |")

    md_body_3b = "\n".join(md_rows_3b)
    md_str_3b = (
        "# Table 3b: Per-Class F1-Score Comparison\n\n"
        f"{md_header}\n"
        f"{md_sep}\n"
        f"{md_body_3b}\n"
    )

    md_path_3b = output_dir / "table3b_per_class_f1.md"
    md_path_3b.write_text(md_str_3b, encoding="utf-8")
    results["md_3b"] = md_path_3b

    logger.info("Table 3a and 3b generated successfully.")
    return results
