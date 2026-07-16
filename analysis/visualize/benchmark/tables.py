"""Table generation for the CSD observer benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from analysis.visualize.benchmark.data import BenchmarkStore
from analysis.visualize.common.constants import METHODS, METHOD_DISPLAY_NAMES, PATIENT_COUNTS, SYSTEMS, SYSTEM_DISPLAY_NAMES
from analysis.visualize.common.latex import save_latex_table


METRICS = ("detection_time", "ew_auc", "fpr")


def _fmt(value: float, precision: int = 3) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.{precision}f}"


def _fmt_mean_std(mean: float, std: float, precision: int = 3) -> str:
    if pd.isna(mean):
        return "nan"
    if pd.isna(std):
        return _fmt(mean, precision)
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def _aggregate(records: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = (
        records.groupby(group_cols, observed=True)
        .agg(
            n=("seed", "count"),
            detection_time_mean=("detection_time", "mean"),
            detection_time_std=("detection_time", "std"),
            ew_auc_mean=("ew_auc", "mean"),
            ew_auc_std=("ew_auc", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            n_epochs_mean=("n_epochs_trained", "mean"),
            n_epochs_std=("n_epochs_trained", "std"),
        )
        .reset_index()
    )
    return agg


def _write_table(base: Path, stem: str, caption: str, label: str, body_md: str, body_tex: str) -> Dict[str, Path]:
    md_path = base / f"{stem}.md"
    tex_path = base / f"{stem}.tex"
    md_path.write_text(body_md, encoding="utf-8")
    save_latex_table(body_tex, tex_path, standalone_header=True)
    return {"md": md_path, "tex": tex_path}


def generate_tables(store: BenchmarkStore, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    # Table 1: 500-patient benchmark summary.
    t1 = _aggregate(store.filtered(patient_count=500), ["system", "method"])
    t1["system"] = t1["system"].astype(str)
    t1["method"] = t1["method"].astype(str)
    t1 = t1.sort_values(["system", "method"])

    md_lines = [
        "# Table 1: 500-Patient Benchmark Summary",
        "",
        "| System | Method | n | DT mean +/- std | EW-AUC mean +/- std | FPR mean +/- std | Threshold mean +/- std | Epochs mean +/- std |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    tex_rows = []
    for _, row in t1.iterrows():
        md_lines.append(
            "| {system} | {method} | {n} | {dt} | {auc} | {fpr} | {thr} | {ep} |".format(
                system=SYSTEM_DISPLAY_NAMES.get(row["system"], row["system"]),
                method=METHOD_DISPLAY_NAMES.get(row["method"], row["method"]),
                n=int(row["n"]),
                dt=_fmt_mean_std(row["detection_time_mean"], row["detection_time_std"]),
                auc=_fmt_mean_std(row["ew_auc_mean"], row["ew_auc_std"]),
                fpr=_fmt_mean_std(row["fpr_mean"], row["fpr_std"]),
                thr=_fmt_mean_std(row["threshold_mean"], row["threshold_std"]),
                ep=_fmt_mean_std(row["n_epochs_mean"], row["n_epochs_std"], precision=1),
            )
        )
        tex_rows.append(
            "    {system} & {method} & {n} & {dt} & {auc} & {fpr} & {thr} & {ep} \\\\".format(
                system=SYSTEM_DISPLAY_NAMES.get(row["system"], row["system"]),
                method=METHOD_DISPLAY_NAMES.get(row["method"], row["method"]),
                n=int(row["n"]),
                dt=_fmt_mean_std(row["detection_time_mean"], row["detection_time_std"]),
                auc=_fmt_mean_std(row["ew_auc_mean"], row["ew_auc_std"]),
                fpr=_fmt_mean_std(row["fpr_mean"], row["fpr_std"]),
                thr=_fmt_mean_std(row["threshold_mean"], row["threshold_std"]),
                ep=_fmt_mean_std(row["n_epochs_mean"], row["n_epochs_std"], precision=1),
            )
        )

    tex_body = "\n".join(tex_rows)
    tex = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{500-patient benchmark summary across systems and observer variants.}\n"
        "  \\label{tab:benchmark_500}\n"
        "  \\begin{tabular}{lllrrrrr}\n"
        "    \\toprule\n"
        "    System & Method & n & Detection Time & EW-AUC & FPR & Threshold & Epochs \\\\\n"
        "    \\midrule\n"
        f"{tex_body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    outputs.update(_write_table(output_dir, "table1_main_benchmark", "Table 1", "tab:benchmark_500", "\n".join(md_lines), tex))

    # Table 2: full sweep summary by patient count.
    t2 = _aggregate(store.records, ["patient_count", "system", "method"])
    t2["system"] = t2["system"].astype(str)
    t2["method"] = t2["method"].astype(str)
    t2 = t2.sort_values(["patient_count", "system", "method"])

    md_lines = [
        "# Table 2: Patient-Depth Sweep Summary",
        "",
        "| Patients | System | Method | n | DT mean +/- std | EW-AUC mean +/- std | FPR mean +/- std |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    tex_rows = []
    for _, row in t2.iterrows():
        md_lines.append(
            "| {pc} | {system} | {method} | {n} | {dt} | {auc} | {fpr} |".format(
                pc=int(row["patient_count"]),
                system=SYSTEM_DISPLAY_NAMES.get(row["system"], row["system"]),
                method=METHOD_DISPLAY_NAMES.get(row["method"], row["method"]),
                n=int(row["n"]),
                dt=_fmt_mean_std(row["detection_time_mean"], row["detection_time_std"]),
                auc=_fmt_mean_std(row["ew_auc_mean"], row["ew_auc_std"]),
                fpr=_fmt_mean_std(row["fpr_mean"], row["fpr_std"]),
            )
        )
        tex_rows.append(
            "    {pc} & {system} & {method} & {n} & {dt} & {auc} & {fpr} \\\\".format(
                pc=int(row["patient_count"]),
                system=SYSTEM_DISPLAY_NAMES.get(row["system"], row["system"]),
                method=METHOD_DISPLAY_NAMES.get(row["method"], row["method"]),
                n=int(row["n"]),
                dt=_fmt_mean_std(row["detection_time_mean"], row["detection_time_std"]),
                auc=_fmt_mean_std(row["ew_auc_mean"], row["ew_auc_std"]),
                fpr=_fmt_mean_std(row["fpr_mean"], row["fpr_std"]),
            )
        )

    tex = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Patient-depth sweep summary across all selected complete benchmark batches.}\n"
        "  \\label{tab:benchmark_sweep}\n"
        "  \\begin{tabular}{lllrrrr}\n"
        "    \\toprule\n"
        "    Patients & System & Method & n & Detection Time & EW-AUC & FPR \\\\\n"
        "    \\midrule\n"
        f"{chr(10).join(tex_rows)}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    outputs.update(_write_table(output_dir, "table2_patient_sweep", "Table 2", "tab:benchmark_sweep", "\n".join(md_lines), tex))

    # Table 3: batch inventory and completeness.
    batch_index = store.batch_index.copy()
    batch_index["selected"] = batch_index.apply(
        lambda row: row["batch_id"] == store.selected_batches[int(row["patient_count"])].batch_id,
        axis=1,
    )
    batch_index = batch_index.sort_values(["patient_count", "timestamp"])

    md_lines = [
        "# Table 3: Benchmark Batch Inventory",
        "",
        "| Patients | Batch | Records | Unique triples | Systems | Selected |",
        "| :--- | :--- | ---: | ---: | ---: | :---: |",
    ]
    tex_rows = []
    for _, row in batch_index.iterrows():
        selected = "yes" if bool(row["selected"]) else "no"
        md_lines.append(
            f"| {int(row['patient_count'])} | {row['batch_id']} | {int(row['n_records'])} | {int(row['n_unique_triples'])} | {int(row['n_systems'])} | {selected} |"
        )
        tex_rows.append(
            f"    {int(row['patient_count'])} & {row['batch_id']} & {int(row['n_records'])} & {int(row['n_unique_triples'])} & {int(row['n_systems'])} & {selected} \\\\"
        )

    tex = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Inventory of benchmark batches used for visualization. The selected batch for each patient count is the most complete batch available.}\n"
        "  \\label{tab:benchmark_inventory}\n"
        "  \\begin{tabular}{llllll}\n"
        "    \\toprule\n"
        "    Patients & Batch & Records & Unique triples & Systems & Selected \\\\\n"
        "    \\midrule\n"
        f"{chr(10).join(tex_rows)}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    outputs.update(_write_table(output_dir, "table3_batch_inventory", "Table 3", "tab:benchmark_inventory", "\n".join(md_lines), tex))

    return outputs
