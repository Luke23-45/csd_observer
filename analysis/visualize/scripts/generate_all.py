"""Master CLI for the CSD observer benchmark visualization pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from analysis.visualize.benchmark.data import load_benchmark_store
from analysis.visualize.benchmark.figures import plot_patient_sweep, plot_training_curves, plot_trajectory_panels
from analysis.visualize.benchmark.tables import generate_tables
from analysis.visualize.common.config import get_config, get_data_root, get_output_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all benchmark figures and tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--figures-only", action="store_true", help="Generate only figures.")
    parser.add_argument("--tables-only", action="store_true", help="Generate only tables.")
    args = parser.parse_args()

    run_figures = not args.tables_only
    run_tables = not args.figures_only

    cfg = get_config()
    data_root = get_data_root(cfg)
    output_root = get_output_root(cfg)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 78)
    logger.info("CSD OBSERVER BENCHMARK VISUALIZATION")
    logger.info("=" * 78)
    logger.info("Data root   : %s", data_root)
    logger.info("Output root : %s", output_root)
    logger.info("Figures dir : %s", figures_dir)
    logger.info("Tables dir  : %s", tables_dir)

    t_total = time.perf_counter()
    store = load_benchmark_store(data_root)

    outputs = {"figures": [], "tables": []}
    if run_figures:
        logger.info("")
        logger.info("Generating figures...")
        t0 = time.perf_counter()
        outputs["figures"].extend(plot_patient_sweep(store, figures_dir))
        outputs["figures"].extend(plot_trajectory_panels(store, figures_dir))
        outputs["figures"].extend(plot_training_curves(store, figures_dir))
        logger.info("Figures completed in %.2f s", time.perf_counter() - t0)

    if run_tables:
        logger.info("")
        logger.info("Generating tables...")
        t0 = time.perf_counter()
        outputs["tables"].extend(generate_tables(store, tables_dir).values())
        logger.info("Tables completed in %.2f s", time.perf_counter() - t0)

    manifest = {
        "data_root": str(data_root),
        "selected_batches": {
            str(k): {
                "batch_id": v.batch_id,
                "batch_dir": str(v.batch_dir),
                "records": v.n_records,
                "unique_triples": v.n_unique_triples,
                "systems": v.n_systems,
            }
            for k, v in store.selected_batches.items()
        },
        "figures": [str(p) for p in outputs["figures"]],
        "tables": [str(p) for p in outputs["tables"]],
    }
    manifest_path = output_root / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest written: %s", manifest_path)

    elapsed = time.perf_counter() - t_total
    logger.info("")
    logger.info("=" * 78)
    logger.info("DONE in %.2f seconds", elapsed)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
