"""Figure generation entrypoint for the benchmark visualization pipeline."""

from __future__ import annotations

import logging
import sys

from analysis.visualize.benchmark.data import load_benchmark_store
from analysis.visualize.benchmark.figures import plot_patient_sweep, plot_training_curves, plot_trajectory_panels
from analysis.visualize.common.config import get_config, get_data_root, get_output_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_all_figures() -> None:
    cfg = get_config()
    data_root = get_data_root(cfg)
    output_root = get_output_root(cfg)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    store = load_benchmark_store(data_root)
    logger.info("Generating benchmark figures in %s", figures_dir)
    plot_patient_sweep(store, figures_dir)
    plot_trajectory_panels(store, figures_dir)
    plot_training_curves(store, figures_dir)


if __name__ == "__main__":
    generate_all_figures()
