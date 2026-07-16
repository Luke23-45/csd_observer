"""Table generation entrypoint for the benchmark visualization pipeline."""

from __future__ import annotations

import logging

from analysis.visualize.benchmark.data import load_benchmark_store
from analysis.visualize.benchmark.tables import generate_tables
from analysis.visualize.common.config import get_config, get_data_root, get_output_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_all_tables() -> None:
    cfg = get_config()
    data_root = get_data_root(cfg)
    output_root = get_output_root(cfg)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    store = load_benchmark_store(data_root)
    logger.info("Generating benchmark tables in %s", tables_dir)
    generate_tables(store, tables_dir)


if __name__ == "__main__":
    generate_all_tables()
