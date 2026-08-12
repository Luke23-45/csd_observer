"""Command-line entry point (§11): compose -> validate -> run.

``@hydra.main(config_path="configs", config_name="run")`` composes the
run config from the structured groups (dataset, model, training,
evaluation, output), merges CLI overrides, then validates (§10.3) and
runs. The composed ``DictConfig`` is converted to plain dicts before it
reaches the runner so every downstream consumer sees pure Python values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from csd_observer.config.store import register_configs
from csd_observer.config.validate import validate_config
from csd_observer.orchestration.runner import run_benchmark

_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "configs")

# Registered before composition so hydra can resolve every group.
register_configs()


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="run")
def main(cfg: DictConfig) -> None:
    _force_utf8_stdio()
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_config(config)
    overrides = list(hydra.core.hydra_config.HydraConfig.get().overrides.task)
    run_benchmark(config, cli_overrides=overrides)


def _force_utf8_stdio() -> None:
    """Reconfigure ``sys.stdout``/``sys.stderr`` to UTF-8 with replace fallback.

    Hydra's job logging uses ``configs/hydra/job_logging/utf8.yaml``
    (``override hydra/job_logging: utf8`` in ``run.yaml`` defaults) to
    open the FileHandler with ``encoding="utf-8"`` — that is the
    primary defense against Lightning's emoji logs crashing on cp1252
    consoles. This helper is **belt-and-suspenders**: it also
    reconfigures the ``StreamHandler``'s underlying stream objects so
    any direct ``print`` or stdout/stderr write through the hydra
    console handler also survives. ``errors="replace"`` keeps unicode
    errors non-fatal in the worst case.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


if __name__ == "__main__":
    sys.exit(main())
