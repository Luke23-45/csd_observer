"""Experiment runner: orchestrate complete, resumable protocol runs.

See :func:`csd_observer.runner.cli.main` for the command line interface
(``csd-sweep`` or ``python -m csd_observer.runner``).
"""

from csd_observer.runner.cli import main

__all__ = ["main"]
