"""Subprocess execution for runner steps.

Runs each command with inherited stdio (so progress bars and per-epoch logs
stream to the console) and raises on a non-zero exit. The console entry
points are resolved via ``shutil.which`` or fall back to ``sys.executable -m csd_observer.cli.main``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from csd_observer.runner.commands import eval_command, train_command
from csd_observer.runner.protocol import Step


class CommandError(RuntimeError):
    """Raised when a runner step's command cannot run or fails."""


def _resolve_executable(cmd_name: str) -> list[str]:
    """Resolve console script or module invocation."""
    found = shutil.which(cmd_name)
    if found is not None:
        return [found]
    # Check next to current python executable
    py_dir = Path(sys.executable).parent
    for candidate_name in (cmd_name, f"{cmd_name}.exe", f"{cmd_name}.cmd"):
        cand = py_dir / candidate_name
        if cand.is_file():
            return [str(cand)]
    # Fallback to python module execution
    if cmd_name == "csd-observer":
        return [sys.executable, "-m", "csd_observer.cli.main"]
    return [cmd_name]


def step_command(
    step: Step, *, ckpt_path: Path | None, outputs_base: Path, defaults: tuple[str, ...]
) -> list[str]:
    """Return the argv for a step (train or eval)."""
    if step.kind == "eval":
        return eval_command(
            step, ckpt_path=ckpt_path, outputs_base=outputs_base, defaults=defaults
        )
    return train_command(
        step, outputs_base=outputs_base, defaults=defaults, ckpt_path=ckpt_path
    )


def run_step(
    step: Step,
    *,
    ckpt_path: Path | None,
    outputs_base: Path,
    defaults: tuple[str, ...],
    cwd: Path,
    log_level: str = "WARNING",
) -> None:
    """Execute one step's command, raising :class:`CommandError` on failure."""
    cmd = step_command(step, ckpt_path=ckpt_path, outputs_base=outputs_base, defaults=defaults)
    prefix = _resolve_executable(cmd[0])
    argv = prefix + cmd[1:]

    print(f"\n[runner] $ {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, cwd=str(cwd))
    if result.returncode != 0:
        raise CommandError(
            f"Step {step.label} failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(argv)}"
        )


__all__ = [
    "CommandError",
    "run_step",
    "step_command",
]
