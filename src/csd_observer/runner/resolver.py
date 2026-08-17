"""Resolve run directories and checkpoints for the runner.

Two sources of truth, in order of preference:

1. The runner state registry — the *exact* run directory/checkpoint a stage
   produced (deterministic even after later runs are added).
2. A strict seed+tag filesystem scan of the output tree (covers runs
   launched manually, outside the runner).

A run directory is an eligible artifact only if it carries the
``<timestamp>.completed`` sibling marker that ``OutputWriter.mark_completed``
writes at the very end of a successful run. The scan therefore can never
select a partial run whose early checkpoints survived a crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from csd_observer.runner.protocol import Method
from csd_observer.runner.registry import RunnerState


class CheckpointError(RuntimeError):
    """Raised when a required run directory or checkpoint cannot be resolved."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _is_completed(run_dir: Path) -> bool:
    """A run is an eligible artifact only if the run finished successfully."""
    marker = run_dir.with_name(f"{run_dir.name}.completed")
    return marker.is_file()


def _iter_runs_newest_first(search_dir: Path):
    """Collect candidate run dirs under search_dir, newest-first."""
    if not search_dir.is_dir():
        return iter(())
    runs: list[Path] = [child for child in search_dir.iterdir() if child.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return iter(runs)


def _find_checkpoint_in_run(
    run_dir: Path, method_name: str | None = None, seed: int | None = None
) -> Path | None:
    """Find the best checkpoint file inside a run directory."""
    ckpt_dir = run_dir / "artifacts" / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = list(ckpt_dir.glob("*.ckpt")) + list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        return None
    candidates = ckpts
    if method_name:
        safe_prefix = method_name.lower().replace("-", "_").replace(" ", "_")
        matched = [
            c
            for c in candidates
            if method_name in c.name
            or safe_prefix in c.name.lower().replace("-", "_").replace(" ", "_")
        ]
        if matched:
            candidates = matched
    if seed is not None:
        seed_matched = [c for c in candidates if f"seed{seed}" in c.name or f"s{seed}" in c.name]
        if seed_matched:
            candidates = seed_matched
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_run_dir(
    outputs_base: Path,
    dataset_name: str,
    stage: int = 1,
    *,
    seed: int,
    tag: str | None = None,
) -> Path:
    """Return the newest completed run directory for dataset/seed."""
    target_dirs = [
        outputs_base / dataset_name,
        outputs_base,
    ]
    for parent in target_dirs:
        if not parent.is_dir():
            continue
        for run in _iter_runs_newest_first(parent):
            if not _is_completed(run):
                continue
            # Check results or timings or config for seed match
            timings = _read_json(run / "times" / "timings.json")
            if timings:
                entries = timings.get("entries", {})
                seed_matches = any(f"s{seed}" in k for k in entries.keys())
                if seed_matches:
                    return run
            # Check results.jsonl
            res_file = run / "results" / "results.jsonl"
            if res_file.is_file():
                try:
                    for line in res_file.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            row = json.loads(line)
                            if row.get("seed") == seed:
                                return run
                except (OSError, json.JSONDecodeError):
                    pass
            # Fallback: if single completed run in folder
            return run

    raise CheckpointError(
        f"No completed run directory found under {outputs_base} for dataset {dataset_name} seed {seed}."
    )


def resolve_checkpoint_path(
    outputs_base: Path,
    method: Method,
    stage: int = 1,
    *,
    seed: int,
    state: RunnerState | None = None,
) -> Path:
    """Resolve the checkpoint file for a method/seed."""
    rel = state.get_ckpt(method.phase_key, seed, stage) if state is not None else None
    if rel:
        candidate = (outputs_base / rel).resolve()
        if candidate.is_file():
            return candidate
    run_dir = resolve_run_dir(
        outputs_base, method.data, stage, seed=seed, tag=method.output_tag
    )
    ckpt = _find_checkpoint_in_run(run_dir, method.name, seed=seed)
    if ckpt is None or not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no checkpoint in artifacts/checkpoints/ "
            f"for method {method.name} seed {seed}."
        )
    return ckpt.resolve()


def resolve_stage_ckpt(
    outputs_base: Path,
    dataset_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
    method_name: str | None = None,
) -> Path:
    """Resolve the checkpoint of a prerequisite provider stage."""
    run_dir = resolve_run_dir(outputs_base, dataset_name, stage, seed=seed, tag=tag)
    ckpt = _find_checkpoint_in_run(run_dir, method_name, seed=seed)
    if ckpt is None or not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no checkpoint in artifacts/checkpoints/."
        )
    return ckpt.resolve()


def checkpoint_exists(
    outputs_base: Path,
    dataset_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
    method_name: str | None = None,
) -> bool:
    """Return whether a valid checkpoint exists for dataset/seed."""
    try:
        run_dir = resolve_run_dir(outputs_base, dataset_name, stage, seed=seed, tag=tag)
        ckpt = _find_checkpoint_in_run(run_dir, method_name, seed=seed)
        return ckpt is not None and ckpt.is_file()
    except CheckpointError:
        return False


def stage_checkpoint_relative(
    outputs_base: Path,
    run_dir: Path,
    method_name: str | None = None,
    seed: int | None = None,
) -> str:
    """Record the run's checkpoint as a path relative to the outputs base."""
    ckpt = _find_checkpoint_in_run(run_dir, method_name, seed=seed)
    if ckpt is None or not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no checkpoint in artifacts/checkpoints/."
        )
    return ckpt.relative_to(outputs_base).as_posix()


def resolve_eval_run_dir(
    outputs_base: Path,
    dataset_name: str,
    *,
    seed: int,
    tag: str | None = None,
) -> Path:
    """Return the newest evaluation run directory."""
    return resolve_run_dir(outputs_base, dataset_name, stage=1, seed=seed, tag=tag)


__all__ = [
    "CheckpointError",
    "checkpoint_exists",
    "resolve_checkpoint_path",
    "resolve_eval_run_dir",
    "resolve_run_dir",
    "resolve_stage_ckpt",
    "stage_checkpoint_relative",
]
