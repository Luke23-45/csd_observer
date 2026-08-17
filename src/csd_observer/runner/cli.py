"""``csd-sweep`` — the experiment runner CLI for CSD Observer.

Runs complete experiments from the frozen protocol manifest: for every
selected method and seed it executes each training stage in order and then
the persistence-aware evaluation of that method's final-stage checkpoint.
A resumable state registry (``<outputs>/_runner/state.json``) records what
completed and the *exact* artifact each stage produced, so evaluation never
targets a stale or seed-mixed checkpoint and an interrupted sweep resumes
in place.

Examples::

    csd-sweep                                           # full matrix x seeds
    csd-sweep --methods 1,5,9                           # by method index
    csd-sweep --methods VAR-CSD,LSTM-AlarmNet           # by method name
    csd-sweep --methods 1 --seeds 0
    csd-sweep --stage 1                                 # train stage 1 only
    csd-sweep --eval-only                               # (re)run evaluations only
    csd-sweep --methods 1 --dry-run                     # preview commands
    csd-sweep --list                                    # show the method matrix

``python -m csd_observer.runner --help`` is equivalent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from csd_observer.runner.executor import (
    CommandError,
    run_step,
    step_command,
)
from csd_observer.runner.protocol import (
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    load_protocol,
)
from csd_observer.runner.registry import RegistryError, RunnerState
from csd_observer.runner.resolver import (
    CheckpointError,
    resolve_checkpoint_path,
    resolve_eval_run_dir,
    resolve_run_dir,
    resolve_stage_ckpt,
    stage_checkpoint_relative,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = "experiments/synthetic_matrix.json"


def _split_list(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="csd-sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Protocol manifest JSON (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[],
        help="Method indices and/or names (e.g. '1,5,9' or 'VAR-CSD,LSTM-AlarmNet'). "
        "Default: all methods.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[],
        help="Protocol seeds to run (e.g. '0,1'). Default: all protocol seeds.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        default=None,
        help="Run only this training stage (no evaluation).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only the evaluation steps (checkpoints must exist for trained models).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run all training stages but skip the evaluation steps.",
    )
    parser.add_argument(
        "--with-dependencies",
        action="store_true",
        help="Auto-run the required Stage 1 pretraining of a shared provider "
        "when a selected method needs it and the provider is not selected.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps that the state registry already marks completed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed step and keep going instead of stopping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exact commands without executing anything.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the method matrix from the manifest and exit.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Output base directory, relative to the project root (default: outputs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output logging.",
    )
    return parser.parse_args(argv)


def _manifest_path(args: argparse.Namespace) -> Path:
    p = Path(args.manifest)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _outputs_base(args: argparse.Namespace) -> Path:
    p = Path(args.outputs)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _build_plan(protocol: Protocol, args: argparse.Namespace) -> list[Step]:
    methods = (
        protocol.select_methods(_split_list(args.methods))
        if args.methods
        else list(protocol.methods)
    )
    seeds: list[int] | None = None
    if args.seeds:
        try:
            seeds = [int(s) for s in _split_list(args.seeds)]
        except ValueError as exc:
            raise ProtocolError(f"--seeds must be integers: {args.seeds}") from exc
    return build_plan(
        protocol,
        methods,
        seeds=seeds,
        stage=args.stage,
        eval_only=args.eval_only,
        skip_eval=args.skip_eval,
        with_dependencies=args.with_dependencies,
    )


def _require_stage2_prereq(step: Step, outputs_base: Path) -> Path | None:
    req = step.required_checkpoint()
    if req is None:
        return None
    model, stage = req
    try:
        source_tag = (
            step.method.output_tag if step.method.stage2_source == "self" else step.method.task
        )
        return resolve_stage_ckpt(
            outputs_base,
            step.method.data,
            stage,
            seed=step.seed,
            tag=source_tag,
            method_name=model,
        )
    except CheckpointError as exc:
        raise CheckpointError(
            f"{step.method.name} stage 2 needs a {model} stage 1 checkpoint for "
            f"seed {step.seed} under {outputs_base}. Run stage 1 first, "
            "or re-run with --with-dependencies."
        ) from exc


def _eval_target(step: Step, outputs_base: Path, state: RunnerState) -> Path | None:
    if not step.method.stages:
        # Indicators require no training checkpoint
        return None
    return resolve_checkpoint_path(
        outputs_base,
        step.method,
        step.method.final_stage,
        seed=step.seed,
        state=state,
    )


def _should_skip_eval(step: Step, outputs_base: Path, state: RunnerState, force: bool) -> bool:
    if force:
        return False
    entry = state.get(step.method.phase_key, step.seed, "eval")
    if entry is None or entry.get("status") != "completed":
        return False
    if not step.method.stages:
        return True
    try:
        target = _eval_target(step, outputs_base, state)
    except CheckpointError:
        return False
    if target is None:
        return True
    return entry.get("ckpt") == target.relative_to(outputs_base).as_posix()


def _step_done(step: Step, outputs_base: Path, state: RunnerState, force: bool) -> bool:
    if force:
        return False
    if step.kind == "train":
        return state.is_complete(step.method.phase_key, step.seed, step.registry_phase)
    return _should_skip_eval(step, outputs_base, state, force=False)


def _print_plan(
    plan: list[Step], outputs_base: Path, state: RunnerState, args: argparse.Namespace
) -> None:
    print(f"\n[runner] plan ({len(plan)} steps, outputs base: {outputs_base})")
    for index, step in enumerate(plan, start=1):
        tag = " [dependency]" if step.dependency else ""
        status = "done" if _step_done(step, outputs_base, state, args.force) else "pending"
        print(f"  {index:>3}. {step.label:<48} {status}{tag}")


def _print_methods(protocol: Protocol) -> None:
    print(f"Protocol: {protocol.name} — {protocol.task} — {protocol.description}")
    for method in protocol.methods:
        stages = ",".join(f"stage{s}" for s in method.stages) if method.stages else "none"
        print(
            f"  {method.index:>2}  {method.name:<24} {method.model:<16} "
            f"data={method.data:<20} stages={stages:<8} "
            f"eval={'yes' if method.evaluate else 'no'}"
        )
    print(f"seeds: {list(protocol.seeds)}")


def _print_summary(protocol: Protocol, state: RunnerState, counts: dict[str, int]) -> None:
    print("\n[runner] summary")
    print(f"  ran={counts['run']} skipped={counts['skip']} failed={counts['failed']}")
    for method in protocol.methods:
        phases = [f"stage{s}" for s in method.stages]
        if method.evaluate:
            phases.append("eval")
        cells: list[str] = []
        for seed in protocol.seeds:
            statuses = []
            for phase in phases:
                entry = state.get(method.phase_key, seed, phase)
                if state.is_complete(method.phase_key, seed, phase):
                    statuses.append("ok")
                elif entry is not None and entry.get("status") == "failed":
                    statuses.append("FAIL")
                else:
                    statuses.append("-")
            cells.append(f"seed{seed}=[{' '.join(statuses)}]")
        label = f"{method.task}/{method.name}" if method.task else method.name
        print(f"  {label:<30} {'  '.join(cells)}")


def _print_dry_run(
    step: Step, outputs_base: Path, state: RunnerState, defaults: tuple[str, ...]
) -> None:
    try:
        ckpt_abs: Path | None = None
        if step.kind == "eval":
            ckpt_abs = _eval_target(step, outputs_base, state)
        else:
            ckpt_abs = _require_stage2_prereq(step, outputs_base)
        cmd = step_command(step, ckpt_path=ckpt_abs, outputs_base=outputs_base, defaults=defaults)
        print(f"  [dry-run] WOULD RUN  {step.label}")
        print(f"    $ {' '.join(cmd)}")
    except CheckpointError as exc:
        print(f"  [dry-run] BLOCKED  {step.label} — prerequisite missing: {exc}")


def run(args: argparse.Namespace) -> int:
    outputs_base = _outputs_base(args)
    try:
        protocol = load_protocol(_manifest_path(args))
    except (ProtocolError, OSError) as exc:
        print(f"[runner] ERROR loading manifest: {exc}", file=sys.stderr)
        return 2

    if args.list:
        _print_methods(protocol)
        return 0

    try:
        state = RunnerState(RunnerState.default_path(outputs_base))
    except RegistryError as exc:
        print(f"[runner] ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        plan = _build_plan(protocol, args)
    except ProtocolError as exc:
        print(f"[runner] ERROR: {exc}", file=sys.stderr)
        return 2

    if not plan:
        print("[runner] No steps match the current selection.", file=sys.stderr)
        return 0

    _print_plan(plan, outputs_base, state, args)

    counts = {"run": 0, "skip": 0, "failed": 0}
    total = len(plan)
    for index, step in enumerate(plan, start=1):
        prefix = f"[{index}/{total}]"
        try:
            if _step_done(step, outputs_base, state, args.force):
                print(f"{prefix} skip {step.label} (already completed)")
                counts["skip"] += 1
                continue

            if args.dry_run:
                _print_dry_run(step, outputs_base, state, protocol.defaults)
                counts["run"] += 1
                continue

            ckpt_abs: Path | None = None
            if step.kind == "eval":
                ckpt_abs = _eval_target(step, outputs_base, state)
            else:
                ckpt_abs = _require_stage2_prereq(step, outputs_base)

            run_step(
                step,
                ckpt_path=ckpt_abs,
                outputs_base=outputs_base,
                defaults=protocol.defaults,
                cwd=PROJECT_ROOT,
                log_level="INFO" if args.verbose else "WARNING",
            )

            if step.kind == "eval":
                run_dir = resolve_eval_run_dir(
                    outputs_base,
                    step.method.data,
                    seed=step.seed,
                    tag=step.method.output_tag,
                )
                state.mark(
                    step.method.phase_key,
                    step.seed,
                    "eval",
                    ckpt=ckpt_abs.relative_to(outputs_base).as_posix() if ckpt_abs else None,
                    run_dir=run_dir.relative_to(outputs_base).as_posix(),
                )
            else:
                if step.stage is None:
                    raise CheckpointError(f"Train step {step.label} has no stage.")
                run_dir = resolve_run_dir(
                    outputs_base,
                    step.method.data,
                    step.stage,
                    seed=step.seed,
                    tag=step.method.output_tag,
                )
                ckpt_rel = stage_checkpoint_relative(
                    outputs_base, run_dir, step.method.name, seed=step.seed
                )
                state.mark(
                    step.method.phase_key,
                    step.seed,
                    step.registry_phase,
                    run_dir=run_dir.relative_to(outputs_base).as_posix(),
                    ckpt=ckpt_rel,
                )
            counts["run"] += 1
            print(f"{prefix} OK {step.label}")
        except (CheckpointError, CommandError) as exc:
            state.mark_failed(step.method.phase_key, step.seed, step.registry_phase, str(exc))
            counts["failed"] += 1
            print(f"{prefix} FAILED {step.label}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                _print_summary(protocol, state, counts)
                return 1

    _print_summary(protocol, state, counts)
    return 0 if counts["failed"] == 0 else 1


def main() -> None:
    args = parse_args()
    try:
        exit_code = run(args)
    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user.", file=sys.stderr)
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
