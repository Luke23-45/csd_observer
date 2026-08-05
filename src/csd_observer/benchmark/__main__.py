"""Command-line interface: ``python -m csd_observer.benchmark [run_name ...]``.

Argument syntax is backwards-compatible with the historical
``studies/runner/benchmark.py``:

    python -m csd_observer.benchmark patients_100 generator=bury difficulty=hard n_seeds=1 methods=VAR-CSD,DMD-CSD

* positional: run config names under ``configs/run/``; when omitted,
  the default sweep ``patients_100..patients_500 high_noise`` runs.
* ``n_seeds=``, ``generator=``, ``difficulty=``: data-config overrides.
* ``methods=``: comma-separated subset of the catalog names, or ``all``.
* ``CSD_EXPERIMENT_PREFIX`` env var: output subdirectory under ``outputs/``.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional, Set, Tuple

from csd_observer.benchmark.methods import METHOD_NAMES
from csd_observer.benchmark.suite import run_config

_DEFAULT_RUNS = [
    "patients_100",
    "patients_200",
    "patients_300",
    "patients_400",
    "patients_500",
    "high_noise",
]

__doc__ = __doc__ or ""
_help = (
    __doc__
    + "\nOutput:\n"
    "    outputs/benchmark/<run_name>/<timestamp>/\n"
    "        configs/resolved.yaml\n"
    "        results/results.jsonl\n"
    "        metrics/metrics.json\n"
)


def parse_args(argv: List[str]) -> Tuple[List[str], Optional[int], Optional[str], Optional[str], Optional[Set[str]]]:
    n_seeds_override: Optional[int] = None
    generator_override: Optional[str] = None
    difficulty_override: Optional[str] = None
    methods_override: Optional[Set[str]] = None
    run_names: List[str] = []
    for arg in argv:
        if arg.startswith("n_seeds="):
            n_seeds_override = int(arg.split("=", 1)[1])
        elif arg.startswith("generator="):
            generator_override = arg.split("=", 1)[1]
        elif arg.startswith("difficulty="):
            difficulty_override = arg.split("=", 1)[1]
        elif arg.startswith("methods="):
            raw = arg.split("=", 1)[1]
            if raw.strip().lower() == "all":
                methods_override = None
            else:
                chosen = {m.strip() for m in raw.split(",") if m.strip()}
                invalid = chosen - set(METHOD_NAMES)
                if invalid:
                    raise ValueError(
                        f"Unknown method(s): {sorted(invalid)}. "
                        f"Valid methods: {', '.join(METHOD_NAMES)}"
                    )
                methods_override = chosen
        elif arg in ("-h", "--help"):
            print(_help)
            sys.exit(0)
        else:
            run_names.append(arg)
    if not run_names:
        run_names = list(_DEFAULT_RUNS)
    return run_names, n_seeds_override, generator_override, difficulty_override, methods_override


def main() -> None:
    run_names, n_seeds, generator, difficulty, methods = parse_args(sys.argv[1:])
    total_started = time.time()
    failed_runs: List[str] = []
    for i, run_name in enumerate(run_names, 1):
        tag = f"[{i}/{len(run_names)}] " if len(run_names) > 1 else ""
        print(f"\n{tag}{'='*70}")
        print(f"{tag}RUN: {run_name}")
        print(f"{tag}{'='*70}")
        try:
            run_config(
                run_name,
                n_seeds=n_seeds,
                generator=generator,
                difficulty=difficulty,
                enabled_methods=methods,
            )
        except Exception as exc:  # keep going: one bad config must not kill the sweep
            import traceback

            print(f"\nERROR: {run_name} failed: {exc}")
            traceback.print_exc()
            failed_runs.append(run_name)
            continue
    total_elapsed = time.time() - total_started
    print(f"\n{'='*70}")
    if failed_runs:
        print(f"WARNING: {len(failed_runs)} run(s) FAILED: {', '.join(failed_runs)}")
    print(f"All runs complete. Total time: {total_elapsed:.1f}s")
    if failed_runs:
        sys.exit(1)


if __name__ == "__main__":
    main()
