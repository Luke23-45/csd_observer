# AGENTS.md

Canonical commands for this repository. Run these before/after any change so the suite stays green.

## Tests

```powershell
python -m pytest tests -q          # full suite (indicator + config + e2e + models)
python -m pytest tests\config\test_config_compose.py -q      # composition/invariant tests
python -m pytest tests\orchestration\test_e2e_smoke.py -q    # end-to-end smoke (fast, tmp dirs)
```

`pytest.ini` registers the `heavy` marker (long reproducibility/training
tests). `scripts/ci.ps1` is the CI gate: ruff + the fast lane
(`-m "not heavy"`) + the heavy lane (`-m heavy`).

## Lint

```powershell
ruff check .        # E/F/I/W/B/UP; line-length 100, E501/B008 ignored
```

## CLI smoke (Hydra)

From the repo root, small synthetic runs that touch the full pipeline
(compose → validate → generate → score → output tree → ledger):

```powershell
$env:PYTHONPATH = "src"
python -m csd_observer.cli.main "+dataset_overrides.n_trajectories=16" "+dataset_overrides.max_length=128" "training=none"
python -m csd_observer.cli.main "+dataset_overrides.n_trajectories=16" "+dataset_overrides.max_length=128" "model=lstm" "models=[LSTM-AlarmNet]" "training.epochs=2" "training.patience=2"
```

Generator knobs must use the `+dataset_overrides.*` syntax — the only
channel into the synthetic generator. Setting them on the `dataset`
group (`dataset.n_trajectories=16`) composes but is a silent no-op and
is **rejected at validation** (R0.1).

For ultra-fast smoke runs (skip the `n_trajectories>=3` / `max_length>=100`
gates), set ``$env:CSD_OBSERVER_SKIP_MIN_LENGTH_GATES = "1"`` before
launching. CI / regression tests use this; production runs should leave
the gates enabled.

Installed console script (same entry point):

```powershell
csd-observer "+dataset_overrides.n_trajectories=16" "+dataset_overrides.max_length=128" "models=[DMD-CSD]" "training=none"
```

Hydra job logging is overridden via `configs/hydra/job_logging/utf8.yaml`
(`override hydra/job_logging: utf8` in `run.yaml` defaults): the file
handler must be UTF-8 or Lightning's emoji messages crash the job log on
cp1252 consoles/locales. The app config dir cannot override a
same-named hydra package file (hydra's `pkg://hydra.conf` wins), hence
the distinct name.

## Determinism knobs

- `training.deterministic=true` (default) enforces
  `torch.use_deterministic_algorithms(True, warn_only=False)` (R2.6): a
  non-deterministic kernel is a hard error, not a warning. Machines
  whose kernels lack deterministic implementations can opt out with
  ``$env:CSD_OBSERVER_ALLOW_NONDETERMINISM = "1"`` (documented escape
  hatch; production runs should leave it unset).
- Run seeds come exclusively from the §13 schedule
  (`seed_offset + s*1000 + 101/202`); there is no `seed` run knob
  (R3.2).

## Architecture notes

- `configs/run.yaml` is the primary Hydra config; the group *values*
  live in `configs/<group>/<name>.yaml` (21 files: 5 dataset, 11 model,
  2 training, 2 evaluation, 1 output). `csd_observer.config.store`
  holds the dataclass schema contract and builds the node tables via
  `register_configs`, but registers **nothing** into Hydra's
  ConfigStore — a same-name file+node pair triggers Hydra's deprecated
  "validated against ConfigStore schema" path (R3.1, later reversed to
  restore the yaml files). Alignment is enforced two ways:
  `tests/config/test_yaml_alignment.py` pins every yaml file to exactly
  its node's non-None values, and `validate_config` merges every
  composed group block into its schema node (unknown keys / wrong types
  fail at validate time). Hydra's compose itself rejects unknown
  top-level keys on plain configs.
- Method display names come from `csd_observer.models.common.registry`
  (e.g. `VAR-CSD`, `LSTM-AlarmNet`), not module keys. Learned methods
  register their Lightning module class there too (`get_lit_module`).
- Feature modes are declared by the dataset (`DatasetConfig.feature_mode`:
  `channel_0` / `radial` / `envelope`), read by the indicator adapters
  (R2.2); the models package never infers the mode from the system name.
- Import layering is enforced by `tests/lint/test_import_graph.py`
  (R5.1): `models`/`datasets`/`outputs` import nothing inside the
  project; `orchestration` is the only layer importing `training`.
- Legacy packages (`benchmark/`, `config/load.py`, `data/`, `utils/`,
  legacy run yamls) were removed at L7.4 — do not import them.
- Do not modify `docs/Implementation_plan/ledger.md` statuses without
  actually completing the item (fail-fast rule).
- The runner is a thin prepare → execute → finalize driver
  (`orchestration/runner.py` + `phase_{prepare,execute,finalize}.py`).
  The `.completed` lifecycle marker is written only after
  `summarize_run` (tables + metrics.json present).
