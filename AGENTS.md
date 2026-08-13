# AGENTS.md

Canonical commands for this repository. Run these before/after any change so the suite stays green.

## Tests

```powershell
python -m pytest tests -q          # full suite (indicator + config + e2e + models)
python -m pytest tests\config\test_config_compose.py -q      # composition/invariant tests
python -m pytest tests\orchestration\test_e2e_smoke.py -q    # end-to-end smoke (fast, tmp dirs)
```

## Lint

```powershell
ruff check .        # E/F/I/W/B/UP; line-length 100, E501/B008 ignored
```

## CLI smoke (Hydra)

From the repo root, small synthetic runs that touch the full pipeline
(compose → validate → generate → score → output tree → ledger):

```powershell
$env:PYTHONPATH = "src"
python -m csd_observer.cli.main "dataset.n_trajectories=16" "dataset.max_length=128" "training=none"
python -m csd_observer.cli.main "dataset.n_trajectories=16" "dataset.max_length=128" "model=lstm" "models=[LSTM-AlarmNet]" "training.epochs=2" "training.patience=2"
```

For ultra-fast smoke runs (skip the `n_trajectories>=3` / `max_length>=100`
gates), set ``$env:CSD_OBSERVER_SKIP_MIN_LENGTH_GATES = "1"`` before
launching. CI / regression tests use this; production runs should leave
the gates enabled.

Installed console script (same entry point):

```powershell
csd-observer "dataset.n_trajectories=16" "dataset.max_length=128" "models=[DMD-CSD]" "training=none"
```

Overrides use Hydra syntax: `+dataset_overrides.n_trajectories=24` (the
composed config is struct; `+` adds whitelisted dataset override keys).

Hydra job logging is overridden via `configs/hydra/job_logging/utf8.yaml`
(`override hydra/job_logging: utf8` in `run.yaml` defaults): the file
handler must be UTF-8 or Lightning's emoji messages crash the job log on
cp1252 consoles/locales. The app config dir cannot override a
same-named hydra package file (hydra's `pkg://hydra.conf` wins), hence
the distinct name.

## Architecture notes

- `configs/run.yaml` is the primary Hydra config; groups resolve from
  `configs/{dataset,model,training,evaluation,output}/` with structured
  schema validation via `csd_observer.config.store.register_configs`.
- Method display names come from `csd_observer.models.common.registry`
  (e.g. `VAR-CSD`, `LSTM-AlarmNet`), not module keys.
- Legacy packages (`benchmark/`, `config/load.py`, `data/`, `utils/`,
  legacy run yamls) were removed at L7.4 — do not import them.
- Neural model files (`configs/model/{lstm,tcn}.yaml`) nest params under
  their block key like indicator yamls (`lstm: {...}`); `TrainingConfig`
  schema carries the trainer knobs (`epochs`, `progress_bar`,
  `label_window`, ...) so CLI overrides are validated.
- Do not modify `docs/Implementation_plan/ledger.md` statuses without
  actually completing the item (fail-fast rule).
