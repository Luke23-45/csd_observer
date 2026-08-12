# CSD Observer

Persistence-aware early-warning evaluation of critical-slowing-down
(CSD) observers. One Hydra-composed pipeline runs ten benchmark methods
(spectral-drift observer, seven published CSD indicators, two neural
baselines) against synthetic bifurcation datasets and real Dryad
datasets (TAC, DaphniaExt) under a fixed-FPR persistence-aware protocol.

## Install

```bash
pip install -e .
```

Requires Python >= 3.10 and PyTorch >= 2.0.

## Usage (Hydra CLI)

The entry point is `@hydra.main(config_path="configs", config_name="run")`:

```bash
# Installed console script
csd-observer

# Or directly (same entry point)
python -m csd_observer

# Small fast runs (tiny synthetic dataset, indicators only)
csd-observer "dataset.n_trajectories=16" "dataset.max_length=128" "training=none"
csd-observer "dataset.n_trajectories=16" "dataset.max_length=128" "models=[VAR-CSD,AC1-CSD,DMD-CSD]" "training=none"

# Spectral-drift observer
csd-observer "dataset.n_trajectories=16" "dataset.max_length=128" "model=spectral_drift" "models=[Kalman-Spectral-Drift]" "training=none"

# Learned neural baseline (needs training)
csd-observer "models=[LSTM-AlarmNet]"
```

### Groups

| Group | Options |
|---|---|
| `dataset` | `synthetic_fold`, `synthetic_hopf`, `synthetic_logistic`, `tac`, `daphnia_ext` |
| `model` | `default` (all blocks), `spectral_drift`, or one of `{var,ac1,skew,sratio,retrate,dfa,dmd}_csd`, `lstm`, `tcn` |
| `training` | `default`, `none` |
| `evaluation` | `persistenceaware`, `baseline_classic` |
| `output` | `default` |

`models` (run-level) selects which methods run — display names from
`csd_observer.models.common.registry` (e.g. `VAR-CSD`,
`Kalman-Spectral-Drift`, `LSTM-AlarmNet`). Dataset generation knobs go
through the whitelist with Hydra `+` syntax:

```bash
csd-observer "+dataset_overrides.n_trajectories=500" "+dataset_overrides.max_length=200"
```

## Output tree

Each run writes `outputs/<run_name>/<timestamp>/`:

```
resolved_config/  resolved.yaml, cli_overrides.yaml
metadata/         environment.json
metrics/          metrics.json, protocol_checks.json
results/          results.jsonl (+ trajectories/, epoch_logs/ when enabled)
artifacts/        checkpoints/, calibration/
logs/             run.log
times/            timings.json
tables/           aggregates.csv, bootstrap_ci.csv, paired_wilcoxon.csv
```

plus a `.completed`/`.failed` lifecycle marker and an append-only ledger
at `outputs/_ledger/runs.jsonl` with an `index.json` mirror.

## Project structure

```
configs/                 # Hydra groups (dataset, model, training, evaluation, output) + run.yaml
src/csd_observer/
  cli/                   # @hydra.main entry point
  config/                # ConfigStore (structured dataclasses) + §10.3 fail-fast validation
  datasets/              # registry, synthetic generators, real-data ingest/manifest pipeline
  evaluation/            # metric primitives, calibration, persistence-aware governance
  models/                # registry, indicators/, spectral_drift/, neural/ (LSTM, TCN)
  orchestration/         # runner (pipeline order only)
  outputs/               # writer, ledger, schema, summarizer, tables
  training/              # Lightning training for neural baselines
docs/Implementation_plan/  # plan + status ledger (authoritative for phases)
tests/                   # pytest suite (config compose, e2e smoke, indicators, spectral)
```

## Evaluation protocol

Persistence-aware governance (plan §8): thresholds are calibrated on the
validation split at fixed null step-FPR (`fpr_target`), alarms must
persist `k_persist` consecutive steps to fire, and per-run protocol
checks are written to `metrics/protocol_checks.json`. Reported metrics
per (method, seed): detection rate, detection time (mean/median/std),
classic EW-AUC, step-FPR, persistent-FPR, and persistence EW-AUC.
