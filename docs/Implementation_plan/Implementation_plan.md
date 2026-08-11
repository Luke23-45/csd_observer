# Implementation Plan — Persistence-Aware CSD Evaluation Protocol

> Research goal: validate the **proposed persistence-aware evaluation protocol** for critical-slowing-down (CSD) early-warning observers, using the `Kalman-Spectral-Drift` observer as the reference method. This is **not** a new-model paper; the contribution is the **evaluation protocol** applied to **real bifurcation datasets**. Synthetic fold / Hopf / logistic generators stay only as a controlled-control channel.

---

## 1. Guiding Principles (non-negotiable)

1. **Strict modular architecture.** Every top-level objective lives in its own package; every package contains one sub-package per concrete implementation plus a `common/` (or `shared/`) sub-package for code reused across implementations. No cross-imports between sibling implementations — only via `common/`.
2. **Self-containment.** Each package owns its objective end-to-end (data → model → metrics → artifacts). The runner orchestrates; it does **not** implement.
3. **Persistence.** Inputs (`final_data/`) and outputs (`outputs/`) are durable, versioned, reproducible. A single run produces exactly **one** timestamped output directory.
4. **Real data first.** Two real bifurcation datasets are first-class citizens; synthetic data is the control channel.
5. **Reproducible configuration.** Hydra + OmegaConf compose the final config for every run; the resolved config is itself an artifact.
6. **Research scope.** Fold (saddle-node) is the theoretically grounded regime for the spectral-drift observer. Hopf / period-doubling / DaphniaExt (transcritical) / TAC (subcritical Hopf) are **empirical** and must be reported with scope caveats.

---

## 2. Modular Package Layout (high level only — no code)

```
src/csd_observer/
    datasets/                # data ingestion + processing (the `datasets/` package)
        common/              # shared ingestion primitives + state machine + split logic
        tac/                 # thermoacoustic combustor (Bonciolini 2018)
        daphnia_ext/         # Daphnia microcosm extinction (Drake & Griffen 2010)
        synthetic/           # fold / Hopf / logistic generators (control channel)
        registry.py          # dataset-name -> loader entry point
    models/                  # CSD methods
        common/              # shared indicator interface + scoring utils
        spectral_drift/      # PROPOSED observer (reference method, unchanged physics)
        indicators/          # 7 statistical baselines (var, ac1, skew, sratio, retrate, dfa, dmd)
            var_csd/ ac1_csd/ skew_csd/ sratio_csd/ retrate_csd/ dfa_csd/ dmd_csd/
        neural/              # 2–3 modern SOTA deep-learning baselines (PyTorch Lightning)
            (one subpackage each)
    training/                # Lightning trainers + callbacks (DL baselines only)
        common/
        (per-trainer subpackages as needed)
    evaluation/              # PROPOSED persistence-aware evaluation protocol
        common/              # metric primitives (DT, EW-AUC, FPR, calibration)
        persistence/         # the protocol itself: persistence-aware metrics + governance
        metrics.py
    outputs/                 # experiment output system (self-contained writer/ledger)
        common/
        writer.py            # OutputWriter — single source of truth for run artifacts
        ledger.py            # JSONL ledger + run index, append-only, atomic
        schema.py            # canonical row schemas + validation
    config/                  # Hydra + OmegaConf composition (replaces current PyYAML loader)
        (Hydra ConfigStore groups: dataset, model, training, evaluation, run, output)
    orchestration/           # thin runner; composes config, dispatches to packages
    cli/                     # `python -m csd_observer` entry point (Hydra-decorated)
```

Persistence directories (outside the package):

```
final_data/
    <dataset_name>/raw/        # untouched download (auto or manual)
    <dataset_name>/processed/   # final research dataset (train/eval/test splits)
outputs/
    <method_or_run_name>/<timestamp>/
        resolved_config/   metrics/   results/   artifacts/   logs/   times/
```

`legacy/` and `studies/` are kept only as transitional shims; once the Hydra CLI and packages are live, `studies/runner/benchmark.py` becomes a one-line delegate and `legacy/` is removed.

---

## 3. Datasets — Automatic Ingestion + Persistent Processing

### 3.1 Per-dataset config (Hydra group `dataset`)
Each dataset ships its own config declaring: `name`, `source` (`dryad` | `huggingface` | `kaggle` | `manual`), `doi` / `url`, `licence`, `checksum` (sha256), `raw_format`, `processing` (resampling, detrending, normalization per data type), `split` policy, `bifurcation_annotation` (where the transition is, per source paper), `expected_columns`, `temporal_resolution`.

### 3.2 Ingestion state machine (`datasets/common/ingest.py`)
States: `CHECK_RAW → DOWNLOAD → VERIFY_CHECKSUM → EXTRACT → READY_RAW → PROCESS → SPLIT → READY_PROCESSED`.

- Idempotent: if `raw/` exists, checksum matches, and `processed/` is present with a `manifest.json`, ingestion is skipped.
- `manual` source: prints exact instructions and waits for the file to appear in `raw/`, then verifies checksum.
- All failures raise with a structured error code; never silently proceed.

### 3.3 Per-dataset processing (`datasets/<name>/process.py`)
- **TAC** (`10.5061/dryad.4cj4k`, Bonciolini/Ebi/Boujo/Noiray 2018, R. Soc. Open Sci.): thermoacoustic combustor pressure trace; subcritical Hopf bifurcation. Processing: downsample to the protocol's native rate, detrend slow drift, extract the dominant acoustic-mode amplitude envelope. Annotation: the rate-dependent transition delay point marked in the source paper. Bifurcation type = **subcritical Hopf**.
- **DaphniaExt** (`10.5061/dryad.q3p64`, Drake & Griffen 2010, Nature): Daphnia magna replicate populations crossing an experimentally induced transcritical bifurcation; CSD signatures appear ~110 days (~8 generations) before transition. Processing: per-replicate population trajectory, align on transition time, build null/control replicates from constant-environment populations. Bifurcation type = **transcritical**.
- Each processing module writes a `manifest.json` to `final_data/<name>/processed/` recording: dataset version, processing git-sha, params, split indices, feature schema, and a content hash.

### 3.4 Splits
- Real datasets: per-replicate / per-trajectory split (`train`/`val`/`test`) — never split inside one trajectory.
- Synthetic control: keep the existing per-trajectory seed-based split.
- Split indices are part of the **processed manifest** (deterministic, reproducible).

---

## 4. Models — Three Families, One Interface

### 4.1 Common interface (`models/common/`)
Every method exposes: `fit(train_arrays, val_arrays, config) -> None` (no-op for non-learned), `score(arrays, config) -> (B, T) float32` collapse-probability-equivalent scores, `meta -> MethodMeta` (name, family, scope_caveat, is_learned). The current catalogs/specs already mirror this — formalize it.

### 4.2 Statistical indicators (already implemented)
`var`, `ac1`, `skew`, `sratio`, `retrate`, `dfa`, `dmd` — relocate each under `models/indicators/<name>/` keeping `indicator.py` and a `__init__.py` re-export. No behavior changes.

### 4.3 Proposed observer — `models/spectral_drift/`
Unchanged physics (Rao-Blackwellised particle filter over the spectral gap `c_k`, exact conditional Kalman over the mode `u_k`, Shiryaev-style `p_collapse`). Two changes only:
1. Wrap to the §4.1 common interface so it is interchangeable with baselines.
2. Add a `scope_caveat` metadata field: "theoretically grounded for fold; empirical elsewhere".

### 4.4 Deep learning baselines (`models/neural/`, Lightning) — pick 2–3 modern SOTA
Concrete selection (finalized for the ledger): candidates are the modern CSD / time-series-anomaly early-warning families. The ledger pins the exact choices during the neural baseline task; the plan only constrains them to:
- Sequence model with explicit temporal structure (not a frozen decoder).
- Trained on the **train split only**, calibrated on **val**, evaluated on **test** under the persistence-aware protocol.
- Loss must be comparable across learned methods (binary alarm cross-entropy + persistence-aware regularization, or a contrastive early-vs-late contrastive loss — finalized in the neural baseline task).
- Each lives in its own subpackage with `model.py`, `lit_module.py`, `config.yaml` ( Hydra `model` group entry), `__init__.py`. No cross-imports between neural subpackages except via `models/common/`.

---

## 5. Training (`training/`)
- PyTorch Lightning `Trainer` factory with deterministic seeding, early stopping, checkpointing, LR scheduler — all from config.
- Callbacks write to the **run's own** `outputs/<run>/<timestamp>/artifacts/` and `logs/`. The trainer never decides output paths — it receives an `OutputWriter` handle from the orchestrator.
- Used only by §4.4. The proposed observer and statistical indicators bypass training entirely.

---

## 6. Evaluation — The Proposed Persistence-Aware Protocol

### 6.1 What "persistence-aware" means (the contribution)
The protocol refuses single-time-point alarms. A positive detection is counted only if the alarm score **persists above threshold for `k_persist` consecutive steps** (config; default `k_persist = 5`). This eliminates transient spikes that classical CSD indicator literature reports as "detections" but which are noise-driven. The protocol also mandates:
1. **Fixed-FPR calibration on val-null** (already implemented) — kept.
2. **Persistence post-filter** on every method's score stream — new.
3. **Persistence-aware EW-AUC**: the AUC is computed on the persistently-alarmed windows, not raw scores.
4. **Scope-tagged reporting**: every metric row carries `bifurcation_type` so cross-type aggregation in the paper is explicit.

### 6.2 Package contents
`evaluation/common/metrics.py` (DT, EW-AUC, FPR — migrated from `utils/metrics.py`), `evaluation/common/calibration.py` (threshold calibration — migrated from `utils/evaluation.py`), `evaluation/persistence/protocol.py` (the new persistence filter + persistence-aware metrics), `evaluation/persistence/governance.py` (the single governance pipeline every method runs through: `score → calibrate → persist → metrics → artifacts`).

### 6.3 Migration
`utils/evaluation.py` and `utils/metrics.py` are absorbed into `evaluation/common/`. `utils/` keeps only genuinely cross-cutting helpers (io is replaced by §7; losses moved to `training/common/`).

---

## 7. Outputs — Self-Contained Output System (`outputs/`)

### 7.1 Run directory (canonical, non-negotiable)
```
outputs/<method_or_run_name>/<timestamp>/
    resolved_config/resolved.yaml      # Hydra-resolved, frozen at run start
    metrics/metrics.json               # aggregated per-system, per-method
    results/results.jsonl              # one row per (system, method, seed/replicate)
    results/epoch_logs/               # per-method training logs (DL only)
    results/trajectories/              # per-run score + artifact .npz
    artifacts/                         # model checkpoints, scalar buffers, plots
    logs/run.log                       # structured run log
    times/timings.json                 # per-stage wall-clock
```
Timestamp = run start, `%Y-%m-%d_%H-%M-%S`. **One run = one timestamp.** The writer mints the timestamp once at construction and reuses it for the entire run (no per-stage re-stamping). Collision = fail loudly, not silently merge.

### 7.2 Writer (`outputs/writer.py`)
Single `OutputWriter` class — migrated and hardened from `utils/io.py`:
- Creates the full subdirectory tree at construction (atomic `mkdir(exist_ok=False)`).
- Atomic row writes (write-tmp-then-rename) for `results.jsonl` so concurrent method drivers cannot corrupt the ledger.
- Schema-validated rows (§7.4) before write; invalid rows raise.
- Methods receive the writer handle; they never construct paths.

### 7.3 Run ledger (`outputs/ledger.py`)
Append-only JSONL `outputs/_ledger/runs.jsonl` (one row per run) plus `outputs/_ledger/index.json` rebuilt from the JSONL. Each row: `run_id`, `timestamp`, `name`, `dataset`, `methods`, `config_hash`, `git_sha`, `status`, `path`. This is the single queryable index for analysis; the per-run `results.jsonl` is the per-row detail.

### 7.4 Schema (`outputs/schema.py`)
Canonical row schema for `results.jsonl`: `run_id, timestamp, dataset, bifurcation_type, system, replicate, method, family, is_learned, detection_time, ew_auc, fpr, persistence_k, threshold, n_epochs_trained, params, git_sha, config_hash`. Validation raises on missing/typed-wrong fields — no silent defaults.

---

## 8. Configuration — Hydra + OmegaConf

### 8.1 Groups (Hydra `ConfigStore`)
`dataset` (tac, daphnia_ext, synthetic_fold, synthetic_hopf, synthetic_logistic), `model` (spectral_drift, var_csd, …, neural_<x>), `training` (default, large, none), `evaluation` (persistenceaware, baseline_classic), `run` (composed runs like `patients_100` → become `real_tac`, `real_daphnia`, `synth_fold_control`, …), `output` (default).

### 8.2 Composition
A run config is `+dataset=<x> +model=<y[,z]> +evaluation=<p> +training=<t>`. Hydra composes the full tree, `csd_observer.config.validate` enforces required keys + cross-group invariants (e.g. `is_learned=true ⇒ training != none`), and the resolved config is written to `resolved_config/resolved.yaml` by the writer at run start.

### 8.3 Migration
The existing PyYAML-template loader (`config/load.py`) is *replaced* by Hydra. The current `configs/{data,model,training,run}/*.yaml` files are migrated group-by-group; the run-name positional CLI (`python -m csd_observer.benchmark patients_100 …`) becomes a Hydra `<run>` override.

---

## 9. Orchestration (`orchestration/`) and CLI (`cli/`)
- `cli/main.py` is a Hydra-decorated entry point: resolves config, constructs `OutputWriter`, dispatches to the dataset registry → model registry → evaluation governance → writer.
- `orchestration/runner.py` is the only place that knows the full pipeline order; it contains **no** metric, output, or model logic. The current `benchmark/suite.py` shrinks to this orchestrator; `benchmark/experiments.py` is dissolved into `evaluation/persistence/governance.py`.
- `studies/runner/benchmark.py` stays as a one-line shim for backward compatibility (delegate to `cli/main.py`), then is removed in a later cleanup task.

---

## 10. Tests
Mirrors the package layout under `tests/` — one test module per implementation subpackage, plus a `test_e2e_persistence_protocol.py` smoke test that runs `+dataset=synthetic_fold +model=var_csd +evaluation=persistenceaware` with `n_seeds=1` and asserts the canonical output tree, ledger row, and schema-valid `results.jsonl`. No real-data downloads in tests; synthetic-only fixtures.

---

## 11. Data Accessibility Summary (pinned)

- **TAC** — Dryad, DOI `10.5061/dryad.4cj4k`, CC-BY, immediate download. Source: Bonciolini, Ebi, Boujo & Noiray, "Experiments and modelling of rate-dependent transition delay in a stochastic subcritical bifurcation," *R. Soc. Open Sci.* 5(3), 172078 (2018). Bifurcation type: **subcritical Hopf**.
- **DaphniaExt** — Dryad, DOI `10.5061/dryad.q3p64`, published Jan 2015, open. Source: Drake & Griffen, "Early warning signals of extinction in deteriorating environments," *Nature* 467, 456–459 (2010). Replicate *Daphnia magna* populations crossing an experimentally induced transcritical bifurcation; CSD signatures appear ~110 days (~8 generations) before transition; constant-environment controls show no such pattern. Bifurcation type: **transcritical**.

---

## 12. What Does NOT Change
- The spectral-drift observer's math (`models/spectral_drift/observer.py`, `grid_search.py`, `preprocess.py`).
- The seven statistical indicator implementations (only their file locations).
- The fixed-FPR calibration principle.
- Determinism: per-seed schedule `seed_offset + s * 1000 + 101/202`.
