# Implementation Ledger — Persistence-Aware CSD Evaluation Protocol

> Companion to `Implementation_plan.md`. Every task is atomic and verifiable. Status: `[ ]` pending, `[~]` in progress, `[x]` done, `[!]` blocked. Update status in place; do not rewrite history.

---

## Phase 0 — Scaffolding & Hygiene

- [ ] **L0.1** Add Hydra + OmegaConf + PyTorch-Lightning + requests + pydryad (or `requests`-based Dryad client) to `pyproject.toml`; bump Python pin to >=3.10 (already).
- [ ] **L0.2** Create the empty package skeleton: `datasets/{common,tac,daphnia_ext,synthetic}/`, `models/{common,indicators/<7 sub>/,neural/}`, `training/common/`, `evaluation/{common,persistence}/`, `outputs/`, `orchestration/`, `cli/`. Each with `__init__.py` and a one-line module docstring.
- [ ] **L0.3** Stub the persistence directories: `final_data/{tac,daphnia_ext,synthetic_fold,synthetic_hopf,synthetic_logistic}/{raw,processed}/` with `.gitkeep`. Add `final_data/` to `.gitignore` except manifests.
- [ ] **L0.4** Create `outputs/_ledger/.gitkeep`; add `outputs/` to `.gitignore` except `_ledger/` and `.gitkeep`.
- [ ] **L0.5** Freeze the current `studies/runner/benchmark.py` behavior behind a feature flag so all tests stay green during migration.

---

## Phase 1 — Outputs system (self-contained, no orchestration coupling)

- [ ] **L1.1** `outputs/schema.py`: define the canonical `results.jsonl` row dataclass + per-field validators; raise on missing/typed-wrong.
- [ ] **L1.2** `outputs/writer.py`: port `utils/io.py::OutputWriter`; add the full canonical subtree creation (`resolved_config/ metrics/ results/ artifacts/ logs/ times/`); single-timestamp-per-run with `exist_ok=False` + collision failure.
- [ ] **L1.3** `outputs/writer.py`: atomic JSONL append (write-tmp-then-rename within the run dir) so concurrent method drivers cannot corrupt `results.jsonl`.
- [ ] **L1.4** `outputs/ledger.py`: append-only `outputs/_ledger/runs.jsonl` + `index.json` rebuilder. Row: `run_id, timestamp, name, dataset, methods, config_hash, git_sha, status, path`.
- [ ] **L1.5** `tests/test_outputs_writer.py`: subtree creation, collision failure, atomicity, schema validation, ledger append.
- [ ] **L1.6** Migrate `benchmark/suite.py` and `benchmark/experiments.py` writes onto the new writer; remove `utils/io.py`.

---

## Phase 2 — Evaluation package (the proposed protocol)

- [ ] **L2.1** `evaluation/common/metrics.py`: move `utils/metrics.py` (DT, EW-AUC, FPR) verbatim; add bifurcation-type tagging on every returned dict.
- [ ] **L2.2** `evaluation/common/calibration.py`: move `utils/evaluation.py` (calibrate_threshold, compute_null_metrics, per_traj_dts).
- [ ] **L2.3** `evaluation/persistence/protocol.py`: implement the persistence post-filter — a method-agnostic filter that, given a `(B, T)` score stream + per-trajectory lengths + threshold, returns a `(B, T)` boolean alarm stream where an alarm is `True` only after `k_persist` consecutive scores >= threshold. Add the persistence-aware EW-AUC and persistence-aware DT.
- [ ] **L2.4** `evaluation/persistence/governance.py`: the single governance pipeline every method runs through: `preprocess → score → calibrate(val-null) → persist → metrics → artifacts → row`. Both `evaluate_indicator` and `evaluate_spectral` collapse into this with a method-handle arg.
- [ ] **L2.5** `config` (interim): add an `evaluation/persistenceaware.yaml` (k_persist=5, fpr_target=0.05) and `evaluation/baseline_classic.yaml` (k_persist=1, fpr_target=0.05) for the protocol-vs-classic ablation.
- [ ] **L2.6** `tests/test_persistence_protocol.py`: synthetic score stream, assert k_persist behavior, assert persistence-aware DT > classic DT on noisy spikes, assert EW-AUC degrades gracefully.
- [ ] **L2.7** `tests/test_governance_pipeline.py`: end-to-end on the synthetic fold fixture; assert one schema-valid row in `results.jsonl`.
- [ ] **L2.8** Remove `utils/evaluation.py`, `utils/metrics.py`. Keep `utils/` only for genuine cross-cutting helpers.

---

## Phase 3 — Datasets package (automatic ingestion + persistent processing)

- [ ] **L3.1** `datasets/common/ingest.py`: the state machine (`CHECK_RAW → DOWNLOAD → VERIFY_CHECKSUM → EXTRACT → READY_RAW → PROCESS → SPLIT → READY_PROCESSED`), idempotent skip if `manifest.json` is present and valid. Support `dryad` and `manual` sources; stub `huggingface`/`kaggle` for later.
- [ ] **L3.2** `datasets/common/checksum.py`: sha256 verification with structured error on mismatch.
- [ ] **L3.3** `datasets/common/split.py`: per-replicate / per-trajectory split with deterministic seed; write split indices into the processed manifest.
- [ ] **L3.4** `datasets/common/manifest.py`: read/write `final_data/<name>/processed/manifest.json` (dataset version, processing git-sha, params, split indices, feature schema, content hash).
- [ ] **L3.5** `datasets/synthetic/`: relocate `data/bifurcation.py` here; keep behavior identical; add a `process.py` that calls the existing generator and writes the processed manifest, so synthetic data goes through the same `final_data/synthetic_*/processed/` pipeline as real data.
- [ ] **L3.6** `datasets/tac/config.yaml`: name, source=dryad, doi=10.5061/dryad.4cj4k, licence=CC-BY, raw_format, expected_columns, bifurcation_type=subcritical_hopf, processing params (downsample, detrend, mode-amplitude extraction), split policy.
- [ ] **L3.7** `datasets/tac/ingest.py`: Dryad DOI → download URL resolver; place archive in `final_data/tac/raw/`; verify checksum.
- [ ] **L3.8** `datasets/tac/process.py`: parse rows, detrend, extract dominant acoustic-mode amplitude envelope per experimental run, normalize, replicate-aware split (train/val/test), write processed arrays + manifest.
- [ ] **L3.9** `tests/test_tac_ingest.py`: mock the Dryad download with a tiny fixture; exercise the state machine end-to-end; assert manifest schema and splits.
- [ ] **L3.10** `datasets/daphnia_ext/config.yaml`: name, source=dryad, doi=10.5061/dryad.q3p64, licence (open), raw_format, expected_columns, bifurcation_type=transcritical, processing params, split policy, transition-time annotation.
- [ ] **L3.11** `datasets/daphnia_ext/ingest.py`: same Dryad flow as TAC; checksum-verified.
- [ ] **L3.12** `datasets/daphnia_ext/process.py`: per-replicate Daphnia population trajectories, align on transition time, mark constant-environment controls as null class, replicate-aware split, write processed arrays + manifest.
- [ ] **L3.13** `tests/test_daphnia_ingest.py`: same pattern as L3.9.
- [ ] **L3.14** `datasets/registry.py`: `get_dataset(name, **overrides) -> DatasetBundle` returning a uniform `{"features", "seq_lengths", "bifurcation_times", "is_positive", "split_indices", "meta"}` dict (the current array contract) — so models/evaluation never branch on dataset type.
- [ ] **L3.15** Remove `data/bifurcation.py` (relocated in L3.5); update `benchmark/suite.py` imports.

---

## Phase 4 — Models package (common interface + relocate + neural baselines)

- [ ] **L4.1** `models/common/interface.py`: define `MethodMeta` (name, family, scope_caveat, is_learned) and the method protocol (`fit`, `score`, `meta`).
- [ ] **L4.2** `models/common/registry.py`: `get_method(name)`, `list_methods()`, validate-method-name helper. Migrates `benchmark/methods.py::get_method`.
- [ ] **L4.3** Relocate the seven indicators: `models/var_csd/` → `models/indicators/var_csd/` (and the other six) with the same `indicator.py` + `__init__.py` re-export. No behavior changes.
- [ ] **L4.4** Adapt the proposed `SpectralDriftObserver` to the §4.1 interface (wrap `__call__` → `score`, no-op `fit`, set `scope_caveat`).
- [ ] **L4.5** `tests/test_indicators_relocated.py`: re-run the existing per-indicator unit tests against the new location; behavior parity asserted.
- [ ] **L4.6** Choose 2–3 modern SOTA neural CSD / time-series-early-warning baselines; record the citations in `docs/Implementation_plan/neural_baselines.md` (one short paragraph each: name, paper, why relevant, what makes it a fair baseline against the spectral-drift observer).
- [ ] **L4.7** `models/neural/<baseline_1>/{model.py, lit_module.py, config.yaml, __init__.py}`: implement under the §4.1 interface; trained by `training/`; scored through `evaluation/persistence/governance.py`.
- [ ] **L4.8** `models/neural/<baseline_2>/...`: same as L4.7.
- [ ] **L4.9** (Optional) `models/neural/<baseline_3>/...`: same as L4.7.
- [ ] **L4.10** `tests/test_neural_<b1>.py`, `tests/test_neural_<b2>.py`: smoke-train on a tiny synthetic fixture (1 epoch, n_trajectories=8) + assert scoring output shape and finite scores; no full-training tests here (those are e2e in Phase 6).

---

## Phase 5 — Training package (Lightning, DL baselines only)

- [ ] **L5.1** `training/common/trainer_factory.py`: `build_trainer(config, writer) -> LightningTrainer` with deterministic seeding, early stopping, checkpointing into `writer.path/artifacts/`, log into `writer.path/logs/`, LR scheduler from config.
- [ ] **L5.2** `training/common/callbacks.py`: a `LedgerCallback` that records stage timings to `writer.path/times/timings.json` and updates the run ledger status on epoch end.
- [ ] **L5.3** `training/common/losses.py`: the canonical DL baseline loss (binary alarm CE + persistence-aware regularization, or a contrastive early-vs-late loss — finalized here per the neural baseline choice in L4.6).
- [ ] **L5.4** `tests/test_trainer_factory.py`: build a trainer on a dummy LightningModule, run 1 epoch on a tiny fixture, assert checkpoint + log files exist under a temp `OutputWriter`.

---

## Phase 6 — Configuration (Hydra + OmegaConf)

- [ ] **L6.1** `config/store.py`: register Hydra `ConfigStore` groups: `dataset` (tac, daphnia_ext, synthetic_fold, synthetic_hopf, synthetic_logistic), `model` (spectral_drift, var_csd, ac1_csd, …, neural_<b1>, neural_<b2>), `training` (default, large, none), `evaluation` (persistenceaware, baseline_classic), `run` (composed), `output` (default).
- [ ] **L6.2** Migrate `configs/data/default.yaml` → `configs/dataset/synthetic_*.yaml` (one per system + a `defaults` common file); keep the field names.
- [ ] **L6.3** Migrate `configs/model/default.yaml` into Hydra model-group files (one per method).
- [ ] **L6.4** Migrate `configs/training/default.yaml` into `configs/training/default.yaml` already aligned with Hydra group naming.
- [ ] **L6.5** Migrate `configs/run/*.yaml` runs into Hydra composed runs (`+dataset=… +model=… +evaluation=…`).
- [ ] **L6.6** `config/validate.py`: cross-group invariants (`is_learned=true ⇒ training != none`; `evaluation=k_persist >= 1`; `dataset ∈ registry`); fail fast.
- [ ] **L6.7** Remove the legacy `config/load.py` PyYAML-template loader; delete `configs/run/default.yaml`, `high_noise.yaml`, `low_data.yaml`, `patients_*.yaml`.
- [ ] **L6.8** `tests/test_config_compose.py`: compose every documented run; assert resolved config has all required keys; assert the catalog cross-check still passes.

---

## Phase 7 — Orchestration + CLI

- [ ] **L7.1** `cli/main.py`: Hydra entry point `@hydra.main(config_path="../configs", config_name="run")`. Resolves config, constructs `OutputWriter`, writes `resolved_config/resolved.yaml`, dispatches to `orchestration/runner.py`.
- [ ] **L7.2** `orchestration/runner.py`: the only place that knows the pipeline order (dataset registry → method registry → training-if-learned → governance pipeline → writer → ledger). **No** metric, output, or model logic.
- [ ] **L7.3** `studies/runner/benchmark.py`: replace body with `from csd_observer.cli.main import main; main()` shim for backward compat.
- [ ] **L7.4** Remove `benchmark/` package entirely (its responsibilities are now in `evaluation/persistence/governance.py`, `models/common/registry.py`, `orchestration/runner.py`).
- [ ] **L7.5** `tests/test_e2e_persistence_protocol.py`: smoke run `+dataset=synthetic_fold +model=var_csd +evaluation=persistenceaware` with `n_seeds=1`; assert canonical output tree, ledger row, schema-valid `results.jsonl`, manifest present, no real-data download.

---

## Phase 8 — Real-data runs (the paper's experiments)

- [ ] **L8.1** Run the proposed observer on TAC: `+dataset=tac +model=spectral_drift +evaluation=persistenceaware`. Save to `outputs/spectral_drift_tac_<timestamp>/`.
- [ ] **L8.2** Run every statistical indicator + every neural baseline on TAC under the same `evaluation=persistenceaware` config (one run per method, fixed training config for the learned ones).
- [ ] **L8.3** Repeat L8.1–L8.2 for DaphniaExt.
- [ ] **L8.4** Repeat L8.1–L8.2 for synthetic fold / Hopf / logistic (control channel) — the existing `patients_*` sweep, now under the new CLI.
- [ ] **L8.5** Run the protocol-vs-classic ablation: every (dataset, method) pair under `evaluation=persistenceaware` and `evaluation=baseline_classic`. Persist as two separate runs; the paper contrasts them.
- [ ] **L8.6** Verify ledger `index.json` enumerates every run; verify each run has `resolved_config/resolved.yaml`, `metrics/metrics.json`, `results/results.jsonl`, `times/timings.json`.

---

## Phase 9 — Verification, reproducibility, cleanup

- [ ] **L9.1** Single-page `README.md` rewrite: Hydra-CLI usage, real + synthetic dataset commands, link to `docs/Implementation_plan/`.
- [ ] **L9.2** `AGENTS.md`: pin the canonical commands — `pytest`, `ruff check .`, the Hydra smoke run, the real-data ingestion commands — so future sessions run them without rediscovery.
- [ ] **L9.3** Full repo `pytest -v` green; `ruff check .` clean; one Hydra smoke run green; one TAC + one DaphniaExt real-data run green with manifests.
- [ ] **L9.4** Remove `legacy/` and `studies/` (after `studies/runner/benchmark.py` shim is replaced in L7.3); archive a tagged commit reference in the ledger.
- [ ] **L9.5** Pin git tag `v0.2.0-protocol` once Phase 8's runs are reproducible from a clean checkout.

---

## Cross-cutting risks (track explicitly)

- [ ] **R1** Dryad download throttling / rate limits — mitigate with retries + cache-on-success; never re-download a verified raw archive.
- [ ] **R2** Real-data ground truth: TAC and DaphniaExt transition annotations come from the source papers; record the annotation source + page/figure in each dataset's `config.yaml`.
- [ ] **R3** Scope creep: the observer is theoretically fold-only. Every reported table must carry `bifurcation_type` so empirical-regime results are never silently pooled with fold results.
- [ ] **R4** Single-timestamp invariant: any code path that re-stamps the run timestamp mid-run is a bug; add an assertion in `OutputWriter` that the root path matches the constructor's.
