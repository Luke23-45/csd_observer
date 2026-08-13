# Implementation Plan v3 — Robustness Hardening of the Persistence-Aware CSD Benchmark

Status: new working plan for the robustness rewrite.
Base: `docs/Implementation_plan/Implementation_plan.md` (v2) + `docs/Implementation_plan/ledger.md` (current ledger state).
Date: 2026-08-13.
Verification basis: full source review of `src/csd_observer` (every module), `configs/`, `tests/`, plus executed probes (compose + generator shape) and the existing suite (110 tests green).

> Purpose of this document: the v2 plan built the feature set; this plan makes the
> implementation **robust across dataset types (synthetic → real-world)**, makes the
> runner a pure orchestrator, and closes every correctness gap found in the review.
> It is the working task list; the v2 ledger remains the feature ledger. Items below
> are numbered `R0..R7` (robustness) and cross-reference ledger items where one exists.

---

## 1. Review verdict (what was verified, 2026-08-13)

Executed:

- `python -m pytest tests -q` → **110 passed** (config compose, ingest + processors with fixtures, e2e smoke, neural smoke, evaluation benchmark, robust patches).
- Probe (Hydra compose + generator): `dataset.n_trajectories=16 dataset.max_length=128` composes into the `dataset` block but **never reaches the generator** — the generated bundle stays `(500, 200, 1)`. See finding A1.
- Full read of: `orchestration/runner.py`, `cli/main.py`, `config/{store,validate}.py`, `datasets/` (registry, provision, common/{pipeline,ingest,split,states,manifest,errors}, synthetic generators), `models/` (common/{interface,registry,indicators,mode,systems,detrend}, indicators/var_csd, neural base + lstm + tcn + patchtst), `training/common/{trainer_factory,callbacks,losses}`, `evaluation/` (common/{metrics,calibration}, persistence/{protocol,governance}), `outputs/` (writer, ledger, schema, metadata, tables, summarize), all `configs/*.yaml`, and the tests.

Verdict: the architecture skeleton is sound (one-way imports, registry, governance driver, FPR calibration, censoring policy). But there are **silent-correctness gaps**, a **runner that still owns training orchestration**, and **dataset semantics hard-coded in the models package**. All are fixable; none requires re-doing the science.

---

## 2. Findings

Severity: CRITICAL (silent wrong results), HIGH (violates an architectural contract), MED (robustness/config dead knob), LOW (hygiene).

### 2.1 Correctness bugs (silent wrong results)

| ID | Sev | Finding | Location |
|---|---|---|---|
| A1 | CRITICAL | `dataset.n_trajectories`, `dataset.max_length`, `dataset.noise_scale` … set via `dataset.X=…` overrides **never reach the generator**; only `dataset_overrides.*` (with `+`) does. The AGENTS.md CLI smoke commands are therefore misleading: they silently run 500×200 trajectories. Verified by execution. | `orchestration/runner.py:80` (`overrides` read from `dataset_overrides` only), `config/validate.py:144-165` (gates read top-level `n_trajectories`, not `dataset.n_trajectories`), `AGENTS.md` smoke commands |
| A2 | HIGH | `evaluation.early_start_delta` / `early_end_delta` (schema + yaml) are **dead knobs**: `governance.evaluate_method` never passes them into `compute_early_warning_auc` / `compute_persistent_ew_auc`, which always use defaults 50/5. | `configs/evaluation/persistenceaware.yaml:4-5`, `config/store.py:183-184`, `evaluation/persistence/governance.py:118-135` |
| A3 | HIGH | Neural training loss `alarm_bce` uses its **hard-coded `k_persist=5`** smoothing regardless of `evaluation.k_persist`. For `evaluation=baseline_classic` (k=1) the network is still optimized with k=5 smoothing — the ablation is asymmetric (learned methods get a persistence prior in both branches; indicators do not). | `orchestration/runner.py:236` (`lit_cls(model_cfg, training, alarm_bce)` — no k_persist), `training/common/losses.py:15` |
| A4 | MED | `processing.min_length` is ignored by the common gate: `pipeline._run_gates` uses module constant `MIN_LENGTH=100`. The two real processors honor the config knob, the common gate does not — so the knob is half-dead, and `_matches_processing` invalidates the cache on its change while the gate output does not change. | `datasets/common/pipeline.py:36,172-177`, `datasets/tac/process.py:250`, `datasets/daphnia_ext/process.py:137` |
| A5 | LOW | `null_seed` is whitelisted as a dataset override but **silently overwritten** by the per-seed schedule in the seed loop; a user-pinned `null_seed` has no effect. | `config/validate.py:36`, `orchestration/runner.py:130-133` |

### 2.2 Runner monolith (violates §3 / §11 of v2: "runner contains pipeline order only")

| ID | Sev | Finding |
|---|---|---|
| B1 | HIGH | `orchestration/runner.py` owns **training orchestration**: `_train_if_learned` constructs Lightning modules, mutates per-method config (injects `checkpoint`, `in_channels`, `__run_seed__`), manages seeding and checkpoint paths. The hard-coded side-table `_LEARNED_LIT_MODULES` (runner.py:29-33, 271-281) **duplicates the model registry** — adding a fourth neural baseline requires editing the runner. |
| B2 | MED | The runner mixes five phases in one function: provisioning, writer/ledger/env setup, seed×method governance loop, summarization, lifecycle markers. Phase boundaries exist but are not callable units; a phase-specific failure is only distinguishable via the ledger status string. |
| B3 | LOW | `meta` is captured from the pre-loop bundle and never asserted stable across per-seed bundles. It is stable today (bif_type fixed), but nothing enforces it. |

### 2.3 Cross-dataset robustness (models)

| ID | Sev | Finding |
|---|---|---|
| C1 | HIGH | `models/common/mode.py::extract_mode` branches on the **system name** ("subcritical_hopf with 1 channel ⇒ envelope; 2 channels ⇒ radial; otherwise channel 0"). The models package therefore encodes dataset semantics; adding a new real dataset forces an edit in the models package. v2 §5.6 says models branch on `meta` — currently they branch on a string baked into `meta.bif_type`. |
| C2 | MED | No **bundle-contract validation** on the registry path: `get_dataset` (synthetic fast path and `_load_processed`) returns bundles unchecked; `evaluate_method` indexes `arrays_signal["features"]` etc. directly — a malformed bundle (missing key, wrong ndim, dtype) surfaces as a KeyError/IndexError deep in evaluation. |
| C3 | MED | **Score contract not validated**: a method returning `(B,T)` wrong dtype, inf, or un-masked padding is silently tolerated until threshold/percentile math produces NaN metrics. |
| C4 | MED | `PatchTstAlarmNet.forward` raises when `T < patch_len` (models/neural/patchtst/model.py:161-164). The `min_length ≥ 100` gate protects *only if* the gate is honored (see A4) and only if `patch_len ≤ 100` — there is no config-time validation linking dataset min length to model window requirements (DFA 100, PatchTST patch_len). |
| C5 | LOW | Neural scoring (`LstmAlarmMethod.score`, TCN, PatchTST methods) runs a **full-array forward pass** with no batching/chunking. Fine for synthetic 200-step traces; a 288 MB TAC archive with long high-rate traces can exhaust memory. `predict_loader` (models/neural/base/data.py:150) already implements batched scoring but is **dead code** (never called). |
| C6 | LOW | `torch.use_deterministic_algorithms(True, warn_only=True)` (training/common/trainer_factory.py:26) can mask real non-determinism in a run that claims determinism. Decide: hard-fail in CI, document exceptions. |

### 2.4 Config / composition

| ID | Sev | Finding |
|---|---|---|
| D1 | MED | **Hydra deprecation on the current pattern**: every compose run prints `'dataset/synthetic_fold' is validated against ConfigStore schema with the same name. This behavior is deprecated in Hydra 1.1 and will be removed in Hydra 1.2.` The yaml-file + same-name node duality must be migrated before a Hydra upgrade. |
| D2 | MED | Two competing override paths for generator knobs (`dataset.X` vs `+dataset_overrides.X`) with **different behavior** (A1). Single documented path required. |
| D3 | LOW | `run.yaml seed: 42` never participates in the §13 schedule (which uses `seed_offset` only) — a confusing dead knob. Either remove it or define `schedule base = seed_offset + seed` semantics explicitly. |
| D4 | LOW | `registry.list_datasets` accepts any dir under `final_data/` with a `manifest.json` as a valid name, but `provision_dataset` then rejects it ("no processor"). Validation must distinguish *resolvable* (registry + materialized) from *provisionable* (REAL_DATASETS). |

### 2.5 Outputs

| ID | Sev | Finding |
|---|---|---|
| E1 | LOW | `OutputWriter.write_metrics` is never called — `metrics/metrics.json` (promised by §9.1) is never written; only `protocol_checks.json` is. |
| E2 | LOW | `write_timings` **overwrites** per (method, seed): in a multi-method run only the last trained method's fit timings survive. |
| E3 | LOW | `set_run_log` (datasets/common/ingest.py:41) is never wired from the runner — manual-drop instructions never land in `logs/run.log` (stdout only). |

### 2.6 Tests (ledger gaps confirmed)

| ID | Sev | Missing |
|---|---|---|
| F1 | MED | **L0.4** import-walk test (one-way import rule) — not written. |
| F2 | MED | **L1.8** outputs tests (tree, collision, atomicity, schema rejection, ledger rebuild, lifecycle, summarize math vs hand fixture) — not written. |
| F3 | MED | **L2.7** protocol property tests (null-anchor bracket, monotonicity in k_persist, k=1 ≡ classic, censoring, causality) and **L2.8** governance tests — not written. |
| F4 | MED | **L4.7** indicator parity tests + per-neural smoke (only a combined neural smoke exists). |
| F5 | MED | **L5.4** trainer-factory determinism test — not written. |
| F6 | — | No regression tests for A1–A5 (they would have caught the bugs). |

### 2.7 Real-data readiness

| ID | Sev | Status |
|---|---|---|
| G1 | — | **L3.18** sandbox real-data dry-run is blocked on the archives / `DRYAD_API_TOKEN`. Processors exist; TDMS internals (R2) and Daphnia zip layout (R3) remain unverified unknowns. Kept as a gated milestone below. |

---

## 3. Answers to the three design questions (as the hardening target)

### Q1. How are the indicator / neural baselines designed to be robust to any dataset?

Current state (works, with caveats):

- **Indicators**: deterministic, windowed, parameter-free per method; score *scale* is irrelevant because the threshold is calibrated per method on val-null at a fixed FPR (`evaluation/common/calibration.py`). NaN where undefined (causal warm-up), NaN never alarms. This is inherently dataset-agnostic **provided** the input is the (B,T,C) bundle and the scalar mode is correct.
- **Neural baselines**: channel-count auto (`in_channels=None` ⇒ inferred from the bundle), causal by construction (LSTM recurrent state, TCN left-padding, PatchTST causal patch masking), labels derived from `bifurcation_times`+`is_positive` (dataset-driven, not name-driven), same governance path as indicators.

The caveats (must fix): mode extraction is keyed on the system *name* (C1); the bundle contract is not validated (C2); the score contract is not validated (C3); PatchTST has a length floor not linked to the dataset gate (C4); neural scoring is not chunked (C5).

Hardening target: methods consume **only** the validated bundle contract + a declared *feature mapping* (radial / channel-0 / envelope) that the dataset declares, never a dataset name. Every score is validated for shape/dtype/NaN/masking before calibration. Window requirements (DFA 100, patch_len, label_window) are cross-checked against the dataset's effective minimum length at compose time.

### Q2. How are configs designed so dataset switching is robust?

Current state: structured dataclasses per group (Hydra ConfigStore) — switching datasets is `dataset=tac`, every group schema-validated at compose; real data requires `split.replicate_based=true`; `dataset_overrides` whitelisted; methods validated against the registry; learned⇒training invariant; seed schedule int32-checked. This part is largely correct.

Gaps (must fix): the `dataset.X` vs `dataset_overrides.X` split creates a silent no-op (A1); the evaluation-group knobs `early_start_delta/early_end_delta` are dead (A2); `processing.min_length` is half-dead (A4); the Hydra yaml+node duality is on a deprecation path (D1); `seed` is a dead knob (D3); resolvable vs provisionable dataset names are conflated (D4).

Hardening target: **one override path** for generator knobs (the whitelisted `dataset_overrides`), with `validate_config` rejecting any generator knob set on the `dataset` group; every evaluation knob read in one place (governance receives the full evaluation block); `processing.min_length` honored by the common gate; Hydra 1.2-safe registration; `seed` either removed or defined in the schedule.

### Q3. What about the runner?

Current state: a single ~280-line function owning provisioning, output/ledger setup, training orchestration, the seed×method loop, summarization, and lifecycle — with a parallel lit-module registry (B1, B2).

Hardening target (v2 §3/§11 taken literally):

```
cli/main.py                       # hydra entry: compose → validate → run
orchestration/
    runner.py                     # pipeline order ONLY: prepare → execute → finalize
    run_context.py                # RunContext (writer, ledger, config, seed schedule, log)
    phase_prepare.py              # provision datasets, mint writer, resolved config, env, ledger row
    phase_execute.py              # seed × method loop; per-method: train? → evaluate_method
    phase_finalize.py             # summarize, lifecycle markers, ledger status
training/common/train.py          # train_method(method, bundle, ctx, cfg) — moved from runner
models/common/registry.py         # also registers each neural baseline's lit-module class
models/common/runner.py           # run_seed: per-seed seeding + method.fit/score glue (optional)
```

The runner calls units; it never implements them. Training moves to `training/common` (which is allowed to import the registry for lit-module lookup — the import rule `cli → orchestration → {datasets, models, training, evaluation, outputs, config}` permits `training` to see `models`; the reverse (`models`→`training`) stays forbidden; loss stays injected).

---

## 4. Task list

Every task has a Definition of Done (DoD). Tests are first-class tasks, not afterthoughts.

### Phase R0 — Correctness fixes (silent-wrong-result class)

- [x] **R0.1** Single override path for generator knobs.
      `validate_config`: reject `dataset.{n_trajectories,max_length,noise_scale,obs_noise_scale,seed,generator,difficulty}` set to a non-default value (fail-fast, tell the user to use `+dataset_overrides.X`); move the min-dimension gates to read `dataset_overrides` only (they already do — keep). Runner: keep reading `dataset_overrides` only; assert `null_seed` handling (see R0.5).
      DoD: the probe from §1 yields `(16, 128, 1)` when using `+dataset_overrides.…` and a loud `ValueError` when using `dataset.n_trajectories=16`; AGENTS.md smoke commands updated to the canonical form; compose test added.
- [x] **R0.2** Evaluation window knobs wired end-to-end.
      `governance.evaluate_method` reads `early_start_delta`/`early_end_delta` from the evaluation block and passes them to `compute_early_warning_auc` and `compute_persistent_ew_auc`.
      DoD: a config with `evaluation.early_start_delta=200` produces different windows (asserted via unit test on a fixture); no call site uses the defaults silently.
- [x] **R0.3** Loss persistence window tied to the run's k_persist.
      The trainer layer builds the loss as `partial(alarm_bce, k_persist=evaluation.k_persist)` and the runner/training helper passes it in; `baseline_classic` ⇒ k=1 smoothing.
      DoD: determinism test asserts two runs differing only in `evaluation` produce different loss curves for the same seed (k=5 vs k=1); smoke for `evaluation=baseline_classic` with LSTM.
- [x] **R0.4** `processing.min_length` honored by the common gate.
      `_run_gates` takes the configured `min_length` (default 100); manifest `processing.params` already carries it so staleness logic is unchanged.
      DoD: unit test — gate rejects a bundle with a 50-step trajectory when `min_length=200` is configured, accepts when default; TAC/Daphnia processor tests unchanged (they already honor it).
- [x] **R0.5** `null_seed` semantics resolved. Either (a) the whitelist entry is removed and the schedule is the only source (documented), or (b) a user-pinned `null_seed` is honored per run and the schedule only fills it when absent.
      DoD: test asserting the chosen behavior; `validate_config` docstring updated.
- [x] **R0.6** Regression tests for R0.1–R0.5 (F6). DoD: each of the five bugs has a failing-before/passing-after test.

### Phase R1 — Modular runner (v2 §3/§11 enforced)

- [x] **R1.1** `orchestration/run_context.py` — `RunContext` dataclass: writer, ledger, resolved config, cli overrides, seed schedule, run_id, git_sha, config_hash, log emitter (`set_run_log` wiring, E3). Constructed once in `phase_prepare`; passed (not re-derived) everywhere.
- [x] **R1.2** `orchestration/phase_prepare.py` — provisioning (real datasets), writer minting, resolved config / overrides / environment, ledger `pending` row, `set_run_log` wiring.
- [x] **R1.3** `orchestration/phase_execute.py` — seed × method loop only: per seed → per method → `train_method` (if learned) → `evaluate_method` → row. No training internals here.
- [x] **R1.4** `orchestration/phase_finalize.py` — `summarize_run`, `.completed`, ledger status; failure path handled in `runner.run_benchmark` (mark failed + re-raise).
- [x] **R1.5** `orchestration/runner.py` — shrinks to the call-order skeleton (prepare → execute → finalize) plus the failure lifecycle. Removes `_train_if_learned`, `_split_bundle`, `_ensure_learned_modules`, `_LEARNED_LIT_MODULES`.
- [x] **R1.6** `training/common/train.py` — `train_method(method, bundle, cfg, ctx, seed)` moves the training orchestration (lit-module construction from the registry, dataloaders via `training.common.fit_model`, checkpoint path, per-method config block update). Registry provides the lit-module class (R1.7).
- [x] **R1.7** Registry extension — each neural subpackage registers both the method factory and its lit-module class (`register_method(name, factory, family, lit_module=…)`); `MethodMeta` gains `lit_module_path` (or the registry keeps a parallel mapping *inside* `models/common/registry.py`, never in orchestration).
- [x] **R1.8** `__run_seed__` removed: seeding flows through `RunContext.seed_schedule` into `train_method` and `evaluate_method` explicitly (no config mutation). Assert `meta.bif_type` stability across seeds (B3).
- [x] **R1.9** e2e regression: all existing orchestration tests green with zero behavior change in results.jsonl for the same seed (compare row bytes against the pre-refactor output for one run).

### Phase R2 — Cross-dataset contracts (models)

- [x] **R2.1** Bundle validator — `datasets/common/contract.py::validate_bundle(bundle)` called by `get_dataset` on both the synthetic fast path and `_load_processed`; checks: `features (B,T,C)` float32 finite, `seq_lengths (B,)` int64 ∈ [1, T], `bifurcation_times (B,)` finite float, `is_positive (B,)` bool with both classes, `split_indices` train∪val∪test == all rows with no overlap; errors raise `DatasetErrorCode.MANIFEST_CORRUPT`/`SPLIT_IMBALANCE`.
- [x] **R2.2** Feature mapping declared by the dataset, not the models package.
      Dataset config/processor declares `meta.feature_mode` ∈ {`radial`, `channel_0`, `envelope`, `raw`} (+ `mode_channels` when radial). `models/common/mode.py::extract_mode(features, mode)` becomes data-driven; the old system-name branch is kept only as a fallback for the three synthetic datasets (which declare it too, so the fallback is never exercised) and deprecated.
      DoD: TAC (envelope), Daphnia (channel_0), Hopf (radial) all resolve through the declared mode; a test constructs a method against a synthetic bundle whose meta declares `radial`/`channel_0` and asserts correct extraction.
- [x] **R2.3** Score validator — `evaluation/common/score_check.py::validate_scores(scores, seq_lengths, method)` called by governance after every `method.score`: shape `(B,T)` matching, float32/float64, finite-or-NaN (no inf), NaN allowed only inside the warm-up/undefined prefix (tolerated: NaN anywhere — the protocol defines NaN ⇒ no alarm — but logged once per method); raises on structural violations (wrong shape/dtype/inf).
- [x] **R2.4** Window-floor cross-checks at compose time — `validate_config` computes the dataset's effective `min_length` (config `processing.min_length` or synthetic `max_length`) and rejects: `dfa_csd.window_size > min_length`, `patchtst.patch_len > min_length`, `training.label_window > min_length`.
- [x] **R2.5** Chunked neural scoring — the three neural `score` implementations use a shared batched scorer (adopt `predict_loader`, which is currently dead code C5, or move it into `models/neural/base/scoring.py`); max batch steps from config (`model.<key>.score_chunk` default e.g. 4096). DoD: scoring 10 000-step traces does not allocate a full-array batch; byte-identical scores to the current path on synthetic fixtures.
- [x] **R2.6** Determinism policy — `deterministic: true` runs set `torch.use_deterministic_algorithms(True, warn_only=False)` except on platforms where an op is non-deterministic; a `CSD_OBSERVER_ALLOW_NONDETERMINISM=1` env escape hatch for CI. Documented in AGENTS.md.

### Phase R3 — Config single-source-of-truth

- [x] **R3.1** Hydra 1.2-safe registration (D1): keep the structured nodes as the schema and drop the same-name yaml duality — either (a) yaml files removed in favor of nodes (compose overrides still work via CLI), or (b) yaml kept with explicit `# @package _group_` markers and nodes renamed to avoid auto-schema-matching. Decide per group; assert no deprecation warning in a compose probe (test asserts empty stderr). **Final design (post-verification reversal)**: the group values live in `configs/<group>/<name>.yaml` (21 files, incl. the previously node-only `training/none` and `output/default`); `store.py` registers **nothing** into Hydra's ConfigStore — the dataclasses are the in-code schema contract only. The strictness the nodes used to provide at compose time is recovered two ways: `tests/config/test_yaml_alignment.py` pins every file to exactly its node's non-None values, and `validate_config` merges every composed group block into its schema node (unknown keys / wrong types fail at validate time); Hydra's compose itself still rejects unknown top-level keys. Compose probe stays warning-free.
- [x] **R3.2** `seed` semantics (D3): define the schedule as `base = seed_offset + seed` (so `seed` becomes meaningful and AGENTS.md stays truthful) or remove `seed` from `RunConfig`; update §13 docstring and tests.
- [x] **R3.3** Resolvable vs provisionable (D4): `list_datasets` splits into `list_resolvable()` (registry + materialized) and REAL_DATASETS (provisionable); `validate_config` accepts resolvable, provisioning only for REAL_DATASETS; error message tells the user which processors exist.
- [x] **R3.4** Composed-config artifact completeness: `resolved.yaml` must capture the post-R0.1 canonical override path; add a round-trip test (compose → resolved → re-compose → identical hash).

### Phase R4 — Outputs completion

- [x] **R4.1** `metrics.json` written once at finalize (aggregates across methods/seeds — feed from the row set via `outputs.summarize` data path, E1).
- [x] **R4.2** `write_timings` keyed per (method, seed) — either JSONL append or a keyed dict; the `LedgerCallback` and governance both write through it (E2).
- [x] **R4.3** Manual-ingest log wiring via `RunContext` (E3) — `manual_drop_required` events land in `logs/run.log`.
- [x] **R4.4** `tables/` completeness: add `protocol_checks` aggregate table (drift counts per method) so PROTOCOL_DRIFT is visible in the paper tables, not only JSON.

### Phase R5 — Test & CI matrix (closes ledger F1–F5)

- [x] **R5.1** (F1/L0.4) import-walk test: parse all `src/csd_observer/**/*.py`, build the import graph, assert the one-way rule `cli → orchestration → {datasets, models, training, evaluation, outputs, config}` and no sibling imports between objective packages except through `common/`; `evaluation` never imports `models`; `outputs` never imports `models`; `models` never imports `training`.
- [x] **R5.2** (F2/L1.8) outputs tests: tree creation + collision retry, atomic JSONL (tmp+rename), schema rejection (unknown key, wrong type, inf), ledger append/rebuild/status, lifecycle markers, summarize math vs hand-computed fixture.
- [x] **R5.3** (F3/L2.7) protocol property tests: FKG anchor brackets empirical FPR within tolerance on i.i.d. nulls; monotonicity (k↑ ⇒ FPR↓, P-DT↑); `k_persist=1` ≡ classic DT/FPR on identical scores; censoring policy (censored excluded from DT denominator math); causality (score at t uses only s≤t — assert via a step-perturbation test).
- [x] **R5.4** (F3/L2.8) governance tests: synthetic-fold fixture end-to-end → schema-valid rows, achieved step-FPR within 2× target, protocol_checks rows present per (method, seed), `k_persist=1` reproduces classic row values.
- [x] **R5.5** (F4/L4.7) indicator parity tests: for each of the 7 indicators, a hand-computed fixture asserts exact scores (window math, detrend, NaN warm-up); neural smoke per baseline (1-epoch, 8 trajectories, score shapes finite).
- [x] **R5.6** (F5/L5.4) trainer determinism: same seed twice ⇒ identical loss curve + identical checkpoint bytes (num_workers=0); different seed ⇒ different curve.
- [x] **R5.7** Multi-dataset e2e: all 5 datasets × 2 evaluations × representative method set (VAR-CSD + LSTM-AlarmNet) with skip-gates env → asserts composition, generation shapes, governance rows, determinism across two runs with same seeds.
- [x] **R5.8** CI wiring: `ruff check .` + `pytest -v` in CI per AGENTS.md; heavy matrix tagged and excluded.

### Phase R6 — Real-data milestone (gated on data access)

- [ ] **R6.1** (L3.18) TAC sandbox dry-run: download (token or manual drop) → ingest → TDMS inspection (record channels/sample rate/units in config, R2) → envelope → chunking policy decided and recorded → gates → manifest → `dataset=tac +models=VAR-CSD` run completes with `.completed`.
- [ ] **R6.2** (L3.18) DaphniaExt sandbox dry-run: zip layout + README coding recorded (R3), per-replicate labels validated against Nature 467:456, `~110 days` annotation spot-check, manifest, run completes.
- [ ] **R6.3** Full real-data matrix (v2 Phase 8): TAC + Daphnia × 7 indicators × 3 neural × both evaluations, per-run artifact verification (resolved_config, metadata, protocol_checks, tables, ledger).
- [ ] **R6.4** If either dataset fails a gate at this stage, the failure is a finding, not a patch: record in ledger (R2/R3 risks), update processors.

### Phase R7 — Verification & release

- [ ] **R7.1** Clean-checkout reproduction: one synthetic cell + one real cell from `final_data` + `outputs` only (no network).
- [ ] **R7.2** Seed-sweep pilot re-run (v2 §5 → literature_review §5 caveat): n_seeds ≥ 20 with CIs on all quantities.
- [ ] **R7.3** Tag `v0.2.0-protocol` once R7.1 passes (closes L9.5); remove `studies/` (L9.4) if still present.

---

## 5. Ordering rationale

R0 first (silent wrong results; everything downstream builds on correct numbers).
R1 second (modular runner makes every later change testable at a phase boundary).
R2 third (dataset/model contracts are the actual "works with any dataset" guarantee).
R3–R4 (config + outputs) can proceed in parallel with R2.
R5 tests are written *with* each phase (fail-before/pass-after), the matrix tasks only after R0–R4 land.
R6 is gated on data access; R7 is the release gate.

Risk register additions (extends ledger cross-cutting risks):

- **R8** Hydra 1.2 removal of auto-schema-matching breaks composition on upgrade → R3.1.
- **R9** TAC long-trace memory pressure in neural scoring → R2.5.
- **R10** Determinism policy mismatch between CI and paper runs → R2.6.

---

## 6. Non-goals (explicit)

- No change to the scientific protocol (FKG anchor, censoring, P-DT/P-EW-AUC definitions) — v2 §8 stays normative.
- No change to indicator math or the three neural architectures (fairness contract v2 §6.4).
- No new real datasets beyond TAC + DaphniaExt (v2 §4).
- No removal of the `baseline_classic` ablation.
