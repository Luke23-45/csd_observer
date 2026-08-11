# Implementation Ledger v2 — Persistence-Aware CSD Evaluation Protocol

> Companion to `Implementation_plan.md` v2. Status: `[ ]` pending, `[~]` in progress, `[x]` done, `[!]` blocked. Update in place. Phases close only when their **Definition of Done (DoD)** passes.

---

## Phase 0 — Dependencies & Scaffolding

- [x] **L0.1** Add to `pyproject.toml` + install: `hydra-core`, `omegaconf`, `pytorch-lightning`, `torchmetrics`, `requests`, `nptdms`, `filelock`; commit lockfile.
- [x] **L0.2** Package skeleton (all with `__init__.py` + docstring): `datasets/{common,tac,daphnia_ext,synthetic/{common,fold,hopf,logistic}}`, `models/{common,indicators/{var_csd,ac1_csd,skew_csd,sratio_csd,retrate_csd,dfa_csd,dmd_csd},spectral_drift,neural/}`, `training/common`, `evaluation/{common,persistence}`, `outputs/common`, `orchestration`, `cli`, `config`.
- [x] **L0.3** Persistence roots: `final_data/<5 datasets>/{raw,processed}` + `.gitkeep`; `outputs/_ledger/.gitkeep`; `.gitignore` (keep manifests + `.gitkeep`, ignore data/artifacts).
- [~] **L0.4** Enforce the one-way import rule (§3 of plan) via ruff config + import-walk test. *(Import-walk test not yet written.)*

**DoD:** clean lockfile install; `import csd_observer` works; import-rule test green.

---

## Phase 1 — Outputs (self-contained; no orchestration coupling)

- [x] **L1.1** `outputs/schema.py` — row dataclass (§9.3); validator rejects missing/wrong-typed/unknown keys.
- [x] **L1.2** `outputs/writer.py` — canonical tree (§9.1); timestamp minted once; `exist_ok=False`; mid-run re-stamp assertion.
- [x] **L1.3** Lifecycle markers `.pending → .completed/.failed`; partial artifacts preserved on failure.
- [x] **L1.4** Atomic JSONL append (tmp+rename); schema-validated writes only.
- [x] **L1.5** `outputs/ledger.py` — append-only `runs.jsonl` + `index.json` rebuilder; status mirrors markers.
- [x] **L1.6** `outputs/metadata.py` — `environment.json` fingerprint (git, python, torch, hydra, dataset hashes, host).
- [x] **L1.7** `outputs/tables.py` + `summarize.py` — mean±std, bootstrap 95% CIs, paired Wilcoxon on P-DT; paper-ready CSVs → `tables/`.
- [ ] **L1.8** `tests/test_outputs_*.py` — tree, collision, atomicity, schema rejection, ledger rebuild, lifecycle, summarize math vs hand-computed fixture.
- [x] **L1.9** Migrate `benchmark/` writes onto new writer; delete `utils/io.py`. *(`benchmark/` still present; will be deleted in L7.4. New code paths use the new writer only.)*

**DoD:** L1 tests green; double-import smoke → exactly one timestamp dir; ledger shows `.completed`.

---

## Phase 2 — Evaluation: persistence-aware protocol (the contribution)

- [x] **L2.1** `evaluation/common/metrics.py` — migrate `utils/metrics.py` verbatim (DT, EW-AUC, FPR) + `bif_type` propagation; legacy signature kept for parity.
- [x] **L2.2** `evaluation/common/calibration.py` — migrate `utils/evaluation.py` (calibrate_threshold, null metrics, per-traj dts). *(Implemented inside `evaluation/common/metrics.py`.)*
- [x] **L2.3** `evaluation/persistence/protocol.py` — §8.1–8.4: alarm indicator, causal run-length `r_t`, persistent alarm (`r_t ≥ k_persist`), P-DT, P-EW-AUC, i.i.d. null-anchor cross-check.
- [x] **L2.4** Censoring policy §8.2: `detection_rate` separate from `detection_time`; no ∞-averaging.
- [x] **L2.5** Achieved-FPR check §8.5: test-null step-FPR + persistent-FPR reported vs target; `PROTOCOL_DRIFT` warn >2×.
- [x] **L2.6** `evaluation/persistence/governance.py` — single driver §8.7; replaces `evaluate_indicator/evaluate_spectral` call sites.
- [ ] **L2.7** `tests/test_persistence_protocol.py` — null-anchor bracket; monotonicity in `k_persist` (FPR↓, P-DT↑); `k_persist=1 ≡ classic`; censoring; causality (no future leakage).
- [ ] **L2.8** `tests/test_governance.py` — e2e on synthetic fold fixture; schema-valid rows; achieved-FPR within 2× target.
- [x] **L2.9** Delete `utils/evaluation.py`, `utils/metrics.py` after parity. *(Done at L7.4: legacy `utils/` package removed; tests migrated to `evaluation/common/*` and `models/common/detrend`.)*

**DoD:** protocol property tests green; governance emits §9.3 rows with `detection_rate` + `persistent_fpr`; `k_persist=1` reproduces classic.

---

## Phase 3 — Datasets: ingestion + processing

- [x] **L3.1** `datasets/common/errors.py` — error enum §5.2 + CLI exit codes.
- [x] **L3.2** `datasets/common/states.py` — state machine §5.2; idempotent skip on valid manifest; per-dataset `filelock`.
- [x] **L3.3** `datasets/common/dryad.py` — client: URL-encoded DOI → dataset → versions → files (pagination), anonymous metadata, bearer download (`DRYAD_API_TOKEN`), retries×3 exp-backoff, rate-limit aware.
- [x] **L3.4** `datasets/common/checksum.py` — md5 vs API digest or config-pinned MD5; mismatch → `INGEST_CHECKSUM`.
- [x] **L3.5** `datasets/common/manifest.py` — §5.5 manifest read/write/validate.
- [x] **L3.6** `datasets/common/split.py` — replicate-based splits (real data), deterministic, counts logged.
- [x] **L3.7** `datasets/tac/config.yaml` — verified facts (§4): DOI, file+size, MD5 `82cc5298…c34941`, licence CC0-1.0, bif_type subcritical_hopf; `expected_columns` marked to-validate.
- [x] **L3.8** `datasets/tac/ingest.py` — auto (token) + manual-drop paths (§5.3).
- [x] **L3.9** `datasets/tac/process.py` — unzip, TDMS parse (npTDMS), `Stationary*`/`Ramp*` sections, channel inspection, dominant acoustic-mode envelope, chunking policy (recorded), leak-free z-score.
- [x] **L3.10** `datasets/tac/validate.py` — gates §5.4: non-finite, length ≥ window max, annotation spot-check vs Bonciolini 2018, balance log.
- [x] **L3.11** `datasets/daphnia_ext/config.yaml` — verified facts: 2 files (MD5 `923e08e6…2ac7`, `11770ce4…40c5`), licence, bif_type transcritical.
- [x] **L3.12** `datasets/daphnia_ext/ingest.py` — auto + manual modes.
- [x] **L3.13** `datasets/daphnia_ext/process.py` — parse README (treatment coding), per-replicate daily counts, deteriorating=signal / constant=null, align on transition (t=0), replicate-based split.
- [x] **L3.14** `datasets/daphnia_ext/validate.py` — gates; annotation consistent with Nature 467:456 (~110 days pre-extinction).
- [x] **L3.15** `datasets/synthetic/` — `data/bifurcation.py` → `synthetic/common/`; per-system wrappers; same manifest pipeline (provenance=generator/difficulty/params).
- [x] **L3.16** `datasets/registry.py` — uniform bundle §5.6; delete `data/`; patch imports. *(Done at L7.4: `data/` removed; callers import `datasets.synthetic.common.generators`.)*
- [ ] **L3.17** `tests/test_ingest_*.py` — mocked-Dryad fixtures (digests): success, checksum mismatch, auth fail, rate-limit retry, manual timeout, manifest-skip, lock.
- [ ] **L3.18** Sandbox real-data dry-run: TAC archive → READY_PROCESSED + manifest (download once, cached).

**DoD:** TAC + DaphniaExt processed + manifests in sandbox; synthetic identical contract; every gate code tested; no network in CI.

---

## Phase 4 — Models

- [x] **L4.1** `models/common/interface.py` — `MethodMeta`, `MethodInterface` (fit/score/meta) §6.1.
- [x] **L4.2** `models/common/registry.py` — supersedes `benchmark/methods.py`; name validation + catalog cross-check retained.
- [x] **L4.3** Relocate 7 indicators (§6.2); zero behavior change; **parity gate**: existing per-indicator tests pass at new paths.
- [x] **L4.4** Wrap spectral-drift to interface; no-op `fit`; `scope_caveat` set; README (chunking + precision policy, verified in Phase 8).
- [x] **L4.5** `docs/Implementation_plan/neural_baselines.md` — pin 2–3 families (§6.4: recurrent alarm net, TCN, patch-transformer) with fact-checked citations, objectives, capacity budget, fairness rationale. No fabricated numbers.
- [x] **L4.6** Implement `models/neural/<b1|b2|b3>/{model.py, lit_module.py, config.yaml}` under the fairness contract. *(LSTM + TCN implemented; `config.yaml` in `configs/model/{lstm,tcn}.yaml`.)*
- [ ] **L4.7** `tests/test_indicators_parity.py` + `tests/test_neural_*.py` — 1-epoch smoke (8 trajectories), score shapes, finite scores.

**DoD:** parity green; each neural baseline smoke-runs; citations fact-checked.

---

## Phase 5 — Training

- [x] **L5.1** `training/common/trainer_factory.py` — deterministic seeding (torch/numpy/workers), early stopping, `artifacts/checkpoints/`, scheduler, `logs/`.
- [x] **L5.2** `training/common/callbacks.py` — `LedgerCallback` (timings, status), fingerprint at fit start.
- [x] **L5.3** `training/common/losses.py` — alarm BCE + persistence-aware smoothing; finalized with L4.5.
- [ ] **L5.4** `tests/test_trainer_factory.py` — 1-epoch dummy under temp `OutputWriter`; checkpoint + logs + timings; same-seed → identical loss curve.

**DoD:** L5.4 green incl. determinism.

---

## Phase 6 — Configuration (Hydra + OmegaConf)

- [x] **L6.1** `config/store.py` — ConfigStore groups as structured dataclasses (§10.1).
- [x] **L6.2** Migrate `configs/{data,model,training,run}/*` → `configs/{dataset,model,training,evaluation,run}/*`; `tac.yaml`/`daphnia_ext.yaml` carry §4 facts.
- [x] **L6.3** `config/validate.py` — invariants §10.3 fail-fast.
- [x] **L6.4** Dataset-key override whitelist; `--multirun` semantics (per-job timestamp).
- [x] **L6.5** Delete legacy `config/load.py` + old run yamls; patch call sites. *(Done at L7.4: `config/load.py`, `configs/run/`, `configs/data/`, `configs/ablation/` removed; call sites migrated to Hydra composition.)*
- [x] **L6.6** `tests/test_config_compose.py` — every documented run composes; bad key / missing group / unknown method / `is_learned=true,training=none` rejected.

**DoD:** compose smoke green for every run in master matrix; invariant tests green.

---

## Phase 7 — Orchestration, CLI, E2E

- [x] **L7.1** `cli/main.py` — hydra entry point (§11). *(`@hydra.main(config_path="configs", config_name="run")`; primary config at `configs/run.yaml`, groups as sibling dirs — defaults resolve from the config root.)*
- [x] **L7.2** `orchestration/runner.py` — pipeline order only.
- [x] **L7.3** `studies/runner/benchmark.py` → shim to CLI; removed later. *(Shim delegates to `csd_observer.cli.main`; removal at L9.4.)*
- [x] **L7.4** Delete `benchmark/` (methods → `models/common/registry`; evaluation → `evaluation/persistence/governance`; suite → orchestration). *(Also removed `config/load.py`, `data/`, `utils/`, legacy run/data/ablation configs, `csd_indicators` legacy block; tests migrated to registry + `evaluation/common`.)*
- [x] **L7.5** `tests/test_e2e_smoke.py` — `+dataset=synthetic_fold +models=var_csd +evaluation=persistenceaware n_seeds=1`: canonical tree, `.completed`, ledger row, schema-valid rows, no network.

**DoD:** e2e smoke green in CI; `--multirun` over 2 seeds → 2 distinct timestamps, one run name.

---

## Phase 8 — Real-Data Runs (paper experiments)

Master matrix — every cell needs a run dir + ledger rows:

| Runs | Observer | 7 indicators | 2–3 neural | classic vs persistence |
|---|---|---|---|---|
| TAC | ✅ | ✅ | ✅ | ✅ |
| DaphniaExt | ✅ | ✅ | ✅ | ✅ |
| Synthetic fold / hopf / logistic (control) | ✅ | ✅ | ✅ | ✅ |

- [ ] **L8.1** TAC: `+dataset=tac +models=spectral_drift +evaluation=persistenceaware` (+ classic twin at `k_persist=1`).
- [ ] **L8.2** TAC: all indicators + neural baselines under both evaluations.
- [ ] **L8.3** DaphniaExt: same matrix.
- [ ] **L8.4** Synthetic control channel under both evaluations (replaces `patients_*` sweep).
- [ ] **L8.5** Verify per run: `resolved_config/`, `metadata/`, `protocol_checks.json`, `tables/` present; ledger `index.json` enumerates all runs.

**DoD:** every matrix cell has a `.completed` run; C1–C5 evidence (§1 of plan) exists on disk.

---

## Phase 9 — Verification, Reproducibility, Cleanup

- [x] **L9.1** README rewrite: Hydra-CLI usage, real + synthetic commands, link to `docs/Implementation_plan/`.
- [x] **L9.2** `AGENTS.md`: pin canonical commands (`pytest`, `ruff check .`, hydra smoke, ingest dry-run) so future sessions don't rediscover them.
- [~] **L9.3** Full `pytest -v` green; `ruff check .` clean; hydra smoke green; TAC + DaphniaExt sandbox runs green with manifests. *(213 tests + `ruff check .` clean; hydra smoke green for indicator, spectral, and trained LSTM runs; sandbox real-data runs pending data access.)*
- [ ] **L9.4** Remove `legacy/` and `studies/` (after L7.3 shim); ledger references archived commit.
- [ ] **L9.5** Tag `v0.2.0-protocol` once Phase 8 runs are reproducible from a clean checkout.

**DoD:** clean-checkout reproduction of one TAC cell + one DaphniaExt cell + one synthetic cell.

---

## Cross-Cutting Risks (track explicitly)

- [ ] **R1** Dryad download auth: file download needs `DRYAD_API_TOKEN` (verified). Mitigate: manual-drop fallback with config-pinned MD5s; document token setup in README.
- [ ] **R2** TAC TDMS internals (channels, sample rate, units) unknown until inspected. Mitigate: `expected_columns` validated at ingest, never assumed; chunking policy recorded in manifest.
- [ ] **R3** DaphniaExt zip layout / README coding unknown until inspected. Mitigate: parse README first; replicate labels validated against paper description (deteriorating vs constant).
- [ ] **R4** Real-data replicate counts are small → split imbalance risk. Mitigate: replicate-level splits, balance logged, bootstrap CIs over replicates.
- [ ] **R5** Observer scope: theoretically fold-only. Mitigate: `bif_type` in every row; empirical-regime results never pooled with fold.
- [ ] **R6** Single-timestamp invariant. Mitigate: writer assertion (mid-run re-stamp = bug) + `.pending/.completed` markers.
- [ ] **R7** Processed-data staleness. Mitigate: content hash in manifest; pipeline change invalidates cache, never silently reuses stale data.