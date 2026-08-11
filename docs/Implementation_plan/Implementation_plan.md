# Implementation Plan v2 — Persistence-Aware CSD Evaluation Protocol

> **v2 corrections (all verified, none assumed):**
> 1. **Licence is CC0-1.0 for both datasets — not CC-BY** (verified from Dryad API `license: https://spdx.org/licenses/CC0-1.0.html`).
> 2. **TAC is one 288 MB TDMS binary archive** (not CSV); needs npTDMS; content = stationary traces at 7 operating points + ramp sections.
> 3. **DaphniaExt is `data-and-code.zip` + a README.txt**; server-side MD5 known for every file.
> 4. **Dryad download endpoints require an authenticated API token** — anonymous users may read metadata (30 req/min) but not download files. Ingestion design must therefore support token-auth download *and* a manual-drop fallback, with checksum verification in both paths.
> 5. Protocol definitions (persistence statistic, P-DT, P-EW-AUC), censoring policy, achieved-FPR check, and significance testing are now **formally specified** (§8) — the research contribution is falsifiable.
> 6. Outputs tree finalized with lifecycle markers, paper-ready `tables/`, environment fingerprint, and artifact-richness policy (§9).

---

## 1. Research Objectives — What the Paper Must Support

The paper's contribution is a **persistence-aware evaluation protocol** for CSD early-warning observers, validated on **real bifurcation datasets**, with the `Kalman-Spectral-Drift` observer as the reference method. Every claim below has a mandatory supporting artifact:

| Claim (paper) | Mandatory evidence |
|---|---|
| C1. Persistence-aware alarms remove transient/spurious detections vs classic scoring | Ablation `evaluation=persistenceaware` vs `evaluation=baseline_classic` on every (dataset, method) pair; per-run dirs + ledger rows |
| C2. Detections degrade gracefully on real data (absence of a known onset or censored detection must be explicit, never averaged away) | Detection-rate reported *separately* from detection-time; censoring policy §8.2 |
| C3. The observer and all baselines are compared under identical governance | One governance driver for every method (§8.7); same splits, same calibration rule, same seeds/replicates |
| C4. Real-data scope is stated per bifurcation type | Every metric row carries `bifurcation_type`; the observer's fold-only theoretical grounding is a scope tag, not a hidden assumption |
| C5. Reproducible from a clean checkout | `final_data/` manifests, git+env fingerprint in `metadata/`, pinned dependencies, seed schedule |

---

## 2. Guiding Principles (unchanged from v1)

1. Strict modular architecture; every top-level objective is a package; every package has per-implementation subpackages + `common/`; no sibling cross-imports.
2. Self-containment: each package owns its objective end-to-end; the runner orchestrates only.
3. Persistence: `final_data/` and `outputs/` durable; one timestamp, one run directory.
4. Real data first; synthetic is the control channel.
5. Hydra + OmegaConf compose a fully validated config per run; resolved config is an artifact.
6. Honest scope tagging (fold = theoretical; everything else = empirical).

---

## 3. Canonical Package Layout (resolved)

Import rule (one-way, enforced by ruff/lint and tests):
`cli → orchestration → {datasets, models, training, evaluation, outputs, config}`.
No package below `orchestration` may import another objective package except through its own `common/`; `evaluation` never imports `models`; `outputs` never imports `models`.

```
src/csd_observer/
    datasets/
        common/            # states, error codes, dryad client, checksum, manifest, split, lock
        tac/               # config.yaml, ingest.py, process.py, validate.py
        daphnia_ext/       # config.yaml, ingest.py, process.py, validate.py
        synthetic/
            common/        # generators moved from data/bifurcation.py
            fold/ hopf/ logistic/
        registry.py
    models/
        common/            # MethodMeta, MethodInterface, registry
        indicators/
            var_csd/ ac1_csd/ skew_csd/ sratio_csd/ retrate_csd/ dfa_csd/ dmd_csd/
        spectral_drift/    # untouched physics; wrapped to MethodInterface
        neural/
            (2-3 subpackages, one per baseline)
    training/
        common/            # trainer factory, callbacks, losses
        (one subpackage per neural baseline)
    evaluation/
        common/            # metrics.py, calibration.py
        persistence/       # protocol.py, governance.py
    outputs/
        common/
        writer.py ledger.py schema.py tables.py metadata.py summarize.py
    config/                # hydra store, structured configs, validate.py
    orchestration/runner.py
    cli/main.py
```

`final_data/<dataset>/raw|processed/` and `outputs/<run>/<timestamp>/` live outside the package.

---

## 4. Verified Dataset Facts (Dryad API, retrieved 2026-08-11)

**TAC — thermoacoustic combustor (Bonciolini et al., 2018)**
| Field | Verified value |
|---|---|
| DOI | `10.5061/dryad.4cj4k` (dataset id 865, version id 869) |
| Files | `Experimental_time_traces_tdms.zip` — 302,008,984 B — MD5 `82cc5298c245ad509e1ee414e2c34941` — file id 5331 |
| Format | application/zip; archive of **TDMS** (NI) files; sections `2016-10-20_StationaryX` (7 stationary operating points) and `2016-10-20_Ramp1Y` (ramping experiment sections) |
| Licence | **CC0-1.0** (SPDX `https://spdx.org/licenses/CC0-1.0.html`) |
| Published | 2018-02-21 |
| Paper | DOI `10.1098/rsos.172078` (R. Soc. Open Sci. 5(3):172078) |
| Bifurcation | stochastic **subcritical Hopf** (thermoacoustic instability); rate-dependent transition delay |
| Known unknowns | internal TDMS channel layout, sample rate, units — to be inspected at implementation and recorded in config (`expected_columns` is *validated against the file*, never assumed) |

**DaphniaExt — Daphnia magna extinction microcosms (Drake & Griffen, 2010)**
| Field | Verified value |
|---|---|
| DOI | `10.5061/dryad.q3p64` (dataset id 25258, version id 25331) |
| Files | `data-and-code.zip` 13,273,254 B MD5 `923e08e6bafae412c43250fef0752ac7` (id 89639); `README_for_data-and-code.txt` 4,848 B MD5 `11770ce4f5a2b5369dce6f6f846340c5` (id 89640) |
| Licence | **CC0-1.0** |
| Published | 2015-01-28 |
| Paper | DOI `10.1038/nature09389` (Nature 467:456–459) |
| Bifurcation | experimentally induced **transcritical**; CSD signatures ~110 days (~8 generations) pre-extinction; constant-environment populations = controls (null class) |
| Known unknowns | zip contents / column layout (Excel? CSV?) — README must be parsed at implementation; expected columns validated against actual files |

**Dryad API facts the ingestion must respect (verified from official docs):**
- Base `https://datadryad.org/api/v2`; DOI must be URL-encoded (`doi%3A10.5061%2Fdryad.4cj4k`).
- Flow: `GET /datasets/{doi}` → `_links.stash:version.href` → `GET /versions/{id}/files` → per-file `{path, size, mimeType, digest, digestType(md5), stash:download}`.
- **File download requires bearer auth** (API token). Anonymous: metadata OK, 30 req/min.
- Therefore: (a) auto-download with `DRYAD_API_TOKEN` env var, or (b) manual drop into `raw/` + checksum verification against the anonymous metadata API or config-pinned MD5s above.

---

## 5. Datasets — Ingestion & Processing

### 5.1 Per-dataset config (`configs/dataset/{tac,daphnia_ext,synthetic_*}.yaml`, structured dataclass)
`name, source(dryad|manual-ref|synthetic), doi, api_url, expected_files[{path,size,md5}], download{mode,env_token_var,retries,backoff}, archive_type, parser(nptdms|zip-csv|...), bif_type, transition_annotation{source,note}, processing{params...}, split{seed,fractions,replicate_based}, output_contract{channels,min_length}`.

### 5.2 Ingestion state machine + error taxonomy
States: `UNKNOWN → RESOLVE → FETCH_MANIFEST → VERIFY_EXPECTED → DOWNLOAD_OR_WAIT → VERIFY_CHECKSUM → EXTRACT → READY_RAW → PROCESS → SPLIT → READY_PROCESSED` (skip-if-manifest-valid at entry; per-dataset file lock so concurrent runs never double-download).

Structured error codes (CLI exit + ledger `status=failed` + log):
`INGEST_NETWORK, INGEST_AUTH, INGEST_MANIFEST_MISMATCH, INGEST_CHECKSUM, INGEST_ARCHIVE, PROCESS_NONFINITE, PROCESS_SHORT_LENGTH, PROCESS_ANNOTATION_MISSING, SPLIT_IMBALANCE, MANIFEST_CORRUPT`.

### 5.3 Download modes (both checksum-verified)
- **auto (token)**: `DRYAD_API_TOKEN`; `GET /versions/{id}/files` → download via file's `stash:download`; retries ×3 exp backoff; no re-download if raw exists + md5 matches.
- **manual**: prints exact file names + server MD5s (pinned in config); polls up to `wait_minutes` for files to appear in `raw/`; verification against config-pinned MD5; READY_RAW only on match.
- Offline mode: if `final_data/<name>/processed/manifest.json` valid → skip straight to READY_PROCESSED.

### 5.4 Processing contract (per dataset)
Mandatory gates after processing (each raises with PROCESS_* codes, never silently fixes):
1. No NaN/Inf anywhere; dtype fixed; shapes match manifest.
2. Every trajectory length ≥ max indicator window (100 for DFA) — policy: truncate-with-mask vs filter must be decided per dataset and recorded in manifest; failing trajectories are excluded with counts logged (never dropped silently).
3. Annotation spot-check: transition source (paper figure/page/section) recorded in config; processed `bifurcation_times` must be consistent with the paper statement (e.g., Daphnia: ~110 days pre-extinction).
4. Split balance asserted (positive/negative replicate counts logged; no stratification tricks — real replicates are few, splits go at replicate level).
5. Leak-free normalization: fit statistics on **train only**; applied to val/test; parameters recorded in manifest.

**TAC**: unzip → parse TDMS (npTDMS) → per operating point extract acoustic pressure channel → dominant-mode envelope (Hilbert amplitude within the combustor eigenfrequency band) → long traces chunked into analysis windows? (chunking policy verified at implementation; recorded in config) → leak-free z-score → replicate-based split (chunks/experiments as units) → manifest.
**DaphniaExt**: unzip → parse README (treatment coding) → per-replicate daily counts → treatment labels (deteriorating = signal, constant = null) → alignment on transition time (t=0 at extinction/transition per paper) → replicate-based split → manifest.
**Synthetic**: wrappers around the existing generators; same processed-dir pipeline; identical manifest contract (provenance `generator=classic|bury`, difficulty, params).

### 5.5 Manifest (`final_data/<name>/processed/manifest.json`)
`schema_version, dataset{doi,version_id,files[{path,md5}],licence,retrieved_at}, processing{git_sha,params,package_version}, gates{passed:[...]}, split{policy,seed,counts}, normalization{fit_on_train:true,params}, content_hash`.

### 5.6 Registry contract
`get_dataset(name, overrides) -> {"features", "seq_lengths", "bifurcation_times", "is_positive", "split_indices", "timestamps"?, "meta": {bif_type, source, licence, n_replicates}}` — models/evaluation branch on `meta`, never on dataset name.

---

## 6. Models

### 6.1 Common interface (`models/common/`)
`MethodMeta{name, family(indicator|spectral|neural), is_learned, scope_caveat, bif_types_supported}`; protocol `fit(train, val, cfg)`, `score(arrays, cfg) -> (B,T) float32`, `meta`. Registry `get_method/list_methods/validate_names` supersedes `benchmark/methods.py`.

### 6.2 Statistical indicators — relocation only
Seven `indicator.py` modules move to `models/indicators/<name>/`; zero behavior change; existing per-indicator tests re-run against new paths.

### 6.3 Spectral-drift observer — physics untouched
Wrap to the interface; `fit` no-op; `scope_caveat = "theoretically grounded for fold; empirical elsewhere"`. Robustness item: TAC/Hopf long high-rate series → document chunking/finite-precision policy in `models/spectral_drift` README; verified during real-data runs.

### 6.4 Neural baselines (2–3, families specified; exact picks recorded as an implementation task with citations)
Families (real, literature-grounded, verifiable):
- **Recurrent alarm network** — LSTM/GRU encoder + alarm head, BCE on late-window positives (classical DL CSD baseline).
- **Temporal-convolutional encoder (TCN)** — dilated causal convs; long-series friendly.
- **Encoder-only transformer / patch-based TS classifier** (e.g., PatchTST-style) — modern SOTA-adjacent family.

**Fairness contract (mandatory)**: identical input pipeline & splits as indicators; same val-null calibration rule; hyperparameters fixed per method (no tuning on test); capacity budget within a defined parameter range; all seeds run and reported (no cherry-picking); training logs + checkpoints persisted. Exact architectures + fact-checked paper citations for the 2–3 chosen families are pinned in the ledger (L4.5) and implemented under L4.6 — no fabricated results.

---

## 7. Training (Lightning; DL baselines only)
- `training/common/trainer_factory.py`: deterministic seeding (`torch.manual_seed`, `deterministic_mode`, `seed_everything` incl. dataloader workers), early stopping on val loss, checkpoint → `artifacts/checkpoints/`, LR scheduler from config.
- `training/common/callbacks.py`: `LedgerCallback` (stage timings → `times/timings.json`; status transitions), `FingerprintCallback` (env+versions at fit start).
- `training/common/losses.py`: alarm BCE (+ persistence-aware smoothing; finalized per selected family in L5.3 + L4.6).
- Trainers receive `OutputWriter` handle; they never construct paths.

---

## 8. Evaluation — Persistence-Aware Protocol (formal)

### 8.1 Definitions
Given per-trajectory score stream `s_{t} ∈ [0,1]` and threshold `θ` (calibrated so a *step-wise* null alarm rate = `fpr_target` on val-null):
- **Alarm indicator** `a_t = 1[s_t ≥ θ]`.
- **Persistence statistic (causal)**: `r_t = run length of consecutive alarms ending at t` (i.e., `r_t = r_{t-1}+1 if a_t else 0`).
- **Persistent alarm** at `t` iff `r_t ≥ k_persist` (config; default 5). Filter is **causal** (uses only `s_{≤t}`), so persistent detection time remains a valid online early-warning quantity.

### 8.2 Persistence-aware detection-time (P-DT) and censoring
`P-DT = min{ t : r_t ≥ k_persist }` over the positive trajectory, measured relative to `bifurcation_time` (same convention as current DT). **Censoring policy (mandatory)**: trajectories without a persistent alarm before end are censored; reports `detection_rate` (fraction detected) *separately* from `detection_time` (mean/median over **detected only**); aggregation with censored rows averaged in as `∞` is forbidden. Same for nulls: `null persistent-alarm rate` per trajectory.

### 8.3 Persistence-aware early-warning AUC (P-EW-AUC)
Generalization of the current EW-AUC: per-step `persistent_alarm_t ∈ {0,1}` in the early-warning window vs null terminal windows → AUC of the persistent-alarm indicator over (positive early window ∪ null window) pooled steps; i.e., how well the **persistent** alarm separates pre-transition positives from nulls. Identity note (precise): at `k_persist=1`, P-EW-AUC equals the classic step-alarm AUC — i.e., the legacy EW-AUC **evaluated on the thresholded alarm stream `a_t`** — exactly. The legacy raw-score EW-AUC is kept as a separately reported metric for continuity with earlier results; the two coincide only when scores sort identically to alarms at the calibrated threshold.

### 8.4 Null anchor (sanity control, not the calibration mechanism)
For i.i.d. null steps with per-step alarm prob `p`, positive association (FKG) gives `P(no persistent alarm in a W-step window) ≥ (1-p^{k_persist})^{W-k_persist+1}` — hence an analytic upper bound on the probability of at least one persistent alarm per window, `≤ 1 - (1-p^{k_persist})^{W-k_persist+1}`, which must bracket the empirically measured null persistent-alarm rate within tolerance; autocorrelated nulls are handled by **empirical** val-null calibration (the mechanism), the bound is only a cross-check reported in `metrics/protocol_checks.json`.

### 8.5 Achieved-FPR check
Threshold θ calibrated on val-null; **achieved test-null step-FPR and persistent-FPR must be reported** next to the target; deviation > 2× target triggers `PROTOCOL_DRIFT` warning in logs (never silently recalibrates on test).

### 8.6 Aggregation & significance
- Per (system/replicate, method, seed) rows written as-is.
- Aggregates: mean ± std over seeds; **bootstrap 95% CIs** over trajectories within a seed (and across seeds for real data); pairwise method comparisons on identical data via **paired Wilcoxon signed-rank** on per-trajectory P-DT; all in `tables/` via `outputs.summarize`.
- Detection-rate and P-DT never merged into one scalar.

### 8.7 Governance driver (single implementation)
`preprocess → score → calibrate(val-null) → persist(k_persist) → metrics(P-DT, P-EW-AUC, FPRs, censoring) → artifacts → row`. All methods (indicators, spectral, neural) pass through this one driver; `evaluation/` never imports `models/` (methods injected via interface).

---

## 9. Outputs

### 9.1 Canonical tree (final)
```
outputs/<run_name>/<timestamp>/
    resolved_config/   resolved.yaml + cli_overrides.yaml
    metadata/          environment.json (git sha, python, torch, hydra, datasets hashes, host)
    metrics/           metrics.json, protocol_checks.json
    results/           results.jsonl (+ epoch_logs/ per trained method, trajectories/ if enabled)
    artifacts/         checkpoints/, calibration/, plots/
    logs/              run.log, hydra.log
    times/             timings.json
    tables/            aggregates.csv, cIs.csv (via summarize)
```

### 9.2 One timestamp, one run; lifecycle
Timestamp minted once at writer construction from run start; **assertion in writer** that `self.root` never re-stamps mid-run. Lifecycle markers: `<timestamp>.pending` at start → renamed `.completed` on success, `.failed` on exception (partial artifacts preserved); ledger transitions mirror it. Hydra `--multirun` = N jobs → N distinct timestamps under one `run_name` (each job writes its own `.pending/.completed`).

### 9.3 Writer / ledger / schema / tables
- `writer.py`: full tree creation (`exist_ok=False`, collision = loud error); atomic JSONL append (tmp+rename); schema-validated rows only.
- `ledger.py`: append-only `outputs/_ledger/runs.jsonl` (+ `index.json` rebuilt); row: `run_id, timestamp, run_name, dataset, methods, k_persist, config_hash, git_sha, status, path`.
- `schema.py`: row dataclass (run_id, dataset, bif_type, system, replicate, method, family, is_learned, detection_rate, detection_time, ew_auc, fpr, persistent_fpr, k_persist, threshold, params, hashes); unknown keys rejected.
- `tables.py`/`summarize.py`: aggregates + bootstrap CIs + paired Wilcoxon — paper-ready CSVs.

### 9.4 Artifact richness policy (config-controlled, no unbounded writes)
`output.store_trajectories: false` by default (the historical per-(system,method,seed) npz writes are gated behind this flag); checkpoints only for trained methods; every byte written is accounted for in `times/timings.json`.

---

## 10. Configuration (Hydra + OmegaConf)

### 10.1 Groups (ConfigStore, **structured dataclasses** — schema errors at compose time, not runtime)
`dataset{tac,daphnia_ext,synthetic_fold,synthetic_hopf,synthetic_logistic}`, `model{spectral_drift,7 indicators, neural_*}` (+ `models:` multi-select for sweeps), `training{default,none}`, `evaluation{persistenceaware, baseline_classic}`, `run{...composed study configs}`, `output{default}`.

### 10.2 Composition & overrides
Full config = `+dataset=… +models=[…] +evaluation=… +training=…`; CLI overrides always win; unknown registry names fail fast; dataset overrides are whitelisted keys only.

### 10.3 Validation invariants (fail fast at compose time)
| Invariant |
|---|
| `model.is_learned ⇒ training != none` |
| `evaluation.k_persist ≥ 1`; `fpr_target ∈ (0,1)` |
| `dataset ∈ registry`; every `model ∈ registry` |
| `split.replicate_based = true` for real datasets |
| `training.epochs ≥ 1`; `data.split.fit_stats = train-only` |

### 10.4 Secrets
`DRYAD_API_TOKEN` via env only, never in config files; config (and stored resolved.yaml) must be token-free.

---

## 11. Orchestration & CLI
`cli/main.py`: `@hydra.main(config_path=..., config_name="run")` → compose → validate → writer (mints timestamp, writes resolved config + cli overrides + metadata fingerprint) → `orchestration/runner.py` (registry resolution → per-method governance via interface → ledger). Runner contains pipeline order only. `studies/runner/benchmark.py` shims to the CLI until removed.

---

## 12. Test & Verification Strategy
- Per-package tests mirroring layout; existing indicator tests re-run at new paths (parity gate).
- Property tests: persistence filter monotone in `k_persist` (FPR↓, P-DT↑); `k_persist=1 ≡ classic`.
- Determinism tests: same seed → identical metrics (torch/numpy seeds, `num_workers=0`).
- Ingestion tests with **mocked Dryad API** (recorded response fixtures incl. files+md5); checksum mismatch, auth failure, manual-mode timeout paths.
- E2E smoke: `+dataset=synthetic_fold +models=var_csd +evaluation=persistenceaware n_seeds=1` → assert tree, ledger row, schema-valid rows, no network.
- CI: `ruff check .`, `pytest -v`; heavy runs tagged and excluded from CI.

---

## 13. Reproducibility Commitments
Pinned deps (`hydra-core`, `omegaconf`, `pytorch-lightning`, `torchmetrics`, `requests`, `nptdms`, `filelock`); lockfile committed; git-sha + env fingerprint written per run; seed schedule preserved (`seed_offset + s*1000 + 101/202`); processed data hash in manifest so a changed pipeline invalidates runs, never silently reuses stale processed data.

---

## 14. What Does NOT Change
- Spectral-drift observer math; the seven indicator implementations; fixed-FPR calibration principle; the per-seed schedule; the two real datasets' provenance (CC0-1.0, DOIs above).

---

## 15. Data Accessibility (corrected, retrievable without requests)
| Dataset | Dryad DOI | Licence | Immediate download | Size |
|---|---|---|---|---|
| TAC (Bonciolini et al. 2018) | `10.5061/dryad.4cj4k` | **CC0-1.0** | Yes (web UI; API needs token) | ~288 MB |
| DaphniaExt (Drake & Griffen 2010) | `10.5061/dryad.q3p64` | **CC0-1.0** | Yes | ~13 MB |