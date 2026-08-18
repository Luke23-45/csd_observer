# Experimental Setup — Report for Supervisor Review

Prepared 2026-08-19, companion to the round-3 revised
`docs/final/literature_review.md`. This report describes the **implemented**
experimental setup (data, baselines, architecture) as it exists in the
repository today, the changes planned in response to your round-3 feedback,
and the specific points where we need your guidance.

---

## 1. Contribution (framing)

The contribution is an **evaluation protocol**, not a new detector: a
*causally persistent* alarm rule (a threshold crossing only counts when the
score stays above threshold for `k` consecutive steps) applied uniformly to
every benchmark method, and a benchmark that ranks methods at **matched
false-alarm operating points**. The protocol and all artifacts live in
`src/csd_observer/evaluation/persistence/`.

## 2. Data

All arrays follow one bundle contract
(`datasets/common/contract.py`):
`features (B,T,C)`, `seq_lengths (B,)`, `bifurcation_times (B,)`,
`is_positive (B,)`, `split_indices {train,val,test}`. Every dataset provides
a matched **signal** set and a **null** set (parameter held at its start
value, same noise streams).

### 2.1 Synthetic (project-generated, seed 42)

| dataset | dynamics | ramp | tau | drive/obs noise | mode | n / length |
|---|---|---|---|---|---|---|
| `synthetic_fold` | saddle-node `x' = r - x²` | `r: 2.0 → -1.0` | 133 | 0.30 / 0.10 | channel_0 | 500 / 200 |
| `synthetic_hopf` | supercritical Hopf `z' = (μ+iω)z − \|z\|²z`, ω=0.1 | `μ: -0.5 → 0.5` | 100 | 0.05 / 0.15 | radial (2 ch) | 500 / 200 |
| `synthetic_logistic` | logistic map `x' = μx(1-x)` | `μ: 2.5 → 4.0` | 67 (μ=3.0) | 0.02 / 0.05 | channel_0 | 500 / 200 |

Generators: `datasets/synthetic/common/generators.py` (per-trajectory
sub-RNG streams, obs noise on independent streams; fold trajectories that
crash far before τ are rejected). A randomized "Bury-style" tier
(`RandomizedBifurcationDataset`, perturbation polynomials, AR(1) coloured
noise, per-trajectory parameter draws) exists in code but is **not** in the
current configs (only `generator: classic` is used).

### 2.2 Real (Dryad, CC0-1.0)

- **TAC** (`tac`, dryad 10.5061/dryad.4cj4k) — 2016 thermoacoustic-combustor
  experiments, **subcritical Hopf**. Processed (TDMS → dominant-mode envelope,
  Mic1/Set0) into **673 windows: 10 signal ramps + 663 null chunks** at
  `analysis_window=4096` samples. Ramp onset is **auto-detected** from the
  envelope; each ramp becomes one window aligned to its onset
  `[onset-2048, onset+2048)`. Split 404/135/134.
- **DaphniaExt** (`daphnia_ext`, dryad 10.5061/dryad.q3p64) — 60 replicates
  (**30 extinction + 30 non-extinction**), **transcritical**. Restart
  populations (H7/H9/J4/K2/K10) recoded at day 154; τ = extinction day − 110
  (the Nature 467:456 CSD window). `min_length=20` gate excludes 5 short
  populations. Split 36/12/12.

Both are provisioned by a processor pipeline with content-hashed manifests
and a staleness guard (`datasets/common/pipeline.py`); processed bundles are
on disk under `datasets/processed/{tac,daphnia}`.

### 2.3 Splits and seeds

Replicate-based split (train 0.6 / val 0.2 / test 0.2, seed 42); real-data
splits must be `replicate_based` (validated). Per-seed schedule (§13):
`run_seed = seed_offset + s·1000 + {101 signal, 202 null}` — there is no
`seed` knob. `n_seeds=3` (`synthetic_matrix`/`real_matrix`), `seed_offset=0`.

## 3. Baselines

Registered in `models/common/registry.py` (10 methods). Each is a fresh
instance per (method, system); every method exposes
`fit(train,val,cfg) → score(features,seq_lengths,cfg) → (B,T) float scores`.

### 3.1 Statistical indicators (pure NumPy, deterministic, window-based)

VAR-CSD, AC1-CSD, SKEW-CSD, SRATIO-CSD, RETRATE-CSD (window 30),
DFA-CSD (window 100), DMD-CSD (window 30, embedding_dim 6, rank 2).
NaN warm-up of `window-1` steps; scores are per-step alarm strengths.
Scope caveat: theoretically grounded for fold-type slowing down, empirical
elsewhere (known inversions on Hopf for AC1/SRATIO/RETRATE/DFA).

### 3.2 Neural (causal, per-step alarm probability, trained)

| method | architecture | config |
|---|---|---|
| `LSTM-AlarmNet` | LSTM encoder (64) + per-step MLP head | hidden 64, 1 layer, dropout 0.1 |
| `TCN-AlarmNet` | causal dilated conv, kernel 5, 3 levels | hidden 32 |
| `PatchTST-AlarmNet` | patch transformer (patch 16, d_model 64, 2 heads, 2 layers), **causally masked attention**, score from latest completed patch | stride=patch_len |

Training (identical for all): AdamW lr 1e-3, weight-decay 1e-5, cosine
schedule, early stopping patience 5, ≤50 epochs, batch 64; alarm targets =
late window `[τ-60, τ)` on positives (nulls all-zero);
**persistence-aware loss** `alarm_bce` = α·BCE + (1−α)·BCE on the causal
running mean of `k_persist` of the probabilities (k folded into the loss).
Determinism: `torch.use_deterministic_algorithms(True, warn_only=False)`
(hard error on non-deterministic kernels; documented escape hatch env).
`label_window=10` is auto-applied for daphnia (`runner/commands.py`).

### 3.3 Planned RQ3 additions (not yet implemented)

Dominant-eigenvalue (DEV) method, a deep-learning EWS baseline, a
Boettiger–Hastings likelihood-ratio detector, and TIPMOC-style variance
forecasting (Masuda 2026).

## 4. Evaluation architecture

### 4.1 Governance pipeline (single driver for every method)

`evaluation/persistence/governance.py::evaluate_method`:

```
fit(train+val) → score(val, val-null, test, test-null)
  → validate scores (B,T) float; NaN warm-up OK, ±Inf raises
  → calibrate threshold on val-null
  → persist(k_persist) → metrics → artifacts → schema-validated row
```

The driver imports no method implementation (duck-typed handle); `evaluation`
never imports `models`. Import layering is CI-enforced.

### 4.2 Persistence protocol (`evaluation/persistence/protocol.py`)

- alarm `a_t = 1[s_t ≥ θ]`; persistence statistic `r_t` = causal run length
  of consecutive alarms ending at t (online/valid for early warning).
- **persistent alarm** when `r_t ≥ k_persist`.
- **P-DT** = first persistent alarm before τ, i.e. lead `τ − t*`; censored
  otherwise (censoring reported separately).
- **P-EW-AUC** = AUC of the persistent-alarm stream over the positive early
  window `[τ−50, τ−5)` ∪ null window `[T−50, T−5)`; at k=1 it equals the
  classic EW-AUC by construction (ablation identity).
- **persistent-FPR** (step) and **persistent trajectory-FPR** (fraction of
  null trajectories with ≥1 persistent alarm).
- **Null anchors (FKG)**: for i.i.d. null steps with per-step alarm prob p,
  `P(no persistent alarm in a W-step window) ≥ (1 − p^k)^(W−k+1)`, giving an
  analytic upper bound on the per-window persistent-FPR. Protocol checks
  attest `anchors_bracket_empirical` and gate drift at 2× target/anchor
  (`protocol_checks.json`).

### 4.3 Calibration — **current vs planned (your round-3 point)**

**Current (implemented)**: `evaluate_method` uses
`calibrate_threshold` = `100·(1−fpr_target)` percentile over the **pooled
per-step** val-null scores (default `fpr_target=0.05`). This is the
per-step design you flagged: the same threshold applied across K makes the
persistent-FPR collapse with K (per-step p≈0.05 ⇒ K=10 per-trajectory FPR
≈ 0.05¹⁰ ≈ 10⁻¹³, ARL₀ ≈ 10¹³ steps), so a K0-vs-K10 ranking conflates
persistence with conservativeness.

**Planned**: per-K threshold calibration to a **common per-trajectory null
budget** — for each K, pick θ from the val-null so that ≈5% of null
*trajectories* alarm, then compare methods/K across matched operating
points. The protocol already computes `persistent_trajectory_fpr` and the
`trajectory_fpr_anchor`, so the machinery exists; the calibration function
is the piece to replace.

### 4.4 Outputs & provenance (`outputs/writer.py`)

One timestamped run dir `outputs/<run>/<timestamp>/` with canonical
subtree: `resolved_config/`, `metadata/` (env + dataset content-hash),
`metrics/` (metrics.json, protocol_checks.json), `results/` (results.jsonl,
atomic append, schema-validated rows), `artifacts/` (checkpoints,
calibration), `logs/`, `times/`, `tables/`. Lifecycle markers
`.completed/.failed` plus a global ledger `outputs/_ledger/runs.jsonl`.

### 4.5 Runner, config, validation

`csd-observer` CLI (Hydra; `configs/run.yaml` defaults; group yamls in
`configs/{dataset,model,training,evaluation,output}/`). Fail-fast
validation gates (`config/validate.py`): registry names, group blocks vs
schema nodes, generator knobs only via `+dataset_overrides.*` whitelist,
`n_trajectories≥3`, DFA `max_length≥100` (DFA-method-aware, so real runs
below 100 validate), `label_window ≤ min dataset length`, TAC early-window
fits the aligned ramp window, seed-schedule int32 overflow. A `csd-sweep`
CLI builds the per-method train→eval commands
(`runner/commands.py`); `experiments/*.json` define the matrices.

## 5. Experimental design → research questions

- **RQ1** — rank methods at matched operating points (P-DT / P-EW-AUC at
  fixed trajectory-FPR).
- **RQ1b (planned)** — persistence sweep K ∈ {1,2,3,5,10} at matched
  per-trajectory FPR (interaction of K × threshold).
- **RQ2** — persistence length × threshold interaction / cost of
  persistence (lead vs false-alarm suppression).
- **RQ3 (planned)** — does persistence reduce *transient* false alarms on
  **hard-negative** systems (changing but not bifurcating, e.g.
  pseudo-bifurcations / transient amplification, cf. Troude et al. 2026)?

Matrices: `synthetic_matrix` (fold: 7 indicators + 3 neural; hopf/logistic:
VAR, AC1 + 3 neural), `real_matrix` (tac: VAR, AC1, DMD + 3 neural;
daphnia_ext: VAR, AC1 + 3 neural), seeds [0,1,2].

## 6. Preliminary runs (smoke, seed 0, current per-step calibration)

- TAC `DMD-CSD`: detection_rate 1.0, P-DT lead ≈ 1807 samples, but
  **persistent trajectory-FPR 0.96** on null windows — the per-step
  calibration does not control trajectory-level FPR on TAC nulls
  (illustrates the round-3 point directly).
- DaphniaExt `VAR-CSD`: detection_rate 0.0 (6/6 censored), EW-AUC 0.33.
- These predate the matched-calibration change; the full 3-seed matrices
  have **not** been run yet.

## 7. Status summary

| item | status |
|---|---|
| Synthetic generators (fold/hopf/logistic) | implemented |
| Bury/randomized tier | implemented (not wired into configs) |
| TAC + DaphniaExt processors, manifests, staleness guard | implemented, data on disk |
| 7 indicators + 3 neural baselines | implemented |
| Persistence protocol + FKG anchors + governance | implemented |
| Per-step calibration | implemented (**to be replaced**) |
| Per-K matched calibration | **planned** |
| K-sweep {1,2,3,5,10} | **planned** |
| Synthetic transcritical generator | **planned** (daphnia covers real transcritical) |
| Hard-negative systems (pseudo-bifurcation class) | **planned** |
| RQ3 baselines (DEV / DL / B&H / TIPMOC) | **planned** |
| 3-seed full matrices | **planned** (specs ready) |
| AMOC case study | follow-up, non-blocking |

## 8. Points where we need your guidance

1. **Per-K calibration target**: per-trajectory null-FPR budget (5%?) —
   confirm the budget and whether it should be per-trajectory or per
   early-window.
2. **K sweep resolution**: K ∈ {1,2,3,5,10} vs {1,3,5,10} — which is
   sufficient for the interaction analysis?
3. **Hard negatives**: which class of "changing-but-not-bifurcating"
   systems (pseudo-bifurcation / transient amplification) to prioritize,
   and whether to extend the classic or Bury generator tier.
4. **RQ3 baseline scope**: the four proposed baselines (DEV, deep-learning
   EWS, Boettiger–Hastings likelihood, TIPMOC-style) — any we can drop or
   must add.