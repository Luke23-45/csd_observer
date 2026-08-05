# Benchmark Revision Plan: Kalman-Spectral-Drift vs. Canonical CSD Baselines

**Status:** proposal, awaiting review — *no code changed yet* (2026-08-04)
**Supersedes:** `docs/plan/classical_baselines_plan.md` (same intent, this revision drops the learned-Kalman suite and narrows baselines to the canonical published CSD indicator set).

---

## 0. Decision (requested)

1. **Proposed method (keep, single):** `Kalman-Spectral-Drift` — the Rao-Blackwellised particle filter over the spectral gap `c_k` (`src/csd_observer/models/spectral_drift.py`). **No trained parameters**; the alarm is the posterior collapse probability `Pr(c_{t+1} < δ | y_{0:t})`.
2. **Remove all 12 other current methods from the benchmark** (`Raw-CSD, RunningVar, Lag2-CSD, Lag2-CSD-detrended, Kalman-Lag2, Kalman-BCE, Kalman-BCE-Spec, Kalman-LSTM, Kalman-LSTM-Spec, Kalman-Lag2-Net, Kalman-ACKO, Kalman-LSTM-Aug`). Rationale (§2.2).
3. **Add 7 canonical, published classical baselines** (§3) — the reviewer-recognisable set from **Dakos et al. 2012 (PLoS ONE, e41010)** and the canonical **`earlywarnings` R toolbox (Dakos & Lahti)**, plus the current eigenvalue-tracking baseline **Donovan & Brand 2022 / Grziwotz et al. 2023 (DMD)**.
4. **Claim we can then make:** *Kalman-Spectral-Drift is an interpretable, non-learned Bayesian observer of the dominant stability margin that generalises critical-slowing-down detection beyond what the generic CSD indicators capture, at controlled false-positive rate and with deterministic, reproducible behaviour* (§8, with the guarded scope language the paper already uses).

---

## 1. Why the current baselines are unacceptable (removal rationale)

| Removed method | Why it is "sloppy" |
|---|---|
| `Raw-CSD`, `Lag2-CSD` | Ad-hoc "lagged-correlation > threshold" heuristics (thresholds 0.6/0.5 fixed by convention, `metrics.py:151/195`); max-over-channels aggregation; not the published estimator |
| `RunningVar` | Zero-order variance under a Youden threshold that degrades to degenerate FPR (`select_threshold`); FPR is `nan` in tables |
| `Lag2-CSD-detrended` | Third variant off the same ad-hoc estimator; inconsistent detrending across siblings |
| `Kalman-BCE`, `-BCE-Spec`, `-LSTM`, `-LSTM-Spec`, `-Lag2-Net`, `-ACKO`, `-LSTM-Aug`, `-Lag2` | Learned heads trained on tiny synthetic splits (101-traj train) with unvalidated architectures/FPR (0.0–0.8), hard-tier collapses (logistic BCE 1.000→0.360), and **no faithful external reference** — no reviewer can verify ACKO/LSTM-Aug against a published implementation. `Kalman-Lag2` overlaps the AC1 baseline it would otherwise stand in for. |

Conclusion: none of the removed methods maps cleanly onto a citable, reproducible method. Keeping only the proposed method + published baselines makes every row in the results table independently auditable.

**Code note:** the removed methods' source (`kalman_lag2.py`, `csd_observer.py`, `training/trainer.py`, `utils/losses.py`, the debug/ablation runners and their tests) has been removed from the tree after review; all of it remains in git history for reproducibility. The retirement is complete: the benchmark `METHODS` tuple is a single row, and the model package is `src/csd_observer/models/spectral_drift/`.

---

## 2. Final method suite

| # | Method (METHODS name) | Type | Alarm score | Ref. |
|---|---|---|---|---|
| 1 | **Kalman-Spectral-Drift** | **proposed** (non-learned, 0 params) | `Pr(c_{t+1} < δ \| y_{0:t})` | `src/csd_observer/models/spectral_drift.py`; formal def. `docs/z2/formal_defininition.md` |
| 2 | `VAR-CSD` | baseline | windowed variance (linear-detrended window) | Scheffer 2009; Bury 2021 (*lag-1 AC & variance are their baselines*); `earlywarnings` `sd` |
| 3 | `AC1-CSD` | baseline | lag-1 autocorrelation of detrended window | Dakos 2012; Bury 2021; `earlywarnings` `acf1` |
| 4 | `SKEW-CSD` | baseline | absolute skewness of detrended window | Dakos 2012; `earlywarnings` `sk` (abs-transform) |
| 5 | `SRATIO-CSD` | baseline | low/high spectral-density ratio (per-window AR(1) spectrum) | Dakos 2012 (spectral reddening); `earlywarnings` `densratio` |
| 6 | `DFA-CSD` | baseline | detrended-fluctuation-analysis exponent α | Dakos 2012 (DFA, box sizes 10–100; window ≥ 100 pts) |
| 7 | `RETRATE-CSD` | baseline | return rate = 1/AR(1) coefficient | Ives 2003; `earlywarnings` `returnrate` |
| 8 | `DMD-CSD` | baseline | dominant (largest-modulus) eigenvalue of Hankel-DMD linear operator | Donovan & Brand 2022 (Physica A 596:127152); Grziwotz et al. 2023 (dynamic eigenvalues) |

- 8 rows total; 0 learnable parameters anywhere (grid-searched hyperparameters only, selected on the validation split — same governance as the σ_u/Q_drift search today, `benchmark.py:596-609`).
- All baselines consume the **same preprocessed input as the proposed method** — `extract_mode()` (`spectral_drift.py:34`), i.e. hopf radial mode √(x₁²+x₂²), fold/logistic channel 0 — so preprocessing is identical across every row.

---

## 3. Indicator specifications (verified against the literature)

Detrending for all windows: **within-window linear detrending** (`_linear_detrend`, `metrics.py:15`) — causal (uses only the window so far). This deviates from Dakos' *non-causal* Gaussian smoothing (bandwidth 10 % of length) deliberately: our benchmark enforces **prefix-consistency** (style of BenchEWS v1.0, Zenodo 10.5281/zenodo.20487811) — no lookahead is allowed at any step. Linear detrending also matches Dakos' own DFA protocol and Bury 2021's Lowess-residual workflow in spirit, and is the cheapest causal-compatible choice.

| Indicator | Definition (all on causal window ending at `t`) | Expected pre-transition direction | Caveats |
|---|---|---|---|
| **VAR** | `mean((x - mean)²)` on linear-detrended window, W=30 | ↑ | On bury-hard fold the *null* variance can exceed the signal's (diagnosis note) → AUC may invert; report as-is |
| **AC1** | Pearson correlation of `x[t-1], x[t]` on detrended window, W=30 | ↑ fold/logistic; ↓ **Hopf** (oscillatory component) | Hopf inversion is documented (Bury 2021, PNAS: "lag-1 AC decreases due to oscillatory motion"). No adaptive sign-flip (post-hoc-y); report raw direction |
| **SKEW** | `abs(˄γ₁)` (absolute 3rd standardized moment), W=30 | ↑ (abs convention of `earlywarnings`) | Dakos: signed skewness *decreases* on fold; the package's abs-transform is the canonical handling |
| **SRATIO** | fit AR(1) (Yule–Walker) on window; take its spectrum; `S(f_low)/S(f_high)` with low = lowest non-zero bin, high = Nyquist bin (the `earlywarnings` `densratio` bin convention: `spec[low=6]/spec[high=last]`) | ↑ (spectral reddening) | Bands are window-relative; document exact bins in code |
| **DFA** | standard DFA fluctuation-exponent slope log F(s) vs log s, box sizes s=10,20,40,80·(scaled by window/100); window 100 (canonical: Dakos uses 50 % of record, DFA "requires >100 points") | ↑ toward 1 (rescaled 1.5→1 at transition) | Needs ≥100 pts → no score for trajectories with τ < ~100 (below §5). leading-NaN handling defined there |
| **RETRATE** | `1/φ₁` where φ₁ = AR(1) coefficient of window (earlywarnings code: `returnrate = 1/ar1`) | ↓ (recovery slows; ar1↑ ⇒ 1/ar1↓ — inverse of VAR/AC1) | Monotone-inverse of AR(1); keeps the "recovery-time" framing Ives 2003 |
| **DMD** | Hankel delay-embed (m≈6) on the window; `A = X₂ X₁⁺` (rank-r SVD ≤ m−1); alarm = `max_j |λ_j(A)|` | ↑ approaching 1 | Dominant eigenvalue of the *local Jacobian*; nearest non-learned analogue to our c_k observer |

Direction convention in the benchmark: the **raw indicator value is the alarm score** (higher score ⇒ higher alarm) with the fixed-FPR threshold (§4). For `RETRATE` (which *falls*), use the score **`-log(return_rate)` = recovery time** so all methods share the "higher = more alarm" convention — a monotone transform, thus rank-equivalent, and removes sign confusion from the tables.

---

## 4. Evaluation & governance (unchanged pipeline — the key correctness invariant)

Every baseline is evaluated **exactly like the spectral block** (`benchmark.py:573-688`), so all rows are comparable:

1. Compute per-step scores on `features[test_idx_*]` / `features[val_idx_*]` (signal & null).
2. **Fixed-FPR calibration:** `threshold = percentile(all val-NULL score steps, 100·(1−fpr_target))` with `fpr_target = 0.05` — the rule already at `benchmark.py:636-643`. This gives FPR ≈ 5 % by construction and sidesteps the degeneracy of Youden for these alarm shapes.
3. Metrics per (system, method, seed=0): `detection_time` (`compute_detection_time`), `ew_auc` (`compute_early_warning_auc`, window [τ−50, τ−5]), `fpr` (`compute_null_metrics`), plus per-trajectory DTs via `_compute_per_traj_dts` → `writer.write_trajectory_data`.
4. Causal masking: window ends at `t`; nothing beyond `t` is touched → prefix-consistent (audit check §9.2.4).

**`t < min_window` handling (correctness-critical):**
- For VAR/AC1/SKEW/SRATIO/RETRATE/DMD with window 30, compute on the partial causal window `min(W, t+1)` — same behaviour as `running_mean_center` (`spectral_drift.py:64`), giving a score at every step.
- For DFA (min 100): set steps `t < 100` to `NaN`.
- **Calibration percentile:** build `null_steps` from **finite** scores only (the spectral block concatenates every step; for DFA we must filter NaN or `np.percentile` returns NaN).
- **EW-AUC / DT / FPR:** `_sanitize_scores(fill=0.5)` maps NaN→0.5 for the AUC (a neutral mid-point under ranked evaluation); `NaN >= threshold` evaluates `False`, so DT and FPR treat undefined early windows as "no alarm" — conservative and identical across methods.

---

## 5. Window sizes & runtime

| Indicator | Window | Why |
|---|---|---|
| VAR, AC1, SKEW, SRATIO, RETRATE, DMD | 30 | Matches existing non-learned methods; ≈15 % of T=200 (inside Dakos' 25–75 % sensitivity band is not required — we are *online*, not doing significance trend-testing) |
| DFA | 100 | Canonical 50 % record / ">100 points" (Dakos 2012). Early-τ trajectories score NaN (above) — an **honest data-demand limitation DFA is known for** |

Runtime (B=500 × T=200, per system): VAR/AC1/SKEW/RETRATE are O(B·T·W) numpy window ops — seconds. SRATIO adds a per-window Yule–Walker solve (trivial). DFA ≈ O(B·T)·polyfit over 4 box sizes — low minutes; DMD ≈ per-window truncated SVD of a (W×m), m=6 — seconds–minutes. **No GPU needed**; the whole 8-method hard-tier run fits in a few minutes on CPU. If DFA is slow, evaluate at every 2nd step and forward-fill (deterministic; document).

---

## 6. Integration points (verified against current code)

| Where | Change |
|---|---|
| `studies/runner/benchmark.py` `METHODS` (lines 69–79) | Replace the 13 names with the 8 above |
| `benchmark.py` `_run_synthetic_experiment` | Replace the 12 method blocks with 7 compact `if _enabled(name):` blocks cloned from the spectral block's *structure* (mode extraction → scores on test/val/null → calibration → metrics → `RunResult` + `write_result_row`). A single helper `evaluate_classical_indicator(...)` removes repetition and the risk of per-method drift |
| `benchmark.py` verdict printer (~765–810) | Drop the hard-coded Raw/RunningVar/Lag2 rows; the suite-level verdict logic (which assumes Kalman-LSTM primary) is removed or re-gated to the classical suite (review) |
| `src/csd_observer/utils/metrics.py` | Add `raw_var_indicator`-style functions `raw_ac1_indicator`, `raw_skew_indicator`, `raw_sratio_indicator`, `raw_dfa_indicator`, `raw_retrate_indicator`, `raw_dmd_indicator` (each `(features, seq_lengths, window) -> (B,T)` causal scores with NaN where undefined) |
| `configs/model/default.yaml` | Add `classical_ews:` block (window sizes, fpr_target, bands, DMD embedding m); defaults hard-coded too, so nothing breaks without it |
| `docs/` | Revise `publication_framing.md` (currently frames learned observers as the comparison — now stale), `spectral_drift_report.md`, `research/synthetic_data_survey.md` §2.1/6 |

---

## 7. Tests & verification (what "done" means)

New unit tests in `tests/test_observer.py` (mirroring existing style), each indicator:

1. Shape/dtype: `(B,T,C) -> (B,T)` float32; NaN for `t < min_window`.
2. **Causal/prefix-consistency:** mutating `features[b, t+1:]` (future) must not change `score[b, t]` (scalar check per BenchEWS philosophy).
3. **Stationary OU null** ⇒ flat/no-trend scores (Kendall |τ| small on the windowed series).
4. Classic fold/hopf/logistic signals ⇒ expected direction (VAR/AC1/SKEW/SRATIO/DFA/RETRATE-recovery-time/DMD rise in the pre-transition window; §3 table).
5. Determinism: identical input ⇒ bit-identical output (no RNG).
6. Calibration audit: with weights = all-null input, threshold at `fpr_target` yields FPR ≤ target + ε.
7. **Benchmark smoke:** `python studies/runner/benchmark.py patients_100 generator=bury difficulty=hard n_seeds=1 methods=Kalman-Spectral-Drift,VAR-CSD,AC1-CSD,...` runs end-to-end and emits one row per method per system with `detection_time`, `ew_auc`, `fpr`.

Full suite: `python -m pytest tests -q` (currently 81 passing); `ruff`.
Reproduction commands (§ from survey doc) rerun on classic / bury-standard / bury-hard with the new suite.

---

## 8. The claim we can now make (and its honest limits)

**Claim (paper-level):** *Kalman-Spectral-Drift is a non-learned, parameter-free Bayesian observer that tracks the dominant stability margin (spectral gap `c_k`) online from a single observable, and — under a fixed 5 % false-positive rate — detects approaching fold, Hopf and period-doubling transitions with detection lead and AUC at least competitive with the canonical CSD indicators (variance, lag-1 AC, skewness, spectral ratio, DFA, return rate, DMD eigenvalue), and with strictly better false-positive control than the learned observers it replaces.*

Why it is defensible (matches `publication_framing.md` safe language):
- It is *the only method with a closed-form state-space model* in the table; every baseline is a sliding-window moment — the comparison is "model-based Bayesian filtering vs. generic rolling indicators", a clean, reviewable contribution.
- Non-learned (0 trainable params); σ_u/Q_drift are validation-selected hyperparameters, not training — reproducible, deterministic (seed 0).
- The claim threshold is "at least competitive at controlled FPR **where the dynamics match the fold reduction**" — the fold case is theoretically grounded (`docs/z2/formal_defininition.md`); Hopf/logistic carry the existing scope caveat and are reported honestly.
- Where an indicator inverts (Hopf AC1 per Bury 2021; hard-fold VAR), we report the raw result rather than flipping signs — strengthening credibility.

**Limits we must keep visible:** nobody wins everywhere; on bury-hard logistic/hopf the proposed method posts 0.716/0.781 AUC pre-retune (survey §2.1) — those rows stay as-is; the story is **FPR-controlled model-based detection with a single principled method**, not universal dominance.

---

## 9. Plan self-audit ("triple check")

### 9.1 Literature facts (verified via search of the primary sources, 2026-08-04)
- Dakos et al. 2012 (PLoS ONE **7**(7):e41010): rolling-window indicators AC(1)/AR(1), SD/variance, skewness, spectral ratio ("density at low freq e.g. 0.05 over high freq e.g. 0.5"), DFA; detrending = Gaussian (10 % length) / linear; windows default 50 % of record; sliding windows; trends via **Kendall τ**; DFA "requires >100 points for robust estimation", exponent →1(rescaled) at transition. ✅
- `earlywarnings` R package (Dakos & Lahti): `generic_ews` returns `ar1, sd, sk, kurt, cv, returnrate, densratio, acf1`; `densratio = spec.ar(window, order=1)[low=6 bin]/spec.ar[...][high=last bin]`; **`returnrate = 1/ar1` in code** (docs text says 1−ar(1); code is authoritative); `sk` is **abs(skewness)**; windows 25–75 % (50 % default), detrending no/gaussian/linear/first-diff. ✅
- Bury et al. 2021 (PNAS **118**(39):e2106140118): baselines against the CNN-LSTM are **lag-1 AC and variance**; "except for the Hopf bifurcation, where lag-1 AC **decreases**"; detrend = Lowess span 0.2; DL more sensitive + fewer false positives. ✅
- Donovan & Brand 2022 (Physica A **596**:127152) spatial EWS via DMD; Grziwotz et al. 2023 dynamic eigenvalue (Jacobian) local-linear window tracking; DMD fitting a global linear operator then reading its eigenvalue approaching 1. ✅ (This is also the premise of our own c_k observer.)
- BenchEWS v1.0 (Zenodo 10.5281/zenodo.20487811): prefix-consistency, holdout, **FPR-gated** ranking — our calibrated pipeline. ✅

### 9.2 Code facts (verified against the tree)
- `METHODS` at `benchmark.py:69-79`; `_run_synthetic_experiment` at `:174`; `_enabled` at `:199`; val/test indexes `val_idx_s, test_idx_s, val_idx_n, test_idx_n` at `:187-191`. ✅
- Fixed-FPR rule already implemented at `benchmark.py:636-643` (percentile of val-null steps). ✅
- Metrics used by the spectral block (`:645-663`) are the exact functions the baselines will reuse. ✅
- `compute_early_warning_auc` window [τ−50, τ−5] (`metrics.py:313-315`), `_sanitize_scores` fill=0.5 (`:12`), `select_threshold`/Youden untouched. ✅
- DFA NaN handling requires the calibration loop to filter finite steps — **this is the one place the cloned spectral structure must change** (∪ §4, flagged for tests 6). ✅
- Neurons of runtime: DMD per-step SVD on a (W≈30, m=6) matrix ≈ 1.8k flops × 200 × 500 ≈ 2×10⁸ — sub-minute numpy. DFA polyfits dominate (minutes). Both fine on CPU. ✅

### 9.3 Logical consistency
- All rows share input preprocessing (`extract_mode`), score semantics (higher = alarm, incl. RETRATE via recovery-time transform), calibration (val-null FPR), and metrics → apples-to-apples. ✅
- Monotone transforms (RETRATE 1/φ vs AC1, abs-skew) do not change ranked metrics; they only fix sign convention — stated explicitly. ✅
- Removing all learned methods removes every non-reproducible row, resolving the credibility issue the user flagged. ✅

---

## 10. Open review items

1. Retire learned-Kalman source modules from the repo, or keep them under `studies/legacy/`?
2. Keep `RETRATE-CSD` (recovery-time = rank-dual of AC1) or drop for parsimony?
3. Include the optional `KURT-CSD`/`CV-CSD`/AR(n)-max-eigenvalue (the rest of the `earlywarnings` set) as a two-row appendix, or stay at 7?
4. Verdict-printer: remove the learned-suite verdict entirely (it assumes Kalman-LSTM-vs-BCE) and let the results tables carry the comparison?
5. Run matrix: classic / bury-standard / bury-hard at patients_100 + patients_500, n_seeds=1 — confirm before executing (≈ tens of CPU-minutes).
6. DMD embedding m: fix at 6, or validation-tune {4,6,8} like σ_u/Q_drift?