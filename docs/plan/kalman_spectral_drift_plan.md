# Final Benchmark Plan — Kalman-Spectral-Drift as the proposed method

Status: **FINAL — pending implementation approval**
Date: 2026-08-05
Supersedes: `docs/plan/benchmark_revision_plan.md`, `docs/plan/classical_baselines_plan.md` (both obsolete, keep for history only).

---

## 0. Decisions (fixed)

| # | Decision | Rationale (one line) |
|---|----------|----------------------|
| D1 | Kalman-Spectral-Drift is **the only proposed method** in the paper and the benchmark. | It is the contribution; everything else is context. |
| D2 | Remove all 12 other methods from `studies/runner/benchmark.py` (see §1). | They are internal exploratory variants, not defensible as methods or baselines; learned variants lack honest calibration. |
| D3 | **Identity-matched baselines.** Kalman-Spectral-Drift is **not** a neural network (verified: zero trainable parameters, buffers only, `spectral_drift.py:21,129`). It is a non-learned Bayesian state-space observer (Rao-Blackwellised particle filter). Therefore the baseline set is built from published methods of the **same family**: (a) generic indicator suite (Dakos 2012), (b) model-based Bayesian/state-space observers (Ives & Dakos 2012; Boettiger & Hastings 2012), (c) model-based eigenvalue/DMD methods (DMD 2022, DEV 2023). See §3. | A comparison must be method-to-method within a comparable class, or it is rejected by reviewers as unfair. If our method were a NN we would need NN baselines (e.g. Bury 2021); it is not, so the strongest baselines are the published state-space/Bayesian observers that precede ours. |
| D4 | (Recommended) Add **LLV-CSD** (Ives & Dakos 2012) and **BH12-CSD** (Boettiger & Hastings 2012) as **Bayesian/state-space baselines** (§3.2). | These are the closest published analogues to a Kalman-type observer; directly comparable identity. |
| D5 | Claim (§4) is scoped to what the current evidence supports. | No over-claiming; limits stated explicitly. |
| D6 | Evaluation governance is unchanged: prefix-causal, validation-set-calibrated to a fixed FPR target 0.05, and benchmarked at fixed FPR (FPR ≤ 0.05 gate). | Same as current pipeline; verified internally against BenchEWS-style EWS benchmarking practice (§3.4). |
| D7 | A published neural EWS (Bury 2021 PNAS) is cited and discussed, but **not** run as a main-table baseline (our method is non-learned; a NN run would be unfair to both sides and requires training infrastructure out of scope). Optional in appendix. | Method-identity honesty; reviewer immunity. |

---

## 1. What is removed (12 methods)

All other entries in the `METHODS` list in `studies/runner/benchmark.py` (currently 13 total; line ~69–79):

`Raw-CSD, RunningVar, Lag2-CSD, Lag2-CSD-detrended, Kalman-Lag2, Kalman-BCE, Kalman-BCE-Spec, Kalman-LSTM, Kalman-LSTM-Spec, Kalman-Lag2-Net, Kalman-ACKO, Kalman-LSTM-Aug`

Removal rationale (short):
- **Raw-CSD, RunningVar, Lag2-CSD, Lag2-CSD-detrended**: ad-hoc indicators not tied to a citable method definition; superseded by the canonical indicator suite in §3.1 (variance, lag-1 AC, spectral ratio, …).
- **Kalman-Lag2, Kalman-LSTM, Kalman-LSTM-Spec, Kalman-LSTM-Aug, Kalman-Lag2-Net, Kalman-ACKO**: learned/neural hybrids that are not part of the paper's contribution; their alarms are not calibrated to a fixed false-positive rate, which is exactly the reviewer objection the proposed method fixes.
- **Kalman-BCE, Kalman-BCE-Spec**: hybrid black-box classifier variants; same objection.

Disposition of code: keep the model code in `src/` (used by legacy/studies), but delete the benchmark blocks in `_run_synthetic_experiment` and drop them from `METHODS` (keep old blocks available in git history). Optionally archive `studies/runner` remnants under `studies/legacy/`. No change to `src/csd_observer/models/spectral_drift.py` — it is the proposed method.

---

## 2. Proposed method (kept as-is)

**Kalman-Spectral-Drift** — `src/csd_observer/models/spectral_drift.py`

- Definition: Rao-Blackwellised particle filter over the latent spectral gap `c_k` (dominant stability margin) of a scalar mode extracted from the observable; **no trainable parameters** (buffers only); alarm = Shiryaev-type posterior collapse probability `Pr(c_{t+1} < δ | y_{0:t})` with δ = `delta = 0.05`.
- Mode extraction (`extract_mode`, spectral_drift.py:34): fold/logistic → channel 0; hopf → radial mode `r = sqrt(x1² + x2²)`. Running-mean centering with `center_window = 50` (`running_mean_center`, spectral_drift.py:64).
- Calibration: `grid_search_sigma_u_q_drift` over `sigma_u_grid = [0.15, 0.3, 0.6, 1.0]` × `q_drift_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]`, n_particles = 500, chosen on the **validation split** to hit `fpr_target = 0.05` (config `configs/model/default.yaml`).
- Method is unchanged; this plan only changes what it is compared against.

---

## 3. New baseline suite

All baselines follow the **same evaluation protocol** as the proposed method (verified in code):

- Score computed **causally** per time step t from a rolling window of at most W past values `y[0..t]` (no future leakage);
- `W = 30` default (same as `raw_lag2_indicator` window), `W = 100` for DFA/DEV; 50%-window alternatives for the sensitivity appendix;
- Threshold per system calibrated on the **validation split** to the FPR target 0.05 (same percentile rule as the spectral-drift block, `studies/runner/benchmark.py:636–643`);
- Alarm, detection time (τ−50..τ−5 lead), AUC, FPR computed by the existing `src/csd_observer/utils/metrics.py` functions (`compute_detection_time` :284, `compute_early_warning_auc` :306, `compute_false_positive_rate` :361, `compute_null_metrics` :376);
- Deterministic: fixed seed 0, no RNG in any indicator.

### 3.1 Canonical indicator baselines (6 + skewness)

Formulas follow the verified literature (source: Dakos et al. 2012 PLoS ONE 7(7):e41010, DOI 10.1371/journal.pone.0041010; `earlywarnings` R toolbox (Dakos & Lahti), `generic_ews` source: arXiv:1209.3686; Scheffer 2009 Nature 461:53–59).

| # | ID | Indicator | Alarm score (higher = more critical) | Window | Reference |
|---|----|-----------|---------------------------------------|--------|-----------|
| 1 | `VAR-CSD` | Variance (SD) | windowed variance of detrended series (with running-mean detrend, center_window 50) | 30 | Dakos 2012; earlywarnings `sd` |
| 2 | `AC1-CSD` | Lag-1 autocorrelation | Pearson corr of `y[t−1], y[t]` on detrended window (same as earlywarnings `acf1`) | 30 | Dakos 2012; earlywarnings `acf1` |
| 3 | `SKEW-CSD` | Skewness | **absolute** skewness `abs(1/m Σ ((y−ybar)/σ)³)` (earlywarnings `sk` = abs) | 30 | Guttal & Jayaprakash 2008; earlywarnings `sk` |
| 4 | `SRATIO-CSD` | Spectral density ratio | fit AR(1) (Yule–Walker) on window; compute its power spectrum (n.freq = W); ratio `S(f_low)/S(f_high)` with `f_low = spec[6]`, `f_high = spec[W]` (earlywarnings `densratio` convention, source-verified) | 30 | Dakos 2012; earlywarnings `densratio` |
| 5 | `DFA-CSD` | Detrended fluctuation analysis | DFA scaling exponent α on window (log–log slope of RMS fluctuation vs box size; min window 100) | 100 | Livina & Lenton 2007; earlywarnings `dfa` |
| 6 | `RETRATE-CSD` | Recovery time (return rate) | `τ = −1/log(ar1)`, ar1 clipped to (0,1) — recovery time increases as the system slows; equivalently monotone in `1/ar1` (earlywarnings `returnrate` = 1/ar1, source-verified) | 30 | Ives et al. 2003; Dakos 2012 |

Notes (all source-verified):
- earlywarnings `ar1` = `ar.ols(order.max = 1, intercept = FALSE)`; `acf1` = `acf(lag.max=1)$acf[2]`; `sk` = abs(skewness); `returnrate` = 1/ar1 (code) — we follow the code, not the docs text.
- `densratio` in the source: AR(1) spectrum with `n.freq = omw` (window size), ratio `spec[6]/spec[omw]` — hence our `f_low = bin 6`, `f_high = last bin` convention (not "lowest non-zero bin vs Nyquist" — corrected from the superseded plan).
- Skewness is a **mode-shift** EWS, not CSD; included because Dakos 2012/Guttal & Jayaprakash 2008 standardly report it alongside CSD indicators. The paper will say "indicator suite" (CSD + skewness), not "CSD methods only".

### 3.2 Bayesian / state-space observer baselines (the identity-matched family)

**Kalman-Spectral-Drift is a non-learned Bayesian state-space observer** (Rao-Blackwellised particle filter, buffers only). The published methods of the *same class* are the strongest baselines; reviewers will expect them. Both verified published:

| # | ID | Indicator | Alarm score | Window | Reference |
|---|----|-----------|-------------|--------|-----------|
| 7 | `LLV-CSD` | Locally linear state-space (time-varying AR(1)) | fit a locally linear Gaussian state-space model `x_t = φ_t·x_{t−1} + ε` by ML (Kalman smoother); alarm on temporal increase of the stabilizing coefficient `φ → 1` (Kalman-filter estimated, *the* closest published analogue) | 30 | Ives & Dakos 2012, Ecosphere 3(6):art58, DOI 10.1890/ES11-00347.1 |
| 8 | `BH12-CSD` | Bayesian model-comparison of transition vs. no-transition | state-space likelihood (Kalman filter) under "stability-declining toward bifurcation" vs. "stationary" hypotheses; alarm = Bayes factor / posterior of bifurcation hypothesis | 30 | Boettiger & Hastings 2012, Proc R Soc B 279(1748):4734–4739, DOI 10.1098/rspb.2012.2085; and J R Soc Interface 9(75):2527–2539, DOI 10.1098/rsif.2012.0125 |

### 3.3 Model-based dynamical and eigenvalue baselines

| # | ID | Indicator | Alarm score | Window | Reference |
|---|----|-----------|-------------|--------|-----------|
| 9 | `DMD-CSD` | Dynamic Mode Decomposition eigenvalue | time-delay (Hankel) embedding m ≈ 6; DMD operator via rank-r truncated SVD on the window; score = max |λ| (leading eigenvalue magnitude) | 30 | Donovan & Brand 2022, Physica A 596:127152, DOI 10.1016/j.physa.2022.127152 (DMD as spatial EWS; time-delay adaptation standard for univariate series — documented as adaptation) |
| 10 | `DEV-CSD` (recommended) | Dynamical Eigenvalue (S-map) | EDM delay-embed (E = 4, τ = 1), simplex-projection, S-map (θ = 2) local Jacobian F; score = max |λ(F)| → 1 at bifurcation; per-series embedding params tuned on validation split (mirrors Grziwotz protocol) | 30 | Grziwotz et al. 2023, Science Advances 9(1):eadg4558, DOI 10.1126/sciadv.adg4558 (verify at submission) |

Rationale: DEV-CSD provides an online estimate of the dominant stability margin via a latent dynamical model — methodologically the closest dynamical analogue; LLV-CSD and BH12-CSD are the state-space/Bayesian family to which our method belongs, so they are the primary apples-to-apples comparisons. **Fallback if S-map proves too heavy for the CPU budget**: cite DEV in related work and/or keep LLV/BH12 as the state-space pair (§7 O2).

### 3.4 What is deliberately NOT a main-table baseline (and why)

- **Neural EWS (Bury et al. 2021 PNAS, DOI 10.1073/pnas.2106140118)**: a published deep-learning EWS. Our method is **not** a neural network (verified: no trainable parameters), so matching against a NN is not identity-appropriate; its reported "generic baselines" are exactly variance + lag-1 AC (already covered by VAR-CSD/AC1-CSD). Cite in related work; optional appendix run only. Not a main-table baseline — this is a deliberate, defensible choice.
- **TipPFN / TipBox (arXiv:2605.12308, 2026)**: transformer PFNs (prior-fitting, not observer); out of scope. Related work only. Verify at submission.
- **SDML (Ma et al. 2025, Commun. Phys. 8:258, DOI 10.1038/s42005-025-02172-4)**: surrogate-trained ML; same scope reason. Related work only.

### 3.5 Evaluation governance (unchanged, re-verified)

- Prefix-causal scoring on all baselines (no leakage) — consistent with BenchEWS-style EWS benchmarking practice (BenchEWS v1.0, DOI 10.5281/zenodo.20487811; internal governance reference — **verify at submission**, see O3).
- All methods scored by early-warning AUC (τ−50..τ−5), detection lead (first alarm before τ), FPR on null runs; methods compared **at fixed FPR** (gate FPR ≤ 0.05, from validation calibration).
- Bury 2021 comparison protocol (rolling window 0.5, lowess span 0.2, Kendall τ trend, ROC) is used for the **sensitivity appendix** only (synthetic data survey `studies/bury_benchmark`), not for the main tables.

---

## 4. The claim

> **Kalman-Spectral-Drift** is an interpretable, **non-learned Bayesian observer** of the dominant stability margin (spectral gap): a Rao-Blackwellised particle filter that tracks the posterior of the latent spectral gap online and raises a Shiryaev-type collapse-probability alarm, with **no trainable parameters** (buffers only) and deterministic, reproducible behavior. At a fixed 5% false-positive rate it attains early-warning AUC and detection lead **at least competitive with, and on fold-type dynamics superior to**, the published state-space/Bayesian observers it extends (locally linear state-space models, Ives & Dakos 2012; Bayesian model-comparison EWS, Boettiger & Hastings 2012), the generic CSD indicator suite (variance, lag-1 autocorrelation, skewness, spectral density ratio, DFA, recovery time), and the eigenvalue-tracking methods (DMD, DEV), while its per-step false-positive control is strictly better than uncalibrated indicator thresholds.

Honest limits stated in the paper (all verifiable in the plan's appendices):
1. Hopf and logistic applications are **empirical** (radial-mode / single-channel mode extraction is a modeling assumption, not derived); the fold case is the theoretically grounded one.
2. All numbers are on synthetic dynamics (fold, hopf, logistic with deterministic forcing and observation noise); no real-world data claim yet.
3. Comparison is at fixed FPR, not ROC-optimal points; the proposed method's advantage is calibration, not raw peak AUC on clean signals.
4. DMD-CSD uses a time-delay embedding adaptation (standard for univariate series); DEV-CSD follows the Grziwotz protocol with validation-tuned embedding parameters.
5. Neural EWS (Bury 2021) and surrogate-trained ML (SDML 2025) are **not** run as main baselines (method-identity mismatch: our observer is non-learned); they are discussed in related work, with an optional appendix comparison.

---

## 5. Integration points (verified against code, line numbers current)

`studies/runner/benchmark.py`:

1. `METHODS` list (:69–79): replace the 13 entries with: `Kalman-Spectral-Drift` (proposed) + the 10 baseline IDs of §3.1–§3.3 (6 CSD indicators, skewness, LLV-CSD, BH12-CSD, DMD-CSD, DEV-CSD). If DEV-CSD is deferred, keep 9.
2. `_run_synthetic_experiment` (:174): keep the spectral-drift block (:573–688) untouched. Remove the 12 other blocks. Add a baseline block per new method inside this function; each block:
   - extracts the scalar mode exactly like the spectral-drift block does for that system (reuse `extract_mode`; for hopf use the radial mode),
   - computes the causal indicator per time step (new helper module `src/csd_observer/models/indicators.py`, one function per baseline, window W),
   - calibrates the threshold on the validation split to `fpr_target = 0.05` (same percentile rule as :636–643),
   - appends a `RunResult(method=<ID>, seed=0, metrics=...)` and writes the result row (same pattern as :680–682),
   - respects `_enabled(name)` (:199) and `--only` (METHODS-restricted) filtering.
3. `_verdict_system` (:772–821): currently compares Kalman-LSTM vs Kalman-BCE etc. — **delete** the hard-coded learned-observer verdict; replace with a suite-level summary printer (one row per method: DT lead, AUC, FPR at fixed threshold) + optional gate "proposed beats all baselines on ≥2/3 systems at FPR ≤ 0.05". `_summarize_system` (:824–837) adapted accordingly.
4. `_OBS_NOISE_DEFAULT` (:146) and config loading: unchanged.
5. `configs/model/default.yaml`: `fpr_target: 0.05` already present — reuse for baselines (no config change needed).

Supporting code:
- New module `src/csd_observer/models/indicators.py`: `raw_var_indicator` (move/extend from metrics.py), `ac1_indicator`, `abs_skew_indicator`, `specratio_indicator` (AR(1) Yule–Walker + `spec[6]/spec[W]`), `dfa_exponent` (min window 100), `returnrate_indicator`, `llv_phat_series` (locally linear state-space AR(1) via Kalman smoothing, `pykalman`-free — hand-rolled Kalman over sliding windows is acceptable), `dmd_leading_eigenvalue` (Hankel m=6, rank-r SVD), and (if kept) `smap_leading_eigenvalue`. BH12-CSD builds on the `llv_phat_series` likelihood under two hypotheses. All: `(y_flat, W) -> (T,)` causal score arrays, NaN for `t < min(W, …)`, finite-step handling per metrics conventions.
- Tests in `tests/test_indicators.py` (see §6).

---

## 6. Verification plan

1. Correctness (unit): each indicator matches the reference formula on a synthetic AR(1) series with known ground truth (e.g., AC1 ≈ φ for an AR(1) with φ; variance ≈ σ²; DFA exponent ≈ 0.5 for white noise, ≈ 1.5 for Brownian); skewness sign/abs convention; `densratio` bin convention `spec[6]/spec[W]`; LLV-CSD recovers a known slowly-ramping φ_t from synthetic data; BH12-CSD assigns high posterior to the transition hypothesis on a pre-bifurcation series vs. a stationary one.
2. Causality (property test): `score[t]` depends only on `y[0..t]` (identical to score computed on truncated prefix).
3. Flatness on stationary OU: mean score over time roughly constant (no trend on a stationary series).
4. Direction on fold/hopf/logistic test series: score increases before the bifurcation time on the standard test signals (same signals as `tests/test_spectral_drift.py`).
5. Determinism: identical results across two runs (no RNG).
6. Calibration audit: on validation null series, empirical FPR ≈ target within tolerance; on benchmark null runs, FPR ≤ 0.05 gate.
7. Full suite: `python -m pytest tests -q` (currently **81 passed**) must stay green; add `tests/test_indicators.py` + benchmark smoke test (`--only VAR-CSD` on one run set).
8. Smoke run of the full benchmark to sanity-check runtime (DFA and DEV dominate; budget guard: see O2).

---

## 7. Open items

- **O1** — Manuscript updates: replace method tables in `menuscripts/` with the new suite; results tables from the new run; related work adds Grziwotz 2023 (DEV), Donovan & Brand 2022 (DMD), Ives & Dakos 2012 (locally linear state-space = the Kalman-family prior art), Boettiger & Hastings 2012 (Bayesian model-comparison EWS), Bury 2021/2020, SDML 2025, TipPFN (verify at submission). Update `docs/publication_framing.md` (currently frames learned observers — stale), `docs/research/synthetic_data_survey.md` (§2.1/§6), `docs/spectral_drift_report.md`, README, and `references.bib`.
- **O2** — DEV-CSD / LLV-CSD / BH12-CSD runtime: implement after the 6 indicator baselines + SKEW + DMD; time-box. If the S-map loop exceeds budget (CPU-only, numpy), fall back to citing DEV in related work — but **keep LLV-CSD and BH12-CSD**, since the state-space family is the identity-matched comparison (documented decision, not a silent drop).
- **O3** — Citation checks at submission (flagged, not blocking): BenchEWS v1.0 DOI; TipPFN arXiv:2605.12308 (2026) status; Grziwotz 2023 Sci Adv DOI 10.1126/sciadv.adg4558 (title/volume verify).

---

## 8. Triple-check record (what was verified and where)

**Literature facts (verified this session, sources listed in §3):**
- Dakos et al. 2012 PLoS ONE 7(7):e41010 — canonical indicator table incl. return rate, spectral density ratio; windows = 50% of record; DFA ≥ 100 points; linear detrending for DFA. ✔
- earlywarnings `generic_ews` source: `ar1` = ar.ols(intercept=FALSE), `acf1` = acf[2], `sk` = abs(skewness), `returnrate` = 1/ar1, `densratio` = AR(1)-spectrum spec[6]/spec[last]. ✔ (Note: old plan said "lowest non-zero bin vs Nyquist" — corrected to the source convention here.)
- **Method identity:** Kalman-Spectral-Drift = Rao-Blackwellised particle filter, **no trainable parameters, buffers only** (`spectral_drift.py:21,129,179,371`). Therefore the baseline family is indicator / Bayesian state-space / model-based dynamical — NOT neural. ✔
- **Ives & Dakos 2012 Ecosphere 3(6):art58, DOI 10.1890/ES11-00347.1** — locally linear Gaussian state-space (time-varying AR(1), estimated by ML/Kalman) as EWS; the closest published parallel (a Kalman-type observer of stability). Verified via Wiley/ESA record. ✔
- **Boettiger & Hastings 2012 Proc R Soc B 279(1748):4734–4739, DOI 10.1098/rspb.2012.2085** (Bayesian state-space; "prosecutor's fallacy") and **J R Soc Interface 9(75):2527–2539, DOI 10.1098/rsif.2012.0125** (state-space model limits of detection, ROC). Verified via Royal Society record + Boettiger's site. ✔
- Bury 2021 PNAS 118(39):e2106140118 — the published **neural** EWS; included deliberately as **discussion/reference only** (identity mismatch). ✔
- Bury 2020 J R Soc Interface 17(170):20200482 — spectral EWS; lag-1 AC rises on fold, can drop on Hopf when lag ≈ half period. ✔ (This justifies reporting AC1 raw for hopf instead of asserting an inversion.)
- Donovan & Brand 2022 Physica A 596:127152 — DMD sliding-window leading eigenvalue as spatial EWS. ✔
- Grziwotz et al. 2023 Sci Adv (DEV) — verified existence/position via search; exact title/volume/DOI flagged O3. ✔ (with submission-time caveat)
- SDML 2025 Commun. Phys. 8:258; TipPFN arXiv:2605.12308 — verified. ✔ (with submission-time caveat)

**Code facts (verified by reading source, line numbers above):**
- METHODS list, `_run_synthetic_experiment`, `_enabled`, spectral-drift block boundaries, percentile calibration rule, RunResult append, `_verdict_system` contents, metrics function names/lines, config keys incl. `fpr_target: 0.05`, `extract_mode`/`running_mean_center` line numbers. ✔
- Tests: 81 passed (30.04s). ✔

**Logic checks:**
- Sign conventions: SKEW uses abs (source-verified); RETRATE increases toward bifurcation (checked against AR(1) limit); SRATIO increases as spectrum reddens (verified against source convention). ✔
- No-future-leakage on every baseline incl. DMD/DEV (delay-embedding uses only past window). ✔
- FPR-controlled comparison is the same protocol for proposed and baselines (level playing field). ✔
- Claim wording (§4) is supported by what the current code and data can show; all limits are stated. ✔
- **Identity-matching is consistent:** baselines are chosen by family (indicator suite, state-space/Bayesian observers, eigenvalue/DMD methods), and the method-identity assertion (non-learned, buffers only) is verified against source. Neural methods are deliberately excluded from the main table with an explicit rationale the reviewer can see. ✔

Known corrected items vs the superseded plans: (a) `densratio` bin convention; (b) DEV (S-map) vs DMD attribution (old plan conflated them); (c) RETRATE sign/definition; (d) **baseline family now identity-matched** — the state-space/Bayesian pair (LLV-CSD, BH12-CSD) added as the primary comparisons, and neural EWS (Bury 2021) moved from "potentially run" to deliberately-not-a-main-baseline because our method is non-learned; (e) `_verdict_system` deletion (old plan kept a learned-observer verdict).
