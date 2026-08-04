# Implementation Plan: Classical (Real) EWS Baseline Methods

> **SUPERSEDED 2026-08-04** by `docs/plan/benchmark_revision_plan.md` — the revision keeps the same intent but drops the learned-Kalman suite and fixes the baseline list to the canonical published CSD indicators (Dakos 2012 / earlywarnings / DMD-eigenvalue).

**Status:** draft for review — no code added yet (2026-08-04)
**Scope:** add the established statistical early-warning-signal indicators from the literature (Scheffer 2009; Dakos et al. 2010, 2012; the `earlywarnings` R toolbox) as fixed-FPR-calibrated non-learned benchmark methods, evaluated on the existing synthetic tiers (classic / bury-standard / bury-hard).

---

## 1. Rationale

Every established EWS benchmark (Bury 2021 PNAS; Ma 2025 SDML; TipPFN 2026; BenchEWS 2026) compares against the classical CSD indicators — variance, lag-1 autocorrelation, skewness, spectral ratio, DFA, Hurst — because they are the real-world baseline methods a reviewer expects. We currently have variance (RunningVar, Raw-CSD) and lag-2 autocorrelation (Lag2-CSD) only; AC(1), skewness, spectral ratio, DFA, Hurst, covariance-eigenvalue and return rate are missing.

## 2. Method registry & integration points

All new methods follow the exact pattern of the existing non-learned methods (RunningVar / Lag2-CSD) in `studies/runner/benchmark.py`:

| File | Change |
|------|--------|
| `studies/runner/benchmark.py` `METHODS` (line ~69) | add 7 names (below) |
| `studies/runner/benchmark.py` `_run_synthetic_experiment` | one `if _enabled("X"):` block per method, mirroring the `Lag2-CSD` block (lines ~271–303): compute scores on test/val/null arrays → fixed-FPR threshold from val-null 95th pct → `evaluate_*` → `RunResult` |
| `csd_observer/utils/metrics.py` | new `raw_*_indicator` functions + `evaluate_*` wrappers (existing pattern: `raw_var_indicator`, `evaluate_raw_var`) |
| `_enabled()` / `methods=` CLI | automatic via METHODS registry |

New method names (deterministic, seed=0, 0 trainable params):

| Method | Indicator | Definition (sliding window, causal) |
|--------|-----------|-------------------------------------|
| `AC1-CSD` | lag-1 autocorrelation | AR(1) coefficient of linearly detrended window (`_linear_detrend` + `np.corrcoef(x[:-1], x[1:])`) |
| `Skewness-CSD` | skewness | third standardized moment of detrended window |
| `SpectralRatio-CSD` | spectral ratio | S_ratio = mean low-band / mean high-band power (Welch FFT, per-window) |
| `DFA-CSD` | DFA exponent | detrended fluctuation analysis scaling α over window (box sizes 4..window/4, log-log slope) |
| `Hurst-CSD` | Hurst exponent | rescaled-range R/S statistic, log-log slope |
| `CovEig-CSD` | dominant eigenvalue | largest eigenvalue of delay-embedding (m=3) covariance of detrended window |
| `ReturnRate-CSD` | return time | −1/ln(AC1) of detrended window (return time; rises pre-bifurcation) |

**All indicators rise near a fold/Hopf/period-doubling bifurcation** (CSD), so score = raw indicator value (same convention as our existing non-learned methods; EW-AUC over the score trajectory). Detrending reuses the existing `_linear_detrend` (Lag2-CSD-detrended) for AC1/Skewness/SpectralRatio/CovEig; DFA and Hurst are trend-robust by construction.

## 3. Evaluation & governance (unchanged pipeline)

- Causal masking identical to existing non-learned methods (score at t uses only y_{0:t}, no lookahead → prefix-consistency per BenchEWS).
- Fixed-FPR threshold = 95th percentile of **val-null** score steps (`fpr_target: 0.05`), reusing `select_threshold` / `compute_null_metrics`.
- Metrics per (system, method, seed): EW-AUC (`compute_early_warning_auc`), detection time (`_compute_per_traj_dts`), null FPR — exactly the columns the report tables use.
- Verdict printer (`_verdict_system`, lines ~778–810) currently hardcodes Raw/RunningVar/Lag2 rows — extend with the 7 new rows (cosmetic, same helper functions).

## 4. Implementation notes / risk

| Item | Decision / risk |
|------|-----------------|
| Window size | 30 (matches Lag2-CSD) for AC1/Skewness/CovEig; FFT/DFA/Hurst windows need ≥64/≥16 points → use window=64 for DFA/Hurst, 30 elsewhere. Keep constants in code (documented), config override later if needed. |
| S_ratio bands | Literature default: low band ≈ 0–0.15 cycles/sample, high band ≈ 0.15–0.5·Nyquist. Hopf oscillation at ω=0.1 rad/step ≈ 0.016 cycles/sample — inside low band; bands stay system-agnostic (validated on val split if needed). |
| DFA / Hurst cost | O(window²) per step × 200 steps × 500 trajectories — ~10⁷–10⁸ ops per method, acceptable CPU (≈1–2 min); if slow, compute indicator every k-th step and forward-fill (deterministic). |
| Hopf AC1 direction | For oscillatory systems lag-1 autocorr can drop near onset (SDML 2025: "increase or decrease depending on lag") — raw-value convention may invert; EW-AUC <0.5 then means "detects" (reported as-is, same as BCE 0.360 on logistic hard). No special-casing. |
| Logistic period-doubling | Variance/AC1 rise; expected mid-table, mirroring RunningVar/Lag2 behaviour. |

## 5. Tests

New tests in `tests/test_observer.py` (mirror `test_evaluate_raw_*` style):
- shape/dtype: indicator over (N,T,C) → (N,T); causal masking (score[t] unchanged when future is re-scaled);
- stationary OU null → flat scores (Kendall |τ| < threshold);
- fold/hopf/logistic bury-standard signals → rising trend pre-τ;
- determinism (same input → same output);
- integration: `methods=AC1-CSD,SpectralRatio-CSD` benchmark run produces metrics.json entries for all systems.

## 6. Documentation

- `docs/research/synthetic_data_survey.md`: new section "§2.2 Classical baseline methods" (table + literature anchors) and a full 20-method comparison table per tier once runs are done.
- `docs/spectral_drift_report.md`: note that baseline tables now include the classical indicators.

## 7. Phased scope (review decision)

- **Phase 1 (this plan):** 7 classical indicators above.
- **Phase 2 (optional):** DMD max-eigenvalue spectral EWS (Bury 2020 J. R. Soc. Interface) — delay-embedding + per-window SVD; heavier, ~2–4× DFA cost.
- **Phase 3 (optional):** port of the canonical Bury 2021 PNAS CNN head (their architecture + weights, or retrain on our generator) as the "real DL baseline"; and/or ResKMD (Koopman residual, 2026) as the SOTA non-trained method.
- TipPFN (2026) noted as SOTA reference; out of scope (weights/infra heavy).

## 8. Open questions for review

1. Window sizes: 30 (AC1/Skew/CovEig) + 64 (S_ratio/DFA/Hurst) — or uniform 30 with DFA/Hurst subsampled?
2. Include `ReturnRate-CSD` (≈ transformed AC1, adds little) or drop for parsimony?
3. Phase 2 DMD-eigenvalue: in scope now or later?
4. Verdict printer extension: cosmetic rows now or when runs are redone?
5. S_ratio bands fixed vs per-system tuned on validation?
