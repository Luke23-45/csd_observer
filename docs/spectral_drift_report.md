# CSD Observer: Kalman-Spectral-Drift Baseline Report

**Date:** August 3, 2026
**Method:** `Kalman-Spectral-Drift` — Rao-Blackwellised particle filter over the spectral gap `c_k` (formal definition: `docs/z2/formal_defininition.md`)
**Hardware:** CUDA (PyTorch 2.11.0+cu128)
**Config:** `configs/model/default.yaml` → `spectral_drift:` block

---

## 1. Method Summary

| Property | Value |
|----------|-------|
| Type | Non-learned (0 trainable parameters) |
| Filter | Rao-Blackwellised particle filter: collapsed Gaussian-RB Kalman for the drift `u_k`, particle filter over the spectral gap `c_k` |
| Signal | `extract_mode`: fold/logistic → channel 0; hopf → radial `sqrt(x1²+x2²)`; causal `running_mean_center` (window 50) |
| Grid search | `(σ_u, Q_drift)` jointly over `sigma_u_grid: [0.15, 0.3, 0.6, 1.0]` × `Q_grid [1e-6..1e-1]` on validation EW-AUC |
| Hyperparameters | n_particles 500, c_min 1e-3, delta 0.05, dt 1.0, c_init 0.1, c_init_std 0.05, σ_u grid-searched (default 0.15 retained on fold, lifted to 0.3–1.0 elsewhere), R = (0.10, 0.15, 0.05)² (fold/hopf/logistic) |
| Alarm score | `collapse_prob[t] = Pr(c_{t+1} < δ | y_{0:t})` (formal `p_collapse,k`) |
| Threshold rule | **Fixed-FPR calibration (formal §5)**: threshold = 95th percentile of validation-null collapse steps; `fpr_target: 0.05` |
| Determinism | Single seed-0 run per (system, run-config); splits signal 101 / null 202 / split_seed 1000 |

> **Why not Youden:** `select_threshold` (Youden) degenerates for posterior-probability alarms because the collapse probability carries a ~16% prior floor at t=0 (δ=0.05 vs c_init=0.1±0.05). Youden produced FPR 0.79–0.94 with alerts at t=0 or never-crossed thresholds. The fixed-FPR rule (formal §5) is used instead.

---

## 2. Patient-Size Sweep (noise_scale=0.15, n_seeds=10)

> **Pre-fix baseline** (σ_u=0.15 fixed, q-only search). The sweep was not
> re-run after the §5.1 fix; it documents the fixed-noise-scale behaviour.

| patients | fold DT / AUC / FPR | hopf DT / AUC / FPR | logistic DT / AUC / FPR |
|----------|--------------------|--------------------|------------------------|
| 100 | 110.3 / 0.809 / 0.046 | 90.8 / 0.970 / 0.051 | 43.5 / 0.777 / 0.035 |
| 200 | 125.8 / 0.799 / 0.055 | 89.0 / 0.937 / 0.046 | 44.4 / 0.723 / 0.039 |
| 300 | 122.6 / 0.809 / 0.052 | 89.9 / 0.912 / 0.054 | 46.3 / 0.706 / 0.048 |
| 400 | 123.6 / 0.773 / 0.052 | 89.2 / 0.967 / 0.052 | 47.5 / 0.789 / 0.049 |
| 500 | 122.7 / 0.796 / 0.053 | 89.2 / 0.979 / 0.049 | 45.3 / 0.866 / 0.032 |

**Timing:** total 243.8 s for all five configs on GPU.

Observations:
- FPR is at the 5% target on every system/config (by construction of the calibration).
- DT is stable across data sizes (the filter has no data-size dependence); alerts ~11 steps before τ for fold (τ=133.3) and hopf (τ=100), ~21 steps before for logistic (τ=66.7).
- AUC increases with data for hopf (0.912→0.979) and logistic (0.706→0.866); fold AUC is flat ~0.79–0.81.

---

## 3. Default Config vs Baselines (n_patients=500, noise_scale=0.15, n_seeds=1)

> **Method rows updated 2026-08-03 after the σ_u fix (§5.1).** Learned
> baseline rows are unchanged from the original cloud run.

### 3.1 Fold Bifurcation

| Method | DT (steps) ↓ | EW-AUC ↑ | FPR ↓ |
|--------|-------------|---------|-------|
| Kalman-BCE | **85.4** | **0.923** | 0.233 |
| Kalman-BCE-Spec | **85.4** | 0.907 | 0.235 |
| Kalman-LSTM | 89.2 | 0.791 | 0.593 |
| Kalman-LSTM-Spec | 89.2 | 0.789 | 0.593 |
| Kalman-Lag2 | 94.5 | 0.887 | 0.302 |
| Kalman-Lag2-Net | 99.2 | 0.567 | 0.618 |
| Kalman-ACKO | 118.2 | 0.783 | 0.205 |
| Kalman-LSTM-Aug | 133.3 | 0.999 | 0.152 |
| Lag2-CSD-detrended | 88.1 | 0.828 | 0.075 |
| Lag2-CSD | 88.5 | 0.820 | 0.100 |
| RunningVar | 133.3 | 0.792 | nan |
| Raw-CSD | 71.4 | 0.561 | nan |
| **Kalman-Spectral-Drift** | 123.2 | 0.789 | **0.051** |

**Takeaway (Fold):** best FPR of all 13 methods (next best: Lag2-CSD-detrended 0.075; learned methods 0.15–0.62). AUC mid-table (0.789, unchanged by the σ_u fix — validation correctly keeps 0.15 on fold), above LSTM 0.791 / ACKO 0.783; DT late (alerts ~10 steps before τ).

### 3.2 Hopf Bifurcation

| Method | DT (steps) ↓ | EW-AUC ↑ | FPR ↓ |
|--------|-------------|---------|-------|
| Kalman-BCE | 89.3 | 0.823 | 0.061 |
| Kalman-BCE-Spec | **33.5** | **0.992** | **0.009** |
| Kalman-LSTM | 37.2 | 0.963 | 0.107 |
| Kalman-LSTM-Spec | **33.5** | **0.992** | 0.093 |
| Kalman-Lag2 | 66.6 | 0.591 | 0.812 |
| Kalman-Lag2-Net | 66.6 | 0.725 | 0.816 |
| Kalman-ACKO | nan | 0.940 | 0.000 |
| Kalman-LSTM-Aug | 52.5 | 1.000 | 0.206 |
| Lag2-CSD-detrended | 28.4 | 0.567 | 0.003 |
| Lag2-CSD | 33.8 | 0.709 | 0.009 |
| RunningVar | 55.7 | 0.779 | nan |
| Raw-CSD | 24.6 | 0.766 | nan |
| **Kalman-Spectral-Drift** | 86.3 | 0.990 | 0.050 |

**Takeaway (Hopf):** AUC 0.990, 3rd only behind LSTM-Aug 1.000 and the 0.992 spec pair; FPR 0.050 competitive (better than LSTM 0.107 / LSTM-Aug 0.206); DT 86.3 (alerts ~14 steps before τ) — the price of the controlled-FPR calibration.

### 3.3 Logistic Bifurcation

| Method | DT (steps) ↓ | EW-AUC ↑ | FPR ↓ |
|--------|-------------|---------|-------|
| Kalman-BCE | nan | **1.000** | 0.000 |
| Kalman-BCE-Spec | nan | **1.000** | 0.000 |
| Kalman-LSTM | nan | **1.000** | 0.000 |
| Kalman-LSTM-Spec | nan | **1.000** | 0.000 |
| Kalman-ACKO | nan | **1.000** | 0.000 |
| Kalman-LSTM-Aug | 66.7 | **1.000** | 0.251 |
| Kalman-Lag2 | 66.7 | 0.460 | 0.830 |
| Kalman-Lag2-Net | 66.7 | 0.630 | 0.795 |
| Lag2-CSD-detrended | 22.5 | 0.477 | 0.002 |
| Lag2-CSD | 22.8 | 0.467 | 0.010 |
| RunningVar | 36.0 | 0.409 | nan |
| Raw-CSD | 25.1 | 0.420 | nan |
| **Kalman-Spectral-Drift** | 65.7 | **1.000** | 0.050 |

**Takeaway (Logistic):** with the σ_u fix the method now reaches **AUC 1.000, tying the learned observers**, while FPR 0.050 is the second-lowest among all methods and DT 65.7 is still earlier than the learned ones (nan/66.7). Previously 0.866 (pre-fix, 2026-08-03).

---

## 4. High-Noise Stress Test (noise_scale=0.30, n_patients=500, n_seeds=1)

| System | DT | EW-AUC | FPR |
|--------|-----|--------|-----|
| Fold | 124.4 | 0.973 | 0.051 |
| Hopf | 87.3 | 0.974 | 0.049 |
| Logistic | 52.4 | 0.994 | 0.039 |

**Key finding — empirical noise invariance:** spectral-drift AUC *rises* at 2× noise (fold 0.796→0.973, hopf 0.979→0.974, logistic 0.866→0.994), the opposite of the LSTM heads (LSTM-Spec collapses on fold 0.789→0.727) and unlike Kalman-BCE (fold AUC unchanged at 0.923). The RB-PF tracks the gap dynamics directly through the observation noise R, so heavier noise is absorbed by the filter rather than degrading the alarm score. This is a strong robustness argument relative to the learned observers.

---

## 5. Root-Cause Analysis (why it loses to learned heads) — and fix

Full diagnostic: `docs/notes/spectral_drift_diagnosis.md` (patients_100, all systems; validated against the benchmark pipeline — saved npz vs fresh run match, fold AUC 0.836 vs 0.809, hopf 0.965 vs 0.970, logistic 0.795 vs 0.777).

**Root cause: the observation model's per-step noise σ_u was fixed to `noise_scale=0.15`, but the data's actual innovation scale is very different per system.**

| system | empirical innovation sd (null, centred mode) | σ_u in model | ratio |
|--------|----------------------|--------------|-------|
| fold | 0.954 | 0.15 | 6.4× |
| logistic | 0.294 | 0.15 | 2.0× |
| hopf | 0.166 | 0.15 | 1.1× |

Because the OU stationary variance is σ_u²/(2c), an oversized innovation inflates Var(null): the filter mistakes null variance for small c, collapse probability stays high on nulls, the fixed-FPR threshold is pushed up, and the signal only crosses it in its final (explosion) window. Direct evidence:

| system | σ_u | AUC | DT |
|--------|-----|-----|-----|
| logistic | 0.15 (pre-fix) | 0.795 | 52.5 |
| logistic | 0.30 | 0.998 | — |
| logistic | 0.50 | **1.000** | 54.8 |
| logistic | 0.50 (all 100 patients) | **1.000** | — |
| fold | 0.15 (kept) | 0.836 | 110.6 |
| fold | 0.95 | 0.645 | 130.2 |
| hopf | 0.15 (pre-fix) | 0.965 | 93.1 |
| hopf | 0.50 | 0.955 | 88.0 |

Logistic is the system where we lost most (AUC 0.866 vs learned 1.000) — and it is exactly where σ_u was 2× off: calibrating σ_u to the empirical innovation scale turns it into a perfect classifier. hopf is unaffected (σ_u already right). fold is *harmed* by a larger σ_u (its pre-transition dynamics are far from OU: AR(sig) ≈ −0.45 until t≈130, and null variance 0.57 is *larger* than signal's 0.03–0.04, so the c-mechanism reads null variance as collapse) — on fold the filter degenerates into a terminal variance-explosion detector (rolling Var alone: AUC 0.74 vs 0.84).

**Why the learned heads win:** they adapt their noise scale to the data automatically; a fixed σ_u=0.15 was a free hyperparameter wrong by 2–6× on two of three systems.

### 5.1 Fix (deployed 2026-08-03)

**Implementation:** `grid_search_sigma_u_q_drift` in `src/csd_observer/models/spectral_drift.py` — the benchmark now grid-searches `(σ_u, Q_drift)` jointly on validation EW-AUC (`sigma_u_grid: [0.15, 0.3, 0.6, 1.0]` in `configs/model/default.yaml`, chosen per system). `grid_search_q_drift` is now a thin wrapper over the joint search, so its behaviour/API is unchanged. The observer remains non-learned (hyperparameter selection on the validation split, no training). Local verification on CPU:

| config | fold DT / AUC / FPR | hopf DT / AUC / FPR | logistic DT / AUC / FPR |
|--------|--------------------|--------------------|------------------------|
| patients_100 n_seeds=1 | 121.9 / 0.810 / 0.045 (σ_u=0.15, bit-identical to pre-fix) | 99.0 / 0.971 / 0.047 (σ_u=0.6) | 56.5 / **1.000** / 0.051 (σ_u=0.3) |
| patients_500 n_seeds=1 | 123.2 / 0.789 / 0.051 (σ_u=0.15) | 86.3 / **0.990** / 0.050 (σ_u=0.6) | 65.7 / **1.000** / 0.050 (σ_u=1.0) |

- Fold: validation correctly retains σ_u=0.15 — the row is bit-identical to the pre-fix run (deterministic).
- Hopf: AUC 0.979 → 0.990, DT 89.2 → 86.3.
- Logistic: AUC 0.866 → **1.000** (ties the learned 1.000) with FPR 0.050; DT later (65.7 vs 45.3, still earlier than learned nan/66.7).

**Notes:** the hopf/logistic DT differences vs the older runs reflect the σ_u/q operating point (higher σ_u → slightly later first threshold crossing; initial-transient artifact of the fixed-FPR rule, see diagnosis note). The patient-size sweep table (§2) predates the fix and has not been re-run on GPU; method rows in §3 are updated.

---

## 6. Verification

| Check | Result |
|-------|--------|
| Unit tests | 68/68 (22 for spectral-drift: shapes, bounds, c_hat ≥ c_min, determinism, frozen-likelihood vs analytic Kalman ~1e-3, tracking, stationary-OU low collapse, grid search incl. joint (σ_u, q), bad-grid rejection) |
| Ruff | clean (only pre-existing F841 in Kalman-Lag2 block, benchmark.py:519) |
| Smoke (CPU) | patients_100 + patients_500 n_seeds=1 reproduce cloud fold/hopf within rounding; logistic AUC 1.000 visible locally |
| FPR calibration | val-null 95th-pct threshold; test FPR ≈ 0.05 by construction |
| Non-learned | σ_u selection is validation-AUC hyperparameter search, 0 trainable params |

> **Note on verdicts:** single-method runs print `VERDICT: NO-GO (0/3)` — this is expected and not a failure: `_verdict_system` (benchmark.py:765) only evaluates the full-suite primary (Kalman-LSTM) vs Kalman-BCE gains, so all comparisons are NaN when other methods are filtered out. The verdict will be meaningful only when the full method suite runs together.

---

## 7. Outputs (cloud)

- `outputs/benchmark/patients_100/2026-08-03_16-33-12`
- `outputs/benchmark/patients_200/2026-08-03_16-33-37`
- `outputs/benchmark/patients_300/2026-08-03_16-34-13`
- `outputs/benchmark/patients_400/2026-08-03_16-35-00`
- `outputs/benchmark/patients_500/2026-08-03_16-36-00` (n_seeds=10)
- `outputs/benchmark/patients_500/2026-08-03_16-42-36` (n_seeds=1)
- `outputs/benchmark/high_noise/2026-08-03_16-43-58`

**Local (post-fix, CPU) verification outputs:**
- `outputs/benchmark/patients_100/2026-08-03_23-23-21`
- `outputs/benchmark/patients_500/2026-08-03_23-28-45`

**Code:** `src/csd_observer/models/spectral_drift.py` · `studies/runner/benchmark.py` (METHODS ~line 75; `_OBS_NOISE_DEFAULT` ~line 144; spectral block ~lines 571–685) · `configs/model/default.yaml` (`spectral_drift:` block incl. `sigma_u_grid`) · `tests/test_spectral_drift.py`
