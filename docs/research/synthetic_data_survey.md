# Synthetic Early-Warning-Signal Data: Literature Survey & Generator Design

**Date:** August 4, 2026
**Scope:** survey of state-of-the-art synthetic EWS data, the `classic` / `bury` / `bury-hard` generator tiers implemented in `src/csd_observer/data/bifurcation.py`, the three generator bugs found and fixed (2026-08-04), and honest post-fix benchmark results.

---

## 1. Literature Survey

### 1.1 Bury et al. 2023 (Nat. Commun.) — the canonical generator
- Randomized normal-form models (fold, Hopf, pitchfork, transcritical, period-doubling/logistic) with:
  - per-trajectory randomized parameters (initial condition `mu_0` drawn conditioned on the bifurcation being observable, `|lambda| < 0.8` pre-transition),
  - polynomial perturbation terms up to degree ~10 with geometrically decaying random coefficients,
  - the transition = control crossing >10× the noise amplitude,
  - partial observability and coloured (AR(1)) noise.
- Used to train deep early-warning-signal classifiers (`dl_discrete_bifurcation`, Code Ocean capsule 2209652; github.com/ThomasMBury/dl_discrete_bifurcation).
- This is the design ancestor of `RandomizedBifurcationDataset` (the `bury` generator) and of the recent TipPFN model.

### 1.2 TipPFN / TipBox (arXiv:2605.12308, 2026)
- 14-dimensional randomized nonlinear SDE networks; partial observability of 1–16 features; ~1M generated tasks.
- Zero-shot transfer to AMOC, predator–prey, power-grid data; the current state of the art for task-agnostic tipping-point detection.
- Governance: tip-box papers adopt prefix-consistency (causal) scoring and fixed-FPR reporting — the same philosophy as our fixed-FPR calibration and causal filter.

### 1.3 SDML (Ma et al., Commun. Phys. 2025, s42005-025-02172-4)
- Surrogate-data ML: augment small real historical datasets by resampling pre-transition vs neutral windows.
- Not a generator for our benchmark tiering (method, not dataset) but relevant for the real-data evaluation tier later.

### 1.4 Coloured noise robustness (Mathematics 2025, 13(17), 2782)
- Systematic study of AR(1) red noise (persistence 0.1–0.5): CNN-LSTM EWS performance degrades smoothly with increasing noise colour; variance-based indices most affected.
- Motivated our per-trajectory `color` draw (standard tier 0–0.6, hard tier 0–0.8 for hopf/logistic).

### 1.5 BenchEWS v1.0 (Zenodo 10.5281/zenodo.20487811, 2026)
- Community benchmarking governance: prefix-consistency (causal) evaluation, holdout splits, FPR-gated leaderboards.
- Validates our design choices: causal running-window scoring, fixed-FPR threshold calibration, validation-based hyperparameter selection.

### 1.6 Real datasets identified (for the future real-data evaluation tier)
| Dataset | Source | Notes |
|---------|--------|-------|
| Chick-heart period-doubling | Bury 2023 Nat. Commun. repo | 23 bifurcating + 23 null aggregates; onset defined by return-map slope < −0.95; labeled + nulls |
| Thermoacoustic combustor | Pavithran & Sujith, Chaos 2021 (in Bury PNAS repo) | real blowout series with labeled onset; noise-heavy |
| Daphnia microcosm (DaphniaExt) | TipPFN paper suite | sparse W=16; extinction tipping |
| SWEC-iEEG seizures | TipPFN paper suite | regime transition (ictal onset), real clinical |
| MIMIC ICU vitals | Zenodo 10.5281/zenodo.19813052 | documented *negative* CSD results — qualitative case study only, not a benchmark pillar |

**Recommendation:** add chick-heart + thermoacoustic as a qualitative detection tier (no AUC) after the synthetic tiers are finalized; MIMIC as a documented null/failure case.

---

## 2. Generator Architecture (`src/csd_observer/data/bifurcation.py`)

`build_dataset(system, generator=..., difficulty=..., ...)`:
- `generator="classic"` → `FoldBifurcationDataset` / `HopfBifurcationDataset` / `LogisticMapDataset` (fixed-ramp normal forms, white noise).
- `generator="bury"` (default after 2026-08-04) → `RandomizedBifurcationDataset`:
  - **standard tier** (`difficulty="standard"`): per-trajectory bifurcation time τ∈[0.2,0.75]·T, polynomial perturbation (degree 1–10, N(0,1) coeffs, geometric decay at `perturbation_scale`), log-uniform drive noise `(0.5,2.0)` (fold: `(0.5,0.9)`), AR(1) colour `(0,0.6)` (fold: `(0,0.3)`), log-uniform obs noise `(0.5,2.0)`, logistic ramp end capped at μ=3.2.
  - **hard tier** (`difficulty="hard"`):
    - ramp kinds: `stat` (constant null), `co` (exponential approach to near-critical plateau, never crosses), `bif` (linear crossing), `rate` (sigmoid crossing concentrated at t0∈[0.55,0.75], steep 15–30);
    - nulls: 75% co-moving + 25% static; signals: 50% rate + 50% linear;
    - frozen per-system constants (2026-08-04):

      | system | param_end | cross_at | plateau | start | drive_mult | colour |
      |--------|-----------|----------|---------|-------|-----------|--------|
      | fold | −1.0 | 0.0 | 0.15 | (0.3, 3.0) | (0.3, 1.0) | (0, 0.3) |
      | hopf | 0.5 | 0.0 | −0.02 | (−2.0, −0.4) | (0.3, 3.0) | (0, 0.8) |
      | logistic | 3.15 | 3.0 | 2.95 | (2.0, 2.9) | (0.3, 3.0) | (0, 0.8) |

    - noise: drive log-U (0.3,3.0) except fold (above), obs noise log-U (0.2,2.5);
    - τ sentinel = T+1 for no-cross trajectories; fold signals carry a rejection rule (see §3).

### 2.1 Difficulty tier summary (spectral observer, patients_100, n_seeds=1)

| tier | fold | hopf | logistic |
|------|------|------|----------|
| classic | 0.985 (DT 132.3) | 0.971 (99.0) | 1.000 (56.5) |
| bury standard | 0.940 (89.0) | 0.891 (106.0) | 0.990 (61.2) |
| bury hard | 0.984 (126.6) | **0.781** (139.5) | **0.716** (131.1) |

Fixed-FPR operating point (recall @ FPR 0.1), spectral observer:

| tier | fold | hopf | logistic |
|------|------|------|----------|
| bury standard | 0.389 | 0.100 | 0.096 |
| bury hard | 0.337 | 0.092 | 0.110 |

Learned heads on the hard tier (full 13-method run, pre-param-end retune): logistic BCE 1.000→0.360, LSTM-Aug 0.367; hopf LSTM 0.728→0.547, LSTM-Aug 0.502; fold LSTM 0.675→0.573.

**Honest difficulty ceiling:** fold recall@FPR0.1 ≈ 0.34 on both bury tiers — the saddle-node crash is an inherently detectable regime change; co-moving nulls already neutralize variance-growth false positives. Further tightening (lower fold plateau, higher drive) breaks null purity (co-null crashes) or introduces early-crash mislabelling.

---

## 3. Generator Bugs Found & Fixed (2026-08-04)

1. **Explicit-Euler instability on the fold** (all three fold generators): `dt=1` Euler on `dx = (r − x²)dt + σdW` is unstable for r > 1 (linearized multiplier |1 − 2√r| > 1) → trajectories oscillated into the ±5 integration clip and pinned there for most of their length. **Fix:** 10× substep integration of the deterministic drift, noise injected once per output step. Post-fix clip fractions: classic fold nulls 99.6%→0%, bury nulls ~0%.
2. **Degenerate classic fold dataset:** `x[t] = x_i` wrote the t-th row (all N trajectories) instead of `x[i, t]` — all 500 trajectories were byte-copies of the last one; the classic fold baseline (0.810/0.789) was measured on this degenerate data. **Fix:** correct indexing (`x[i, t] = x_i`); 500 unique trajectories now.
3. **Noise-driven early crashes (mislabelling):** strong AR(1) drive flipped the fold across the unstable branch while r ≫ 0 (crash at t≈4 for a trajectory labelled τ=48). **Fix:** per-system drive/colour caps for fold + rejection sampling (fold trajectories are regenerated until the crash occurs at or after τ − 10; nulls are regenerated until no crash). Verified: 0 early crashes in 500 bury-standard + 500 hard fold signals.

**Note:** hopf and logistic were never affected (numerically stable integrator / discrete map); their numbers are unchanged across the fix and provide a consistency check of the pipeline.

---

## 4. Reproduction

```powershell
# classic baseline (honest post-fix)
python studies/runner/benchmark.py patients_100 generator=classic n_seeds=1
python studies/runner/benchmark.py patients_500 generator=classic n_seeds=1

# bury standard tier
python studies/runner/benchmark.py patients_100 generator=bury difficulty=standard n_seeds=1
python studies/runner/benchmark.py patients_500 generator=bury difficulty=standard n_seeds=1

# bury hard tier (frozen knobs per §2)
python studies/runner/benchmark.py patients_100 generator=bury difficulty=hard n_seeds=1
python studies/runner/benchmark.py patients_500 generator=bury difficulty=hard n_seeds=1

# spectral-only quick check (≈1–2 min each)
python studies/runner/benchmark.py patients_100 generator=bury difficulty=hard n_seeds=1 methods=Kalman-Spectral-Drift
```

Unit tests: `python -m pytest tests -q` (81 passing; 6 bury + 6 hard-generator tests).

---

## 5. Outputs (honest, post-fix)

- classic patients_100 (13 methods): `outputs/benchmark/patients_100/2026-08-04_22-21-10`
- bury standard patients_100 (13 methods): `outputs/benchmark/patients_100/2026-08-04_22-34-31`
- bury hard patients_100, frozen knobs, spectral-only: `outputs/benchmark/patients_100/2026-08-04_23-17-30`
- bury hard patients_100, 13 methods (pre-retune knobs hopf end 0.8 / rate 0.35): `outputs/benchmark/patients_100/2026-08-04_22-46-41`

**Outdated (corrupted generators — do not cite):** `2026-08-03_23-23-21` / `2026-08-03_23-28-45` (classic fold), `2026-08-04_01-08-09` / `2026-08-04_01-19-21` (bury standard fold), `2026-08-04_18-32-17` (hard fold).

---

## 6. Open Items

- patients_500 full runs under the frozen hard knobs (user runs manually).
- Full 13-method hard table under frozen knobs (current 13-method hard run predates hopf end 0.6→0.5 and rate 0.35→0.5).
- Real-data evaluation tier: chick-heart + thermoacoustic loaders (qualitative detection vs measured onset, no AUC).
- Persist SDML/coloured-noise/BenchEWS governance details into `docs/z2/literature_review.md`.
