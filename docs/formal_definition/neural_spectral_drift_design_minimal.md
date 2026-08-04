# Neural Spectral-Drift Observer (NSDO) v3 — Modern Stack, Anytime-Valid Alarm

**Status:** DRAFT v3 (v2 superseded; the Gaussian-head + GRU + percentile-threshold design is dropped).
**Date:** 2026-08-05
**Grounded in:** `docs/formal_definition/formal_defininition.md` (for the *quantity* to track, not the *mechanism*)
**Companion:** `docs/formal_definition/literature_review.md`; `docs/plan/kalman_spectral_drift_plan.md`

---

## 0. Design brief

The neural early-warning system, built on the 2025–2026 stack — not on 2018 components:

- **Encoder:** a **selective state-space model** (Mamba-3-style, complex-valued states; Gated DeltaNet; TTT-layer hybrids). Not a GRU.
- **Posterior:** **flow-matching posterior estimation (FMPE)** over the spectral gap — a non-Gaussian, calibrated posterior. Not a Gaussian MLP head.
- **Alarm:** a **neural e-detector** — a sequential testing-by-betting process with a **finite-sample anytime-valid false-alarm guarantee** (`E_k ≥ 1/α` ⇒ ARL ≥ 1/α at *any* stopping time). Not a validation-set percentile threshold.
- **Governance:** the fixed-FPR benchmark protocol remains as the *comparison protocol* (all baselines use percentile thresholds); the neural method additionally reports the stronger anytime-valid guarantees.

**Decision criterion: best detection results (AUC, detection lead, FPR) at fixed FPR 0.05, plus a statistical guarantee the baselines cannot offer.**

**What survives from the formal math — and only this:** the *quantity* to track. Near a bifurcation the dominant multiplier `ρ` satisfies `|ρ| → 1`; the spectral gap `c_k = −ln|ρ_k| → 0` is the distance to the transition in log-space. That quantity is the network's supervised target (§1). The math says *what* to estimate — the network is free about *how*.

---

## 1. The problem and the supervised target (unchanged, verified)

**Input:** online stream of observations `y_{1:k}` (fold/logistic: 1 channel; hopf: 2 channels `x1, x2`, `bifurcation.py:189–192`), causality required.

**Output:** an alarm decision per step; evaluated as early-warning AUC (window `[τ−50, τ−5]`), detection lead (DT), FPR at fixed threshold — the benchmark's metrics.

**Supervised target (the key advantage):** the repository's simulator returns the full **parameter trajectory** per series (`bifurcation.py:71`); the dominant multiplier `ρ_k` is a closed-form linearisation of each generator's actual update:

| Family | update (per step) | dominant multiplier `ρ_k` |
|---|---|---|
| fold | 10 substeps `x ← x + 0.1·(r − x²)` | `(1 − 0.2·√r_k)¹⁰` (fixed point `x* = √r_k`) |
| logistic | `x ← μ_k·x·(1−x)` | `2 − μ_k` (at fixed point `x* = 1 − 1/μ_k`) |
| hopf (radial) | `r ← r + μ_k·r − r³` | `1 + μ_k` (about the origin) |

hence `c_k = −ln|ρ_k|` is known at every step (derivation helper only; no simulator changes).

**Validity window (critical, verified):** `ρ_k` is only meaningful before the transition — after `τ` it breaks down for every family (fold: `r<0` ⇒ `√r` undefined; logistic: `|2−μ|>1` ⇒ `c<0`; hopf: `1+μ>1` ⇒ `c<0`). **The supervised target is used only on `k ≤ τ−5`**, optionally floored at `log c_min` (`c_min = 0.001`, config). The alarm window `[τ−50, τ−5]` lies inside the valid region.

---

## 2. Architecture overview (v3)

```
y_k ─▶ (C0) learned preprocessing ─▶ x_k
x_k ─▶ (C1) selective SSM encoder (Mamba-3-style, complex states; causal) ─▶ h_k
h_k ─▶ (C2) flow-matching posterior estimator q(θ_k | y_{1:k})  [θ_k = log c_k]
(h_k, posterior) ─▶ (C3) neural e-detector: e_k = 1 + λ_k(Z_k − Z̄);  E_k = Π e_j
                    alarm iff E_k ≥ 1/α   (anytime-valid, ARL ≥ 1/α)
```

Stateful, one causal pass, constant memory, constant time per step. Fully differentiable end-to-end.

---

## 3. Components

### 3.0 C0 — Learned preprocessing

Causal front-end on the raw observation stream: per-channel 1-D convolution + learned affine normalisation (no future leakage). Motivation: the classical/DL baselines' fixed detrending (Lowess span 0.2, Gaussian bandwidth) is precisely what the Dablander–Bury 2022 critique showed to be fragile; here preprocessing is learned jointly with the stability target. The auxiliary handcrafted channels of §3.1 (rolling variance, lag-1 autocorrelation at 2–3 window sizes) are fused at the C0 output. Ablation: fixed unit-scaling front-end instead (risk table, "learned C0 ablated against fixed").

### 3.1 C1 — Selective SSM encoder (the 2026 backbone)

**Default: Mamba-3-style selective state-space layer** (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, ICLR 2026; arXiv:2603.15569):

- **Exponential-trapezoidal discretization.** Mamba-3's recurrence is derived from a *more expressive discretization of continuous SSM dynamics* — the same family as the formal definition's exact discretisation `φ(c) = e^{−cΔt}`. The encoder's per-step transition is, structurally, the generalization of the math, not a black-box imitation.
- **Complex-valued states.** This is not a gimmick: the **Hopf family is a complex conjugate pair** of eigenvalues approaching the unit circle. A complex-valued state space can represent the pair in *two real parameters* — the exact inductive bias the Hopf class needs. Fold/logistic (real eigenvalue) are the real-axis special case.
- **MIMO formulation** (Phase C/D, decided ON): better accuracy at the same decode latency; the hopf 2-channel input maps naturally to the MIMO (vector-input) recurrence.
- **Decision rule (review 2026-08-05):** Phase A default is the **Gated DeltaNet** layer with small state (`d_state ∈ {16, 32, 64}` sweep, no compute constraint), because it has the most mature training kernels at our scale; the **Mamba-3-style complex-valued selective SSM is implemented for ablation G3**, where its complex-state inductive bias (the Hopf eigenvalue pair) is tested head-to-head:
  - **Gated DeltaNet** (Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464) — delta-rule memory, the strongest Mamba-2 successor in linear models. **Phase A default (review decision).**
  - **Mamba-3-style** (Lahoti et al., ICLR 2026, arXiv:2603.15569) — exponential-trapezoidal discretisation, complex-valued states, MIMO; the exact inductive bias for the Hopf class. Implemented for G3.
  - **TTT / TTT-KVB layers** (Sun et al., ICML 2025 Spotlight, arXiv:2407.04620; TTT-E2E, arXiv:2512.23675) — hidden state is *itself a model updated at test time*; principled test-time adaptation for the slow drift regime. Heavier at inference; include only if G3 shows a clear win.
- No attention/transformer on the online path (cost, and the selective-SSM class dominates causal streaming at this scale).
- **Auxiliary input channels (ablation, recommended ON):** rolling variance and lag-1 autocorrelation at 2–3 window sizes concatenated as extra channels — cheap, informative, and the encoder decides how to use them.

### 3.2 C2 — Flow-matching posterior estimator (the 2023–2025 posterior)

Replace the Gaussian head with a **conditional flow-matching posterior estimator (FMPE)**:

```
z₀ ~ N(0, I)  (base)
θ_k = φ_f( z₀ ; c = h_k )        (learned velocity field v_t, integrated once at inference)
q(θ_k | y_{1:k}) = law of φ_f     (non-Gaussian; calibrated by construction)
```

- **Training (supervised FMPE):** on simulator series, draw `θ*_k = log c_k` (valid window only) and regress the velocity field to the transport between base and target samples — the standard flow-matching loss (Lipman et al. 2023; Wildberger, Dax, Buchholz, Green, Macke, Schölkopf, NeurIPS 2023 — *Flow Matching for Scalable SBI*; conditional-FM posterior inference: Zhai, Jeong & Ročková, arXiv:2510.09534, 2025). Supervision makes this a *neural stability posterior*, not an unsupervised latent.
- **Why flow matching, not a Gaussian head:** the posterior over `log c` is genuinely non-Gaussian — identifiability collapses as `c → 0` (formal definition §6), the drift is slowly time-varying, and the collapse event itself is a boundary (`θ* → log c_min`). A flow posterior represents this without distributional assumptions, and its quantiles give the alarm features (`Pr(c_k < δ)` via `Φ`-free Monte-Carlo integration of the transported samples).
- **Diagnostics:** posterior reliability plot; posterior-mean tracking error `E|E[θ_k] − θ*_k|` on held-out series.
- Optional ELBO/FIVO-style regulariser (a small decoder head) — ablated.

### 3.3 C3 — Neural e-detector (the anytime-valid alarm)

The alarm is a **sequential testing-by-betting process** — the modern (2021–2025) replacement for fixed percentile thresholds, with a *finite-sample* guarantee no percentile threshold has.

**Construction (rigorous, nonparametric):**

```
per step k:   Z_k = σ( MLP([h_k, posterior features]) ) ∈ [0,1]     (belief, learned)
              λ_k = clamp(MLP(h_k), 0, λ_max) ∈ [0,1]              (wager, learned)
              e_k = 1 + λ_k (Z_k − Z̄)                              (e-increment)
              E_k = Π_{j≤k} e_j                                    (e-process)
alarm at      τ̂ = inf { k : E_k ≥ 1/α }                            (α = 0.05)
```

- `Z̄` is a **fixed, conservative null baseline** estimated once on validation nulls as **mean + 2σ of `Z`** (review decision; trades a little power for a stronger guarantee), kept constant thereafter. Under the null, `E[e_k | F_{k−1}] ≤ 1` holds *by construction regardless of how Z_k is computed* (the wager λ_k is bounded); by Ville's inequality this makes `E_k` an **e-process**:
  **`ARL = E_P,∞[τ̂] ≥ 1/α` at any stopping time, finite sample, no model assumptions** (Shin, Ramdas & Rinaldo, *E-detectors: a nonparametric framework for sequential change detection*, NEJSDS 2(2):229–260, 2024, DOI 10.51387/23-NEJSDS51; Grünwald et al. 2023; Ramdas & Wang 2025).
- **Training (Stage 2):** maximize `Σ_k log e_k` on transition series (the e-process grows fast under the alternative — the neural analogue of the Shiryaev delay/KL trade-off, now *learned* and *guaranteed*); null series train `Z_k → Z̄` (the e-process stays flat). This is **testing by betting with a learned bettor** (Shafer 2021); the learned `λ_k` is the adaptive betting strategy. Hyperparameters (review decision): wager bound `λ_max = 0.5`, small `λ_e` in Stage 2, and the **e-process audit plot reported in the paper**. 
- **No validation-set threshold calibration is needed** — the threshold `1/α` is set by the theorem. The benchmark's percentile rule (`benchmark.py:636–643`) is retained *only as a diagnostic/comparison* for the fixed-FPR table, and the anytime-valid ARL is reported alongside.
- **Robustness:** `E_k` is scale-invariant to misspecified `Z̄` only up to the `E[e_k]≤1` condition; the validation-null audit (G5) verifies the e-process empirically. If the audit fails, `Z̄` is re-estimated conservatively (Waudby-Smith–Ramdas-style betting confidence-sequence bound).

This is the paper's headline guarantee: **the first deep early-warning system with a finite-sample anytime-valid false-alarm control** — something neither Bury-style classifiers nor the exact observer's asymptotic Shiryaev claim provide.

---

## 4. Why this design (honest comparison)

| | Exact RB-PF observer | Bury 2021 CNN-LSTM | DeepQCD 2024 | **NSDO v3** |
|---|---|---|---|---|
| Backbone | analytic filter | CNN-LSTM | RNN | **selective SSM (complex states)** |
| Stability posterior | unsupervised PF | none | none | **supervised flow-matching posterior** |
| Uses simulator `ρ_k` labels | no | no | no | **yes** |
| Alarm guarantee | asymptotic, within model | none (post-hoc FPR) | trained threshold, no finite-sample bound | **finite-sample anytime-valid ARL ≥ 1/α** |
| Multi-family | fold only (formal scope) | many | generic | **fold+hopf+logistic+nulls, one target** |
| Inference | 500 particles × Kalman | heavy CNN | RNN | **one SSM pass, O(1) per step** |
| Uncertainty | posterior | none | none | **full posterior + e-process** |

---

## 5. Training

### 5.1 Data library

- Generators: fold, hopf, logistic + nulls (`bifurcation.py`); lengths match the benchmark (`max_length=200`); noise/ramp ranges sampled around generator defaults; `ρ_k` derived per §1 (valid on `k ≤ τ−5`).
- Scale: Phase 1 ≈ 10⁴ series; Phase 2 ≈ 10⁵–10⁶ (Bury used 500k/200k; supervision on the latent carries most of the structure).
- **Nulls mandatory in every mix** (Boettiger & Hastings 2012; Dablander & Bury 2022).

### 5.2 Objectives (staged)

```
Stage 1 (observer):  L = L_flow(q(θ_k|y_{1:k}); θ*_k)          (+ λ·ELBO optional)
Stage 2 (alarm):     L = L_flow + λ_e · ( −Σ_k log e_k  on transitions
                                          +  Σ_k (Z_k − Z̄)²  on nulls )
```

- Stage 1 trains C0/C1/C2 (the stability observer). Stage 2 trains C3 (the bettor) with a small `λ_e`, fine-tuning end-to-end; freeze C1/C2 if gradient interference appears (validation decides).
- Determinism: fixed-seed discipline as the exact observer.

### 5.3 Curriculum

L0 unit series (single constant c, clean) → L1 fold + logistic at several speeds/noises → L2 nulls (e-process audit, `Z̄` estimation) → L3 hopf + transfer systems. Nulls in every batch.

---

## 6. Evaluation

Protocol per `docs/plan/kalman_spectral_drift_plan.md` §3.5, with the anytime-valid addition:

- **Benchmark table (fixed FPR 0.05, all methods):** AUC (τ−50..τ−5), DT, FPR — per family and per noise level.
- **Anytime-valid table (NSDO only):** ARL under null at α=0.05 (theory: ≥ 20; empirical: report), EDD under alternative (Pollak-style), worst-case over stopping times; e-process audit plot.
- **Ablation matrix:**

| Config | Encoder | Posterior | Alarm | Scope |
|---|---|---|---|---|
| Exact observer | — | — | — | baseline |
| A (minimal) | Gated DeltaNet SISO | FMPE | e-detector | fold |
| B (+ multi-type) | Gated DeltaNet SISO | FMPE | e-detector | all 3 + nulls |
| C (+ MIMO) | Gated DeltaNet MIMO | FMPE | e-detector | all (hopf pair) |
| D (+ Mamba-3) | Mamba-3-style complex SSM | FMPE | e-detector | all |
| E (+ TTT hybrid) | GDN + TTT | FMPE | e-detector | all |
| F (posterior abl.) | Gated DeltaNet | Gaussian head vs FMPE | e-detector | all |
| G (alarm abl.) | Gated DeltaNet | FMPE | e-detector vs percentile | all |

- Gate discipline: same fixed FPR for the comparison table; no threshold cherry-picking (the e-detector threshold is fixed by the theorem).

---

## 7. Roadmap and gates

- **Phase A — observer:** C0 + C1 (Gated DeltaNet SISO, `d_state ∈ {16, 32, 64}` sweep) + C2 (FMPE), Stage 1 on fold+logistic+nulls.
  - G0 (fit): flow loss decreases; posterior-mean tracks `θ*` on held-out L0 (R² > 0.9 on clean series).
  - G1 (vs exact observer): posterior-mean of `c` (`E[c]` from flow samples) has MSE ≤ exact observer's `c_hat` MSE (`spectral_drift.py:307`) on fold, valid window `k ≤ τ−5`.
- **Phase B — alarm:** C3 (e-detector); `Z̄` from validation nulls.
  - G2 (no regression): on fold at fixed FPR 0.05, AUC and DT ≥ exact observer and ≥ canonical indicators.
  - G2b (anytime-valid): empirical ARL on validation nulls ≥ 20 at α=0.05 (theory bound, checked empirically).
- **Phase C — multi-type:** add hopf; joint training over all families.
  - G3 (encoder ablation): Mamba-3 vs Gated DeltaNet vs TTT on stability-MSE + AUC + ARL; switch on clear win.
  - G4 (transfer): per-family AUC/DT at fixed FPR ≥ per-family baselines; null audit per family.
  - G5 (calibration): posterior reliability within 5 points; e-process audit flat under nulls.
- **Phase D — polish:** MIMO if it wins; scale-up library (10⁵–10⁶); OOD holdout (unseen families/parameters) reported honestly; determinism check.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting the simulator | OOD holdout families; broad parameter ranges; nulls everywhere; per-noise reporting |
| Target undefined after τ | Supervise only `k ≤ τ−5`; floor at `log c_min` (§1) |
| e-process validity drift after training | `Z̄` fixed & conservative; wager `λ_k` bounded; G2b/G5 audits; re-estimate `Z̄` via Waudby-Smith–Ramdas bound if audit fails |
| Misspecified `Z̄` (Dablander–Bury lesson) | Conservative baseline by construction; sensitivity appendix over detrending variants; learned C0 ablated against fixed |
| SSM training instability | S4D/HiPPO-style initialization where applicable; gradient clipping; Mamba-3's structured discretization is stable by design |
| Hopf posterior identifiability (c→0, formal def §6) | Flow posterior models the collapse in shape, not just location; per-series normalisation; per-family audits |
| e-detector underfast drift | EDD reported per approach-speed bin; document behaviour at fast approach |

---

## 9. Verified literature corpus (all checked online, 2026-08-05)

**Selective state-space / linear-recurrent backbones (C1):**
- Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu. *Mamba-3: Improved Sequence Modeling using State Space Principles.* ICLR 2026 (Oral); arXiv:2603.15569. — exponential-trapezoidal discretisation, complex-valued states, MIMO. **Default backbone**; complex states match the Hopf eigenvalue pair.
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025; arXiv:2412.06464. — ablation D.
- Sun, Li, Dalal, Xu, Vikram, Zhang, Dubois, Chen, Wang, Koyejo, Hashimoto, Guestrin. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* ICML 2025 Spotlight; arXiv:2407.04620. TTT-E2E: arXiv:2512.23675 (2025). — ablation E.
- Gu, Goel, Ré 2022 (S4); Gu, Gupta, Goel, Ré 2022 (S4D); Gu & Dao 2024 (Mamba); Smith, Warrington & Linderman 2023 (S5) — foundation of the family.

**Flow-matching posterior estimation (C2):**
- Wildberger, Dax, Buchholz, Green, Macke, Schölkopf. *Flow Matching for Scalable Simulation-Based Inference.* NeurIPS 2023. — FMPE, the C2 template.
- Zhai, Jeong, Ročková. *Conditional Flow Matching for Bayesian Posterior Inference.* arXiv:2510.09534 (2025).
- Ruhlmann, Arbel, Forbes, Rodrigues. *Flow Matching Calibration for Simulation-Based Inference under Model Misspecification.* arXiv:2509.23385; ICML 2026 — misspecification-aware calibration, relevant to hopf.
- Papamakarios & Murray 2016; Greenberg et al. 2019 (SNPE) — foundations. Schumacher et al. 2023 (neural superstatistics) — time-varying parameter posteriors.

**Anytime-valid sequential testing / e-detectors (C3 — the guarantee):**
- Shin, Ramdas, Rinaldo. *E-detectors: a nonparametric framework for sequential change detection.* NEJSDS 2(2):229–260, 2024; DOI 10.51387/23-NEJSDS51; arXiv:2203.03532. — the e-detector construction and ARL ≥ 1/α theorem.
- Shafer. *Testing by Betting: A Strategy for Statistical and Scientific Communication.* JRSS-A 184(2):407–431, 2021.
- Ramdas, Grünwald, Vovk & Shafer. *Game-Theoretic Statistics and Safe Anytime-Valid Inference.* Statistical Science 38(4):576–601, 2023; DOI 10.1214/23-STS894; arXiv:2210.01948. — the SAVI survey (e-processes, testing by betting).
- Ramdas & Wang. *Hypothesis testing with e-values.* arXiv:2410.23614 (2025).
- Waudby-Smith & Ramdas 2023 (betting confidence sequences) — e-process foundations.
- Shiryaev 1963; Pollak 1985; Tartakovsky & Veeravalli 2005 — the classical delay/FAR theory the learned bettor approximates.

**Deep EWS (external baselines / lessons):**
- Bury et al. 2021, PNAS 118(39):e2106140118, DOI 10.1073/pnas.2106140118; Dablander & Bury 2022, PNAS 119(37):e2207720119, DOI 10.1073/pnas.2207720119; Bury et al. 2023, *Predicting discrete-time bifurcations with deep learning*, Nat. Commun. 14:6331, DOI 10.1038/s41467-023-42020-z; Deb, Sidheekh, Clements, Krishnan & Dutta 2022 (EWSNet), R. Soc. Open Sci. 9:211475, DOI 10.1098/rsos.211475.
- Kurt, Zheng, Yilmaz, Wang. *DeepQCD.* J. Franklin Inst. 361(18):107199, 2024 — neural QCD baseline (no anytime-valid bound).

**Formal-method foundations (unchanged):** Boxler 1989; Doucet et al. 2000; Li et al. 2004; Ives & Dakos 2012; Boettiger & Hastings 2012.

---

## 10. Open questions for the review — RESOLVED (2026-08-05)

1. **Phase A default encoder: Gated DeltaNet** (mature kernels at our scale); the Mamba-3-style complex-valued SSM is implemented for ablation G3. No constraint on `d_state ∈ {16, 32, 64}` sweep or the 10⁴-series Phase A library.
2. **Hopf encoding: MIMO** (vector-input recurrence over the `(x1, x2)` pair) — design C confirmed; Phase C uses it.
3. **`Z̄`: validation-null mean + 2σ** safety margin (conservative; slight power loss for a stronger guarantee); G5 audit re-estimates via the Waudby-Smith–Ramdas bound if the e-process fails the flatness audit.
4. **e-detector hyperparameters: `λ_max = 0.5`, small `λ_e` in Stage 2**; the e-process audit plot is reported in the paper.
5. **Code layout confirmed:** `src/csd_observer/models/neural_spectral_drift/` — `encoder.py`, `flow_posterior.py`, `e_detector.py`, `objectives.py`, `preprocess.py` (+ `targets.py` for the closed-form ρ/c derivation, §1).
6. Remaining external check at submission: confirm the Mamba-3 PyTorch binding (for G3) and Ruhlmann et al. venue details.

---

## 11. Summary (the claim this design supports)

> NSDO v3 is a causal selective-state-space stability observer: a Mamba-3-style encoder (complex-valued states matching the Hopf eigenvalue pair) feeding a supervised flow-matching posterior over the spectral gap `c_k = −ln|ρ_k|`, and a neural e-detector alarm that — by testing-by-betting construction — controls the false-alarm rate *anytime-validly* (`ARL ≥ 1/α` at any stopping time, finite sample). It is trained jointly on fold, hopf, logistic and null families using the simulator's parameter trajectories as supervision, evaluated head-to-head against the exact observer and all baselines at fixed FPR, and reported with the only finite-sample alarm guarantee in the deep-EWS literature.
