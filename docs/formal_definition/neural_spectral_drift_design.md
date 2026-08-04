# Neural Spectral-Drift Observer (NSDO) v2 — From-Scratch Design

**Status:** DRAFT v2 — free design. Supersedes `neural_spectral_drift_design_v1_identity_rbpf.md` (archived).
**Date:** 2026-08-05
**Grounded in:** `docs/formal_definition/formal_defininition.md` (for the *quantity* to track, not the *mechanism*)
**Companion:** `docs/formal_definition/literature_review.md`; `docs/plan/kalman_spectral_drift_plan.md`

---

## 0. Design brief (read this first)

This document designs the neural early-warning system **from scratch**. Previous constraints are explicitly released:

- **No identity-initialisation contract.** The network does not have to contain the exact RB-PF observer at init, and there is no "subsumption" claim. The exact observer is a *baseline to beat*, not a component.
- **No particle filter.** Particles were the exact method's engine. The neural version uses an amortised posterior — one forward pass, fully differentiable, scalable, and trainable with direct supervision.
- **No fold-only scope.** The neural version is trained on **all three benchmark families (fold, hopf, logistic) plus nulls**, with a single unified stability target (below). The fold-only restriction belonged to the formal math's reduction, not to the network.
- **No fixed alarm semantics.** The alarm is a **learned decision head** trained with a quickest-detection-aware objective. It may *use* the posterior probability `Pr(c_k < δ)` as an input feature, but is not forced to be a threshold on it.
- **The fixed-FPR evaluation protocol stays** — but as *measurement governance* (the benchmark compares all methods at FPR 0.05), not as an architectural rule.

**Decision criterion: best detection results** (AUC, detection lead, FPR) on the benchmark at fixed FPR, with the honest measurement protocol already in `docs/plan/kalman_spectral_drift_plan.md`.

**What survives from the formal math — and only this:** the *quantity* to track. Near a bifurcation the dominant multiplier `ρ` satisfies `|ρ| → 1`; the spectral gap `c_k = −ln|ρ_k| → 0` is the distance to the transition in log-space. That quantity is the network's supervised target (§2). The math says *what* to estimate — the network is free about *how*.

---

## 1. The problem, restated for a network

**Input:** online stream of centred observations `y_{1:k}` (scalar or low-dim), causality required (no future information at time `k`).

**Output:** an alarm decision `a_k ∈ {0, 1}` per step, evaluated as: alarm as early as possible before the transition at `τ` (window `[τ−50, τ−5]`), at a fixed false-positive rate (FPR ≤ 0.05 on null series) — the benchmark's existing metrics and protocol.

**Supervision available during training (the key advantage):** the repository's simulator returns, for every training series, the full **parameter trajectory** (`r_k` / `μ_k`; `bifurcation.py:71`). The dominant multiplier `ρ_k` is a closed-form function of that parameter — from the linearisation of each generator's update (the repo's own convention, cf. the multiplier comment at `bifurcation.py:117`) — hence the gap `c_k = −ln|ρ_k|` is known at every step of every training series, for all three families:

| Family | dominant multiplier `ρ_k` | gap `c_k = −ln|ρ_k|` | behaviour at transition |
|---|---|---|---|
| fold (repo: `FoldBifurcationDataset`) | real, `ρ → 1⁻` | `→ 0` | `c` crosses the collapse threshold |
| hopf (repo: `HopfBifurcationDataset`) | complex pair, `|ρ| → 1` | `→ 0` | radial mode loses mean-reversion |
| period-doubling (repo: `LogisticMapDataset`, bifurcates at μ = 3.0, `bifurcation.py:228`) | real, `ρ → −1` | `→ 0` (`\|ρ\| → 1`) | multiplier crosses −1 |

So `c_k` is a **universal, type-independent stability coordinate**: collapse ⇔ `c_k < δ` (with `δ = 0.05` as the config's collapse threshold), exactly matching the formal definition's semantics for the fold case. The network's job: estimate the **posterior over `log c_k`** from the stream (a *neural stability observer*), and alarm on it.

This is the honest, strongest framing: every method in the benchmark (exact observer, LLV/BH12 state-space baselines, canonical indicators) estimates some function of the same eigenvalue quantity; the neural version gets to learn the estimator directly from supervised data.

---

## 2. Architecture overview

```
y_k ──▶ (C0) preprocessing (learned detrend + normalise) ──▶ x_k
x_k ──▶ (C1) causal encoder (GRU; S4D/Mamba optional)      ──▶ h_k   (recurrent state)
h_k ──▶ (C2) stability head: posterior over θ_k = log c_k  ──▶ (μ_k, σ_k), Pr(c_k < δ | y_{1:k})
(h_k, posterior) ──▶ (C3) alarm head: a_k = σ(MLP(...))    ──▶ decision a_k ≥ θ̂  (θ̂: FPR-calibrated)
```

Stateful, one causal pass. All components differentiable; all trained end-to-end (multi-task), staged curriculum (§4).

### 2.1 C0 — Preprocessing (learned, minimal)

- **Detrending:** the Dablander–Bury 2022 lesson is that preprocessing decides the result — so make it *learnable*: a small causal 1-D conv over `y` estimating the trend; residual `x_k = y_k − trend_k` is the network input. **Identity init to the running-mean detrender** (`center_window=50`, as in the exact observer) purely as a *warm start* — training is free to move away from it (this is an init convenience, not a contract).
- **Normalisation:** per-series scale estimated from the first `W₀ = 50` points (z-score), standard for EWS comparability; optionally learned.
- Gate: an ablation with fixed (non-learned) preprocessing; if the learned version does not win, drop it (§5).

### 2.2 C1 — Causal encoder

- **Default: GRU** (hidden 64–128) — simplest, stable, proven for online causal filtering.
- **Phase-2 option: S4D or Mamba** — the state-space models whose scalar-eigenvalue structure mirrors the math (the transition coefficient `φ = e^{−cΔt}` is exactly an SSM discretisation; Mamba's selectivity = time-varying gap). Long-context memory (HiPPO) suits slow drift. **Decision by ablation** (gate G3), not by dogma.
- **Auxiliary input channels (ablation, recommended ON):** the classic EWS statistics are cheap and informative — rolling variance and lag-1 autocorrelation at 2–3 window sizes, concatenated to `x_k` as extra channels. The network can ignore them if useless; they typically shorten training and improve low-noise performance. This is feature engineering, not method cloning: the encoder decides how to use them.
- No attention/transformer on the online path (cost, causality, no evidence of need on this task class).

### 2.3 C2 — Stability head (the neural stability observer)

From `h_k`, a small MLP outputs the **posterior over the log-gap** `θ_k = log c_k`:

```
μ_k = MLP_μ(h_k)
σ_k = softplus(MLP_σ(h_k))          ⇒  q(θ_k | y_{1:k}) ≈ N(μ_k, σ_k)
```

- **Supervised on simulator ground truth:** during training, the target `θ*_k = log(−ln|ρ_k|)` is *known* (the simulator computes `ρ_k`). Loss: Gaussian NLL + optional quantile term.
- The posterior yields the interpretable statistic `Pr(c_k < δ | y_{1:k}) = Φ((log δ − μ_k)/σ_k)` — the neural analogue of the formal definition's collapse probability, used as an *input feature* for the alarm head and as a *diagnostic* (reliability plots, §5). It is **not** the alarm by itself.
- **Regulariser (optional):** an ELBO/FIVO auxiliary objective (observation-model likelihood under the encoder's latent) keeps `h_k` physically meaningful and helps when supervision is sparse. Ablated (§5).

### 2.4 C3 — Alarm head (learned decision)

```
a_k = σ( MLP( [h_k, μ_k, σ_k, Pr(c_k < δ | y_{1:k})] ) )
decision:  alarm at first k with a_k ≥ θ̂
```

- **Training objective (Stage 2; §4):** quickest-detection-aware. For transition series, positives in `[τ−50, τ−5]` with **earlier-is-better weighting** (e.g. weight ∝ 1/(τ−k), clipped); for null series, all steps are negatives. Loss: weighted BCE. This trains the head to trade delay against false alarms *using the posterior as evidence* — the neural analogue of the Shiryaev delay/KL trade-off, learned rather than assumed.
- **Governance:** after training, the threshold `θ̂` is calibrated on the validation-null split to `fpr_target = 0.05` (exact same percentile rule as the benchmark's spectral-drift block, `studies/runner/benchmark.py:636–643`). All reported numbers at fixed FPR 0.05.
- Optionally, the alarm head's score can be interpreted via the posterior feature (`Pr(c_k<δ)` path) — interpretability bridge, reported as diagnostic.

---

## 3. Why this design wins (honest comparison)

| | Exact RB-PF observer | Bury 2021 CNN-LSTM (black box) | **NSDO v2 (this design)** |
|---|---|---|---|
| Latent stability estimate | yes (unsupervised PF) | no | **yes — supervised on ground truth** |
| Uses simulator's `ρ_k` labels | no | no | **yes (the decisive advantage)** |
| Alarm semantics | fixed threshold on posterior | fixed-window classifier | **learned head, QCD-aware, FPR-gated** |
| Multi-type | fold only | trained on many types | **fold+hopf+logistic+nulls, one target** |
| Inference cost | 500 particles × Kalman | heavy CNN | **one GRU pass** |
| Gradient-friendly | not designed for it | yes | **fully differentiable end-to-end** |

The decisive point: **the exact observer estimates the gap without ever seeing a label; the network gets the true gap trajectory on every training series.** It learns the mapping `stream → stability posterior` under supervision, so it can correct every misspecification (OU model error, finite-sample bias, noise heterogeneity) that the exact observer must suffer. Bury's net never tracks the stability quantity at all. This is the "from scratch" combination: modern causal encoder + supervised latent + learned decision.

No subsumption/identity claim is made: the network's value is *empirical* — measured head-to-head with the exact observer and all baselines at fixed FPR.

---

## 4. Training

### 4.1 Data library

- Generator: the repository's synthetic benchmark machinery (fold, hopf, logistic + nulls). The parameter trajectories are already returned per series (`bifurcation.py:71`); `ρ_k` (hence `c_k`) is derived from them in closed form per family (§1), so the supervised target requires no simulator changes — only a small derivation helper.
- Scale: Phase 1 ≈ 10⁴ series; Phase 2 ≈ 10⁵–10⁶ (Bury used 500k/200k; we need less because supervision on the latent carries the structure, but scale is cheap).
- Ranges: transition times `τ` spread over the series; approach speeds (several), noise levels (`σ_u`, `R` grids as in `configs/model/default.yaml`), lengths 500–1500.
- **Nulls are mandatory in every mix** (Boettiger & Hastings 2012 — prosecutor's fallacy; Dablander & Bury 2022).

### 4.2 Objectives (multi-task, staged)

```
Stage 1 (stability):  L = NLL(q(θ_k|y_{1:k}); θ*_k)          [+ λ·ELBO optional]
Stage 2 (alarm):      L = L_stab + λ_alarm · BCE_weighted(alarm, delayed-positive labels)
```

- Stage 1 trains C0/C1/C2 (the observer). Stage 2 adds C3 with a small `λ_alarm`, fine-tuning everything end-to-end. If end-to-end destabilises (gradient interference), freeze C1/C2 in Stage 2 and train only C3 — decision by validation, not by assumption.
- Determinism: fixed seed discipline as the exact observer.

### 4.3 Curriculum

L0 unit series (single constant c; clean) → L1 fold + logistic (period-doubling) at several speeds/noises → L2 nulls (calibration) → L3 hopf and transfer systems. Every level keeps nulls in the batch.

---

## 5. Evaluation

Identical protocol to `docs/plan/kalman_spectral_drift_plan.md` §3.5:

- Metrics: early-warning AUC (τ−50..τ−5), detection lead (DT), FPR at calibrated threshold, per system family and per noise level, all at fixed FPR 0.05.
- **Required ablations:**

| Config | C0 learned | C1 encoder | C2 supervised | C3 QCD head | Scope |
|---|---|---|---|---|---|
| Exact observer | — | — | — | — | baseline |
| A (minimal) | fixed | GRU | ✓ | posterior-threshold only | fold |
| B (+ alarm head) | fixed | GRU | ✓ | ✓ | fold |
| C (+ multi-type) | fixed | GRU | ✓ | ✓ | all 3 + nulls |
| D (+ learned C0) | ✓ | GRU | ✓ | ✓ | all |
| E (+ S4D/Mamba if G3 wins) | ✓ | SSM | ✓ | ✓ | all |

- Diagnostics: reliability plot of `Pr(c_k<δ)` vs empirical frequency; posterior-mean tracking error `E|θ̂_k − θ*_k|` on held-out series with known c (only possible with the neural version); alarm-score calibration.
- Gate discipline: every ablation reported at the same fixed FPR; no threshold cherry-picking.

---

## 6. Roadmap and gates (each gate must pass before the next)

- **Phase A — observer:** C0(fixed)+C1(GRU)+C2, Stage 1 training on fold+logistic+nulls.
  - G0 (fit): training NLL decreases and posterior-mean tracks `θ*` on held-out L0 (R² > 0.9 on clean series).
  - G1 (versus exact observer): on fold series with known `c`, posterior-mean MSE ≤ exact observer's posterior-mean MSE (the exact observer is unsupervised; this should be easy — if it is not, the network is not using the labels, redesign).
- **Phase B — alarm:** C3 with QCD loss; fixed-FPR calibration.
  - G2 (no regression): on fold at FPR 0.05, AUC and DT ≥ exact observer and ≥ canonical indicators.
  - G3 (encoder ablation): GRU vs S4D vs Mamba on stability-MSE + AUC; switch only on clear win.
- **Phase C — multi-type:** add hopf; logistic (period-doubling) enters alongside fold from Phase A, so the joint model covers all three families + nulls.
  - G4 (transfer): per-family AUC/DT at FPR 0.05 ≥ per-family baselines; null FPR audited per family.
  - G5 (calibration): reliability curve within 5 points; no systematic overconfidence on nulls.
- **Phase D — polish:** learned C0 if it wins the ablation; scale-up library; OOD holdout (families/parameters unseen in training) reported honestly; determinism check.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting the simulator (network exploits generator artifacts) | OOD holdout families; broad parameter ranges; nulls everywhere; per-noise-level reporting |
| Supervision target `ρ_k` noisy near transition | Gap in log-domain; NLL robust to outliers; quantile loss option |
| Alarm head overfits the delay weighting | Calibrate on validation; report AUC/DT/FPR, not the raw loss |
| Preprocessing sensitivity (Dablander–Bury 2022) | Learned C0 ablated against fixed; report both if they differ materially |
| Encoder instability (SSM training) | Phase 1 uses GRU; S4D init/HiPPO defaults if SSM adopted; gradient clipping |
| FPR drift after training | Threshold always re-calibrated post-training on validation nulls; per-family audit |
| Identifiability of c as c→0 (formal definition §6) | Per-series normalisation; posterior variance `σ_k` explicitly modelled and calibrated |

---

## 8. Verified literature corpus (all checked online, 2026-08-05)

**Amortised / supervised neural posteriors for time-varying parameters (the core technique):**
- Schumacher, Bürkner, Voss, Köthe, Radev 2023. *Neural Superstatistics.* Sci. Rep. 13:13778, DOI 10.1038/s41598-023-40278-3 — neural posterior over time-varying parameters; template for C2.
- Radev et al. 2023. *JANA: Jointly Amortized Neural Approximation.* PMLR 216 — joint amortised posteriors.
- Papamakarios & Murray 2016; Greenberg, Nonnenmacher & Macke 2019 (SNPE, ICML) — amortised SBI foundations.
- Khabibullin & Seleznev 2022. *Amortized SBI for Bayesian SSMs.* Bank of Russia WP 104.

**Neural quickest change detection (alarm head):**
- Kurt, Zheng, Yilmaz & Wang 2024. *DeepQCD.* J. Franklin Inst. 361(18):107199, DOI 10.1016/j.jfranklin.2024.107199 — RNN + QCD-aware training; template for C3.
- Wu, Diao, Banerjee, Ding & Tarokh 2024. *QCD for Unnormalized Statistical Models.* IEEE Trans. Inf. Theory 70(2):1220–1232, DOI 10.1109/TIT.2023.3328274.
- Shiryaev 1963; Pollak 1985; Tartakovsky & Veeravalli 2005 — the delay/FAR trade-off the head approximates.

**Causal encoders:**
- Gu, Goel & Ré 2022 (S4, ICLR, arXiv:2111.00396); Gu, Gupta, Goel & Ré 2022 (S4D, NeurIPS, arXiv:2206.11893); Gu & Dao 2024 (Mamba, arXiv:2312.00752); Smith, Warrington & Linderman 2023 (S5, ICLR).
- The eigenvalue-is-the-state identity: `φ(c) = e^{−cΔt}` is exactly an SSM discretisation with scalar eigenvalue `−c` — the mathematical reason an SSM encoder may fit this problem class, to be validated by G3.

**Deep EWS (external baselines / lessons):**
- Bury et al. 2021. *Deep Learning for EWS of Tipping Points.* PNAS 118(39):e2106140118, DOI 10.1073/pnas.2106140118.
- Dablander & Bury 2022. *Preprocessing Matters.* PNAS 119(37):e2207720119.
- Bury et al. 2023. *Predicting Discrete-Time Bifurcations with Deep Learning.* Nat. Commun. 14:6331 (verify at submission). Deb et al. 2022 (EWSNet) — pin at submission.

**Differentiable particle filters / deep Kalman (background, and the v1 design's core; here optional regularisers only):**
- Maddison et al. 2017 (FIVO, NeurIPS); Naesseth et al. 2017 (arXiv:1705.11140); Le et al. 2018 (AESMC, ICLR); Karkus et al. 2018 (CoRL, PMLR 87:169–178); Jonschkowski et al. 2018 (arXiv:1805.11122); Chen & Li 2025 (IEEE TSP 73:493–507); Krishnan et al. 2015/2017; Karl et al. 2017 (ICLR, arXiv:1605.06432); de Bézenac et al. 2020 (NeurIPS 33:2995–3007); Fraccaro et al. 2017; Becker-Ehmck et al. 2019 (ICML).

---

## 9. Open questions for the review

1. **Encoder default:** GRU (recommended) vs straight to S4D/Mamba; G3 decides. Confirm no attention on the online path.
2. **Auxiliary channels (rolling var/AC1):** default ON with ablation. Any objection?
3. **Alarm loss form:** weighted-BCE with earlier-is-better weighting (recommended) vs DeepQCD's sequential formulation; decide by G2.
4. **Multi-type headline:** report per family at fixed FPR (recommended) — fold, hopf, logistic each get their own AUC/DT/FPR row.
5. **Supervised target for hopf:** radial-mode gap `−ln|ρ|` (recommended) — confirm the simulator can emit `ρ_k` for all families; if not, train hopf with the fold/lambda-family only and document.
6. **Keep the v1 identity-design file as archive?** Currently archived at `neural_spectral_drift_design_v1_identity_rbpf.md` — delete if unwanted.
7. **Code layout:** `src/csd_observer/models/neural_spectral_drift/` — `preprocess.py`, `encoder.py`, `stability_head.py`, `alarm_head.py`, `objectives.py`. Confirm naming.

---

## 10. Summary (the claim this design supports)

> The Neural Spectral-Drift Observer is a causal neural stability observer: it learns, from simulator ground truth, an amortised posterior over the spectral gap `c_k = −ln|ρ_k|` — the universal distance-to-bifurcation coordinate — and alarms through a quickest-detection-trained head whose threshold is calibrated to FPR 0.05. Trained on the fold, hopf, logistic and null families jointly, it combines the formal math's *quantity* (the eigenvalue) with the neural method's *power* (supervised estimation of it), and is measured head-to-head against the exact observer and all identity-matched baselines at fixed FPR. It makes no subsumption claim: its value is empirical, and every component is ablatable.
