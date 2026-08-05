# Spectral-Drift Observer: In-Depth Research Findings

Status: research notes (no code changes). All numbers: patients_100, seeds 101/202
family, benchmark governance (fixed-FPR calibration on val nulls, DT/AUC/FPR on
test). Reproducible from the scripts used during this study.

==================================================================
## 1. WHY THE CURRENT BENCHMARK NUMBERS ARE ARTIFACTS
==================================================================

The default run (your patients_100..500 results) reported AUC ~1.000 and
DT = tau exactly on every system. Three compounding mechanisms:

(a) Grid search returns the FIRST grid point. The validation criterion
    `grid_search_sigma_u_q_drift` selects by EW-AUC, which saturates at 1.000
    for nearly every (sigma_u, q): the collapse-prob scores are ~0.001-0.05
    everywhere in the early window on nulls, and any tiny rank separation gives
    AUC=1.0. The search cannot discriminate; the first combo (0.15, 1e-6) wins
    for all three systems (verified in results rows: q=1e-6, su=0.15 everywhere).
    The OLD grid (pre-restructure, diagnosis note) had real AUC spread and picked
    q=0.001-0.1 — the current q-grid order + saturated metric is the regression.

(b) DT = tau is a prior-floor crossing, not a detection. Threshold ~0.06; the
    t=0 posterior equals the prior P(c<0.05) ~ 8-9% > threshold, on signal AND
    null trajectories alike. The "alarm" fires at step 0 out of prior
    uncertainty; FPR stays near 0.05 only because t=0 is 1 step of 200.

(c) AUC=1.0 at (0.15,1e-6) is rank separation of two near-zero distributions:
    fold signal early-window max 0.008 vs null 0.003. Not a warning signal.

==================================================================
## 2. THE (sigma_u, q) RESPONSE SURFACE (DT @ FPR 0.05, seed 0)
==================================================================

Best cells (test-selected; stability across seed offsets in parens):

  fold (tau=133.3):  (0.03, 1e-4) DT=109 lead ~25  (24.8 / 27.7 / 24.6, AUC 1.0)
  hopf (tau=100.0):  (0.03, 1e-4) DT=78  lead ~22  (22.4 / 39.1 / 13.0, AUC 1.0)
  logistic (66.7):   no genuine content; "leads" 5-9 are prior-floor crossings

Key structural observations:
  * q=1e-6 (selected) is the degenerate regime: null collapse prob -> 0, so the
    threshold collapses to ~0.06 and the t=0 prior floor dominates. q=1e-4..1e-3
    is the "lively" regime where the score carries information.
  * Small sigma_u (0.03-0.08) is required: it makes the model variance
    comparable to the data, so the likelihood is c-sensitive. At sigma_u=0.15
    (default grid start) the model is ~2x too loud for fold -> c_hat rises,
    collapse prob falls toward 0 as tau approaches (0.08 -> 0.004).
  * These "best" cells are test-selected: the grid search cannot find them via
    AUC (saturated). A different selection criterion is required (section 6).

==================================================================
## 3. WHAT THE COLLAPSE-PROB SCORE ACTUALLY DETECTS (per system)
==================================================================

Same-input detector comparison (fixed-FPR governance, test split, seed 0):

  fold:   collapse-prob (0.03,1e-4)   DT=107.6 lead 25.8 AUC 1.000
          |centred mode| (trend)       DT=114.8 lead 18.5 AUC 0.973
          rollvar30 no-detrend         DT= 77.7 lead 55.7 AUC 1.000
          VAR-CSD (detrended)          DT= 77.8 lead 55.5 AUC 0.430  <- inverted!
  hopf:   collapse-prob (0.03,1e-4)    DT= 80.1 lead 19.9 AUC 0.997
          |centred mode|               DT= 75.2 lead 24.8 AUC 0.630
          rollvar30 no-detrend         DT= 37.5 lead 62.5 AUC 0.615
          VAR-CSD (detrended)          DT= 60.5 lead 39.5 AUC 0.610
  logistic: collapse-prob (0.15,1e-4)  DT= 59.1 lead  7.6 AUC 1.000 (artifact)
          rollvar30 no-detrend         DT= 13.4 lead 53.2 AUC 1.000 (burst artifact)
          VAR-CSD (detrended)          DT= 27.9 lead 38.8 AUC 0.618

Findings:

FOLD — the early rise is TREND LEAKAGE, not CSD:
  * The centred-mode std is FLAT (0.185, 0.180, 0.180, 0.202 in [0,60),[60,110),
    [110,125),[125,132)) until the last ~8 steps. No fluctuation-variance
    content early.
  * The causal running mean (window 50) LAGS the declining equilibrium
    (x* = sqrt(r(t)): 1.41 -> 0.22 from t=0 to 130). The residual one-sided
    offset accelerates toward tau, the filter reads growing innovations as
    variance -> collapse prob rises 0.11 -> 0.52 -> 0.85 from t=0.
  * Symmetric (non-causal) centring destroys the lead: 24.8 -> 7.0.
  * The benchmark's VAR-CSD (within-window linear detrend) correctly removes
    this trend: AUC 0.43 on fold = inverted (null variance exceeds signal
    variance early), matching the documented inversion caveat.
  * The REAL early content on fold is autocorrelation (AC1 of the centred mode:
    0.125@60 -> 0.325@130) — exploited by AC1/SRATIO/RETRATE (lead 54-73),
    invisible to the observer.

HOPF — genuine variance content, but simple detectors beat the observer:
  * Radial mode variance genuinely rises (0.014 -> 0.030 from t=35); no
    equilibrium trend to confound (pre-bifurcation radius ~ 0).
  * The observer gets lead 13-39 (AUC 1.0) — better rank separation than
    VAR-CSD (AUC 0.61) but the lead is variable and smaller than the simple
    detectors' (rollvar 62.5, VAR-CSD 39.5, |mode| 24.8).

LOGISTIC — weak gradual content, no method extracts it:
  * Signal variance genuinely rises early (0.045@35 -> 0.09@133; +26% vs null
    at t=35) but per-step distributions overlap (rolling-var AUC: 0.51@35,
    0.62@45, 0.71@65). The apparent "leads" of 38-53 steps are sporadic-burst
    first crossings (logistic chaos), not sustained detections.
  * Consistent with every benchmark method being ~chance on logistic.

==================================================================
## 4. THE STRUCTURAL LIMITATION (deepest reason)
==================================================================

The OU-gap model ties variance and autocorrelation through the single gap c:
    Var(u) = sigma_u^2 / (2c),   AC1 = exp(-c) * Var(u)/(Var(u)+r).

The fold data's (Var ~ 0.034, AC1 0.13->0.33) combination lies OFF the model
manifold: at any (sigma_u, c) the model cannot simultaneously produce "low
variance with rising autocorrelation". Hence the collapse-prob score can never
exploit the fold's genuine CSD content (the AC rise), which AC1/SRATIO/RETRATE
measure directly. The observer's only measurement channel is variance scale,
and (properly detrended) variance carries no early fold content.

==================================================================
## 5. TWO-MODEL BAYES-FACTOR SCORE (research prototype, no repo changes)
==================================================================

Score = P(c is drifting down | data), from the per-step predictive log-lik of
two RB filters (drift alpha vs static). Results (su=0.03 fold/hopf, 0.15
logistic, q=1e-4, N=300):

  fold:  alpha=0.005 -> lead 61.9, AUC 0.913; alpha=0.001 -> lead 85.4, AUC 0.657
         (best lead of ANY detector measured — but scores saturate at 1.0, so
          FPR calibration is unstable: thr=1.000, FPR 0.06-0.077)
  hopf:  no separation (drift model rejected by the likelihood)
  logistic: fails / inverted

The fold lead is still the trend mechanism (the drift model absorbs the
accelerating equilibrium decline as "c shrinking"), amplified by Bayesian
evidence accumulation. It is NOT a CSD detector.

==================================================================
## 6. ABLATION PROGRAM (user direction: apply our best concepts to the
##    best existing methods; persistence-aware evaluation; no repo changes)
==================================================================

Round 1 (fold) — ground truth + Bayesianised-AC1 matrix (r1_fold.py, r1d_sratio.py):
  detector                 w    thr   AUC   FPR  | K0 frac/DT (lead)    | K10 frac/DT (lead)
  AC1 raw                 30   0.256 0.718 0.077 | 0.80 / 68.0 (65.3)   | 0.55 / 32.5 (77.9 cons)
  AC1 raw                 50   0.254 0.667 0.071 | 0.85 / 54.3 (79.1)   | - / -
  SRATIO                  30   2.704 0.708 0.079 | 0.75 / 59.9 (73.4)   | 0.45 / 25.7
  SRATIO                  50   2.480 0.665 0.073 | 0.85 / 33.7 (99.6)   | 0.55 / 21.0
  RETRATE                 30  -1.357 0.729 0.076 | 0.80 / 59.1 (74.2)   | 0.50 / 23.6
  RETRATE                 50  -1.466 0.667 0.071 | 0.90 / 45.7 (88.0)   | 0.55 / 20.9
  EMA z (lam=0.05/0.1/0.2) 30   -    0.627/0.660/0.685 | DT 33.0/35.7/34.4 (K0), K10 == K0
  Kalman-level z q=1e-4   30  0.241 0.682 0.076 | 0.65 / 34.7           | 0.55 / 37.1
  drift log-BF a=0.002    30 19.90  0.725 0.007 | 0.05 / 27.3 (K0=K10)  | 0.05 (95% never alarm!)
  drift log-BF a=0.005    30   -    0.773 0.005 | 0.05 (same)           | 0.05
  observer CP (0.03,1e-4)  -   0.333 1.000 0.069 | 1.00 / 107.6          | 1.00 / 87.7 (45.6 lead)
  (R = 0.00084 MoM z-innovations from val nulls — consistent with noise-dominated AC1)

Rounds 2-3 (hopf, logistic) — persistence battery (r23_battery.py):
  hopf (tau=100):  VAR-CSD w=30   0.022 0.610 0.053 | 0.80 / 60.5 | 0.60 / 41.2   <- best sustained
                   AC1 w=30       0.260 0.783 0.028 | 0.75 / 57.5 | 0.35 / 35.0
                   rollvar30 no-d 0.024 0.615 0.043 | 0.60 / 37.5 | 0.40 / 36.0
                   observer(0.03) 0.151 0.997 0.015 | 0.85 / 80.1 | 0.25 / 46.4   <- AUC lies: 25% sustain
  logistic (tau=66.7): every method K10 frac <= 0.15 (VAR 0.10, AC1 0.10, rollvar 0.15,
                   observer 0.10; observer K0 0.85/59.1 is the t=0 prior-floor crossing).
                   -> NO method detects logistic under persistence.

PROGRAM FINDINGS:
1. PERSISTENCE IS THE MISSING EVALUATION DIMENSION. Every benchmark headline lead
   is transient: fold baselines' reported 54-100-step "leads" are 21-33-step
   sustained with 35-55% trajectory coverage. The benchmark's DT (first crossing
   before tau, mean over alarmers) must be reported with (a) alarm fraction and
   (b) K-step sustained crossing to be honest.
2. BAYESIAN CONCEPTS DO NOT IMPROVE THE BEST BASELINE. EMA-smoothing,
   Kalman-level filtering, and drift Bayes-factor evidence accumulation on the
   AC1 channel all land AT or BELOW raw AC1 under persistence (K10 frac 0.55,
   lead 33-37). The sustained content is data-limited, not estimator-limited.
   The drift-BF "lead 62-85" was a mean-over-1-alarmer illusion (frac 0.05).
3. THE OBSERVER'S ONLY WIN IS ITS TREND CHANNEL. At (0.03, 1e-4) on fold it is
   the ONLY detector with 100% sustained coverage (lead 45.6, AUC 1.000) — but
   the mechanism is the equilibrium-decline trend (Section 2), not CSD. On hopf
   it collapses to 25% sustained coverage (AUC 0.997 is a rank artifact).
4. AUC MUST BE READ WITH COVERAGE. observer hopf AUC 0.997 / 25% coverage;
   logistic AUC 1.000 / 10% coverage. Near-1 AUC from rank separation of
   mostly-below-threshold scores says nothing about alarmability.
5. LOGISTIC IS UNDETECTABLE BY ANY MEASURED METHOD under persistence
   (all baselines included) — no sustained content exists in its window
   statistics before tau on this data family.

Final sustained-lead ranking (K10):
  fold:  observer trend 1.00/45.6 > AC1 raw 0.55/32.5 ~ EMA/Kalman 0.55/33-37
         > RETRATE 0.50/23.6 ~ SRATIO 0.45/25.7; driftBF 0.05 (dead)
  hopf:  VAR-CSD 0.60/41.2 > rollvar 0.40/36.0 ~ AC1 0.35/35.0 ~ observer 0.25/46.4
  logistic: nothing (<=0.15 coverage)

==================================================================
## 7. FUSION AND EVIDENCE-MEMORY (latch) — the winning concepts
==================================================================

The persistence battery exposed hopf's structure: the observer's collapse prob
is a genuine early PULSE (crossing t~20 on 85% of trajectories) that the filter
un-learns — on 100% of trajectories the score falls back below threshold and
stays below until tau (score@tau mean 0.133 < thr 0.151; "stay" frac 0.00).
Meanwhile the variance channel sustains (0.55) but crosses late (lead 37.5).
Two concepts were tested to exploit this:

(a) MULTI-CHANNEL EVIDENCE FUSION (fusion.py). Per-step upper-tail null
p-values (ECDF per step from val nulls, p = 1 - ECDF(x) -- the p direction
matters: these are upper-tail alarms) combined as min-p (=-log max evidence)
or Fisher. At matched FPR (pooled-null threshold; per-step ECDF tail is too
noisy for the val-only threshold: FPR inflated to 0.15-0.25):
  hopf minp{obs,var}  K10 0.55 / lead 57.7 (FPR 0.043)  vs  raw var 0.55/37.5
  hopf minp{obs,var,ac1} K10 0.60 / 55.9 (FPR 0.072)
  fold: nothing beats the observer alone (1.00/84.4).
Verdict: fusion adds ~20 steps of sustained LEAD on hopf at equal coverage --
the observer pulse arms the alarm early, the variance body carries it -- but
no coverage gain (union null distributions are fatter -> higher thresholds).

(b) EVIDENCE-MEMORY LATCH (latch_fusion.py, latch_sweep.py). Running max-with-
decay of the channel's upper-tail log-evidence: E(t) = max(lambda*E(t-1), e(t)).
The Bayesian-posterior-with-momentum idea: once strong evidence appears, keep
it until decayed. Result (hopf, pooled threshold, FPR ~0.05):
  latched-var w=20 lam=0.99:  K10 0.70 / lead 55.9 (FPR 0.057)
  latched-var w=30 lam=0.99:  K10 0.65 / lead 56.9 (FPR 0.047)
  vs raw var w=20: 0.40/42.5;  raw var w=30: 0.55/37.5  (raw window sweep)
  lambda insensitive (0.97/0.99/0.995 identical: the var body re-arms the latch).
  Latch-then-fuse (per-channel latch, then max) does NOT beat the single
  latched var: union thresholds rise faster than coverage (0.55-0.65).
  Fold: latch adds nothing (observer already 1.00; latched-obs 1.00/86.5,
  FPR 0.098 -- slightly worse). Latched-var fold: 0.45 -- irrelevant.
Seed-family stability (offsets 0/1/2, pool threshold):
  off0 w20 0.70/55.9  w30 0.65/56.9 | off1 w20 0.55/27.1  w30 0.70/24.1
  off2 w20 0.85/42.6  w30 0.80/45.4   (coverage holds 0.55-0.85; lead varies
  with family; FPR 0.047-0.094 -- the latch's null blocks make FPR control
  slightly loose; report it).

VERDICT ON THE CONCEPTS: the ONLY substantive, honest improvement the
Bayesian/evidence-accumulation concepts produced is the latched-variance
detector on hopf: coverage 0.40-0.55 -> 0.65-0.85 at matched FPR, lead
+15-20 steps. Fusion contributes lead (not coverage) on hopf. Everything
else (EMA/Kalman/BF on AC1, fusion on fold, all concepts on logistic) either
matches or loses to the raw baselines -- the data is the limit, not the
estimator.

==================================================================
## 8. OPEN QUESTIONS / RESEARCH DIRECTIONS
==================================================================
1. (var, AC) identifiability: relax the single-c constraint (e.g., add a
   variance-scaling parameter, or a spectral-slope measurement) so AC and
   variance are estimable independently — the only path to a genuine CSD story
   on fold.
2. Selection criterion: the val-AUC saturation must be broken (effect-size
   margin at fixed FPR, or DT-based selection, or MoM sigma_u from val-null
   innovations) before any re-run is meaningful.
3. Bayes-factor calibration: track the unbounded log-BF instead of sigmoid
   scores; tune alpha on validation; then re-test fold.
4. Honest framing: the equilibrium-decline detection (fold) is a real "approach"
   detector but must be labelled as trend detection, benchmarked against
   |mode|/rollvar, and clearly separated from CSD claims.
5. Persistence-aware evaluation: adopt (alarm fraction, K-step sustained DT) as
   the headline pair in the benchmark before any further method comparisons.
6. Only remaining untested fusion: combine the trend channel (observer CP on
   fold) with the AC channel (AC1/VAR) — independent signals, both real on fold
   (trend) and hopf (variance). On hopf the trend channel fails persistence, so
   fusion would need the variance channel to gate it — low expected value;
   suggest NOT pursuing unless user wants the fusion study for completeness.

==================================================================
## VERDICT
==================================================================

* As configured: benchmark numbers are artifacts (broken selection + prior-floor
  crossing + rank artifact). Not defensible as success.
* Best-case (scale-matched, lively q): fold ~25-step lead, hopf ~13-39 — real
  but modest, and beaten in lead by simple baselines on every system.
* The Bayes-factor variant beats everything on fold (62-85) but is trend-based
  and uncalibrated.
* The method's measurement (variance via OU-gap) cannot detect CSD on this
  benchmark's data; the CSD content lives in autocorrelation.
* ABLATION VERDICT (persistence-aware): applying our Bayesian concepts
  (EMA/Kalman/drift-BF) to the best baselines does NOT improve sustained
  detection anywhere — the methods are data-limited, and the Bayesian machinery
  only redistributes the same crossings. The one thing that beats baselines is
  the observer's own trend channel on fold (100% coverage), which is not CSD.
  Recommended report framing: persistence-aware comparison table as above;
  label the observer's fold detection as trend detection; either fix the
  (var,AC) identifiability or drop the CSD claim on this benchmark.
* CONCEPTS VERDICT (fusion + latch): the evidence-memory latch applied to the
  variance channel is the one real improvement: hopf K10 coverage 0.40-0.55 ->
  0.65-0.85 (stable across seed families) and lead 37.5 -> 56 on the reference
  family. Fusion (min-p of calibrated channels) adds ~20 steps of sustained
  lead on hopf but no coverage. Final best sustained detectors at FPR ~0.05:
  fold = observer CP (0.03,1e-4): frac 1.00, lead 84 (trend channel);
  hopf = latched-var w=20/30 lam=0.99: frac 0.65-0.70, lead 56-57 (this report's
  headline result; reproducible from latch_sweep.py / latch_stab.py);
  logistic = no method detects (all frac <= 0.15).
