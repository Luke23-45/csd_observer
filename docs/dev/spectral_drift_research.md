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
## 7. FUSION AND EVIDENCE-MEMORY (latch) — the surviving concepts
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

VERDICT ON THE CONCEPTS: the only genuine improvement from the extra
Bayesian/evidence-accumulation ideas is the latched-variance detector on
hopf: coverage 0.40-0.55 -> 0.65-0.85 at matched FPR, lead +15-20 steps.
Fusion contributes lead (not coverage) on hopf. Everything else
(EMA/Kalman/BF on AC1, fusion on fold, all concepts on logistic) either
matches or loses to the real baselines. In particular, Kalman-Spectral-Drift
itself does not outperform the stronger classical indicators under the
persistence-aware protocol; on fold it only looks good through the trend
channel, which is not a CSD result.

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

===================================================================
## 9. CROSS-DOMAIN DETECTION LOGIC (deep research round)
==================================================================

Direction (user): stop improving increments; import genuinely new detection
concepts from other domains (ML streaming, aeroelastic flutter monitoring,
volcanology/geophysics, computational biology) and test them on this
benchmark under the persistence-aware protocol. Sources surveyed:

  * Voight 1988 (Nature 332:125) + Bell et al. 2011 (GRL): Materials Failure
    Forecasting Method — precursors accelerate as Omega_ddot = A * Omega_dot^a;
    the inverse-rate plot 1/Omega_dot vs t is LINEAR and extrapolates to the
    failure time. Our divergence laws are exactly this structure: hopf
    detrended-radial variance V ~ C/(tau - t) (mu(t) linear, radial
    fluctuation variance ~ sigma^2/(2|mu|)); fold raw window variance is
    trend-dominated (1/r) in the last steps, fluctuation-dominated
    ((tau-t)^-1/2) earlier — the fold's inverse-variance line is CURVED.
  * Flutter monitoring (BASILE, Basseville; Peeters & De Roeck; "mode-shapes
    correlation and CUSUM for online flutter monitoring"): subspace system
    identification + CUSUM on the residual — the flight-test answer to
    damping->0 detection from short noisy records.
  * Sequential/quickest change detection (Shiryaev; Tartakovsky & Veeravalli):
    CUSUM/Shiryaev-Roberts with ARL-ADD operating characteristics — the
    principled version of our persistence protocol; echoed by the online-MMD
    ML literature (alibi-detect ERT calibration; Kalinke et al. arXiv
    2505.17789 RFF-MMD with minimax delay).
  * Kernel two-sample drift (Gretton 2012; Bounliphone et al.): MMD between
    reference and rolling window as a nonparametric distributional-drift
    alarm.
  * Deep EWS (Bury et al. PNAS 2021 / Nat. Commun. 2023): CNNs trained on
    simulated bifurcations; only trajectory-level classification evaluated —
    sequential alarm timing with persistence has NOT been evaluated anywhere.
  * Limits of detection (Boettiger & Hastings 2012, J. R. Soc. Interface):
    error-rate analysis of EWS detection; not sequential, not persistence-aware.
  * Seizure prediction (Iasemidis et al.): the field's lead-time vs FPR
    protocol (74 min, 0.12/h) is the same lead-coverage tradeoff as ours.

Concepts tested (all benchmark governance, seeds 101/202, FPR ~0.05):

(a) VOIGHT INVERSE-RATE DETECTOR (cross_domain.py, voight_v.py).
    Rate-based (dV/dt finite differences) fails everywhere (fold K10 0.20,
    AUC 0.25-0.49): the differenced rate of a rolling variance is unusable.
    DIRECT 1/V(t) linear fit, score = -t-stat of the line slope:
      fold (raw V):  K0 0.45/55, K10 0.15/75, AUC 0.417 — curvature kills it.
      hopf (detrended V): K0 0.55/30, K10 0.20/19, AUC 0.525.
    FORECAST-STABILITY ALARM (score gated by "tau_hat finite, in-horizon,
    stable across last 10 steps" — the detector alarms exactly when its own
    collapse-time forecast becomes trustworthy):
      hopf voight-stab: K0 0.85/34.6, K10 0.55/36.2, AUC 0.925 (FPR 0.043)
        + TAU-FORECAST at first alarm: 17/20 trajectories, MAE 14.3,
        median 9.4 steps on tau=100 — a NEW capability (no other detector
        produces a calibrated time-to-collapse).
      fold voight-stab: K0 0.65/58, K10 0.35/51, forecast MAE 33.
    Latching the model-validity instead of the evidence adds nothing: the
    K10 drop is a model misspecification near tau (1/V is a "knee": floor
    plateau + divergence tail; the crossing leaves the horizon). Floor
    correction V_floor + C/(tau-t) is WORSE (noise amplification, K0 0.65).
    Mature forecasts degrade near tau (knee) — forecast MAE 33 at the last
    pre-tau step vs 14 at first alarm.

(b) MMD KERNEL DRIFT (mmd_drift.py): reference = early 40 steps, window w=20,
    Gaussian kernel with median heuristic, running max-with-decay latch:
      fold: K10 1.00/51.0, AUC 1.000 (FPR 0.042) — perfect persistence, but
        the drift is the LEVEL ramp (same channel as the observer trend),
        and the lead (51) is below the observer's 84.
      hopf: K10 0.10/11 — FAILS (kernel width set by the broad reference
        distribution; the divergence is too subtle for the raw-value MMD).
    MMD's value here is negative: on the pure-CSD system the nonparametric
    distribution drift is invisible while windowed variance catches it.

(c) LEVEL-INVERSION FORECAST (fold): x_bar^2 ~ r(t) (normal-form inversion),
    linear extrapolation to r=0: K0/K10 1.00/84-85, AUC 1.000 — identical to
    the observer's trend channel (the benchmark's tau IS the level crossing
    by construction; this is a sanity check, not a CSD detector). Forecast
    MAE 29-40 (local-slope overshoot from tracking lag); Voight's mature
    forecast on raw V does better (MAE 13, median 13).

CROSS-DOMAIN VERDICT: two of six concepts produce real signal on hopf — the
Voight forecast-stability alarm (AUC 0.925, K0 0.85) and its tau-forecast
(17/20, MAE 0.14*tau) — but NONE beats latched-var (0.70/55.9) on the
coverage/lead protocol; the hopf CSD is genuinely weak (obs-noise floor
0.15 swamps the divergence until ~10-20 steps before tau; the apparent
early variance rise is a detrending artifact of the decaying radius
transient). The fold is solved trivially by its level channel; the logistic
map is undetectable (theory-consistent: no variance divergence at its
period-doubling on this noise level). The literature survey confirms the
field has NO persistence-aware sequential protocol and NO time-to-collapse
forecast evaluation — both are ours to claim.

==================================================================
## 10. MECHANICS/VIBRATIONS/FLUIDS THEOREM ROUND (all negative)
==================================================================

User direction: import theorems from mechanics of solids, vibrations, fluid
mechanics. All tested under benchmark governance (seeds 101/202, FPR ~0.05):

(a) FREE-ALPHA FFM (Bell et al. 2011 GRL — the proper FFM fits the divergence
    exponent by profile likelihood over t_f). log V vs log(t_f - s) grid fit:
      fold: K0 0.65/41, K10 0.20/41, AUC 0.458, forecast MAE 42
      hopf: K0 0.55/20, K10 0.10/18, AUC 0.408, forecast MAE 25
    WORSE than the 2-parameter 1/V fit (MAE 14, K0 0.85): the t_f grid
    overfits 40-point windows (R^2 threshold inflated; AUC < 0.5).
(b) MONKMAN-GRANT min-stretch (life from the most stable 20-step rate):
      fold K10 0.05, hopf K10 0.05 — dead.
(c) STIFFNESS CHANNEL (Euler-buckling analog: drift-reconstruction slope
    kappa(t) -> 0): hopf K0 0.00 (obs-noise attenuation floors the slope
    until the variance grows, by which time it is the variance channel);
    fold K0 0.25/5.1 (the slope is dominated by the equilibrium x_bar which
    stays large until the end). Dead.
(d) CHI-SQUARE CUSUM ON BASELINE-FIT RESIDUAL (flutter-monitoring theorem,
    Basseville/BASILE): S_t = max(0, S_{t-1} + z_t^2 - 1). Theoretically
    persistent-by-construction and ARL-calibrated, but the max-filter gives
    the null CUSUM an intrinsic O(sqrt(t)) upward drift -> empirical and ARL
    thresholds inflate (fold 1405, hopf 51, 2-ch hopf 6919): fold K0 0.15/3.7,
    hopf K0 0.10/1.5; the raw z^2 energy crosses early on hopf (K0 0.85/18.8)
    but does NOT sustain (K10 0.00). Dead.
(e) EVT TAIL / INTERMITTENCY (turbulence concept, type-I/III intermittency
    near period-doubling): Hill tail index degenerates (NaN); rolling
    kurtosis on logistic: K0 0.30/11.2, K10 0.10 — nothing.

MECHANICS-ROUND VERDICT: every imported theorem fails — not from estimator
quality but from the benchmark's physics: the divergence is slow (ramp time
~100 steps) and the observation-noise floor (hopf 0.15 vs fluctuation scale
~0.025-0.125) swamps the signal until ~10-20 steps before tau. Detection is
limited by signal-to-noise and monitoring length, exactly as the limits
literature (Boettiger & Hastings 2012) argues. The two-parameter Voight 1/V
forecast (section 9) remains the best forecast structure; latched-var the
best detector. This closes the cross-domain exploration: the remaining
honest gains are the protocol + forecast dimensions, not a new estimator.

=================================================================
## VERDICT
==================================================================

* As configured: benchmark numbers are artifacts (broken selection + prior-floor
  crossing + rank artifact). Not defensible as success.
* Best-case (scale-matched, lively q): fold ~25-step lead, hopf ~13-39 — real
  but modest, and still worse than the stronger baselines on persistence.
* The Bayes-factor variant beats the observer on fold in raw lead, but it is
  trend-based and uncalibrated, so it is not a CSD win.
* The method's measurement (variance via OU-gap) cannot detect CSD on this
  benchmark's data; the CSD content lives in autocorrelation.
* ABLATION VERDICT (persistence-aware): applying our Bayesian concepts
  (EMA/Kalman/drift-BF) to the best baselines does not improve sustained
  detection in a defensible way. The only thing that outruns the baselines is
  the observer's own trend channel on fold (100% coverage), which should be
  reported as trend detection, not CSD.
* CONCEPTS VERDICT (fusion + latch): the evidence-memory latch applied to the
  variance channel is the one real improvement: hopf K10 coverage 0.40-0.55 ->
  0.65-0.85 (stable across seed families) and lead 37.5 -> 56 on the reference
  family. Fusion (min-p of calibrated channels) adds ~20 steps of sustained
  lead on hopf but no coverage. Final best sustained detectors at FPR ~0.05:
  fold = observer CP (0.03,1e-4): frac 1.00, lead 84, but this is a trend
  channel rather than a CSD result;
  hopf = latched-var w=20/30 lam=0.99: frac 0.65-0.70, lead 56-57 (the only
  genuinely stronger sustained result in this report; reproducible from
  latch_sweep.py / latch_stab.py);
  logistic = no method detects (all frac <= 0.15).
* CROSS-DOMAIN VERDICT (section 9): no imported concept beats latched-var on
  the coverage/lead protocol. The two NEW contributions are (i) the Voight
  forecast-stability alarm — hopf K0 0.85/34.6, AUC 0.925 — which alarms
  exactly when its collapse-time forecast becomes trustworthy, and (ii) a
  calibrated time-to-collapse forecast (17/20 coverage, MAE 0.14*tau, median
  0.09*tau) — a capability no EWS paper in the literature provides. The
  honest paper from this benchmark is: (1) persistence-aware protocol + lead-
  coverage frontier as the missing evaluation standard; (2) the collapse-time
  forecast dimension with the Voight (FFM) mechanism imported from
  volcanology/aero flutter monitoring; (3) the honest per-system verdict —
  fold is level-detectable (1.00/84, trend), hopf is weakly CSD-detectable
  (0.70/56 latched-var; 0.85/35 Voight-stability), logistic is undetectable
  (theory-consistent).
