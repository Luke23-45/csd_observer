# Persistence-Aware Evaluation of Early Warning Signals: Literature Review

Status: new research direction (the previous spectral-drift observer direction is closed).
Every citation below was verified against the primary source at the time of writing
(URLs provided). Our own numbers come from the project benchmark experiments
(seeds 101/202, classic generators, fixed-FPR calibration on validation nulls,
K0/K10 sustained-crossing protocol). Nothing in this document is claimed without
a verified source; where a claim is ours, it is labelled as such.

==================================================================
## 1. Purpose and scope
==================================================================

This review establishes the evidence base for the paper's core claim:

  The early-warning-signal (EWS) literature evaluates detectors with
  single-shot, per-trajectory metrics (trend statistics, ROC/AUC,
  classification accuracy, single confidence-band crossings). None of
  it defines an alarm as a *sustained* event, none reports trajectory-level
  coverage under a controlled per-step false-positive rate, and none
  documents the failure mode in which an indicator crosses a threshold
  early and then never stays above it. This project supplies that missing
  evaluation protocol and demonstrates that its adoption changes the
  ranking of existing methods.

The previous project direction (a Bayesian spectral-drift observer for CSD
detection) failed under exactly this evaluation: its headline AUC values
proved to be rank artifacts of mostly-below-threshold scores, and its alarms
did not persist. That failure is the empirical motivation for the protocol,
not a bug in the benchmark.

==================================================================
## 2. How the field evaluates EWS today (verified)
==================================================================

### 2.1 Trend statistics on running indicators (the canonical approach)

Dakos et al. (2008, PNAS 105(38):14308-14312) — the canonical early-warning
methodology — fit an AR(1) model in a sliding window, extract the lag-1
autocorrelation, and test for an increasing trend with Kendall's rank
correlation coefficient. Significance is assessed against 1,000 surrogate
time series under three null constructions (block bootstrap, iAAFT
phase-randomised, AR(1) with matched variance and autocorrelation). The
evaluation unit is the *trend statistic of one time series*; there is no
alarm event, no alarm duration, and no false-alarm protocol.
https://www.pnas.org/doi/abs/10.1073/pnas.0802430105
(verified: sliding-window AR1 + Kendall tau + surrogate significance)

This template — running-window indicator + trend test + surrogate or
baseline significance — dominates the applied literature (see the
systematic review in 2.6).

### 2.2 Per-trajectory error rates: ROC / AUC

Boettiger & Hastings (2012, J. R. Soc. Interface 9(75):2527-2539) is the
first systematic error-rate analysis: a model-based likelihood approach
compared against summary statistics on ROC curves, with tables of "fraction
of true crashes caught at a 5% false-positive rate". The evaluation is
per-dataset / per-trajectory classification; the ROC sweeps a discrimination
threshold over a summary statistic of the whole record. No sequential alarm
is defined.
https://royalsocietypublishing.org/doi/10.1098/rsif.2012.0125
(verified: ROC curves, 5% FPR tables, likelihood-vs-summary-statistic)

Bury et al. (2021, PNAS 118(39):e2106140118) — deep-learning EWS — evaluate
with F1 score (84.2% on 500-point series, 88.2% on 1500-point series) and
ROC/AUC against lag-1 AC and variance, with classifiers given access to
80-100% vs 60-80% of the series. The evaluation unit is again the trajectory
(binary classification: transition vs no transition).
https://www.pnas.org/doi/10.1073/pnas.2106140118
(verified: F1, ROC/AUC, trajectory-level classification)

The 2024 benchmark in Nonlinear Dynamics
(10.1007/s11071-024-10023-0) — "Early warning signals of complex critical
transitions in deterministic dynamics" — systematically tests EWS on
deterministic systems and summarises performance with ROC/AUC over
transition vs null models, with warnings defined as falling outside a
confidence band derived from the baseline mean +/- sigma_crit. Again a
per-trajectory classification with a swept threshold; alarm duration is
not part of the evaluation.
https://link.springer.com/article/10.1007/s11071-024-10023-0
(verified: CI-based warnings, ROC/AUC over transition vs null models)

### 2.3 Single confidence-band / threshold crossings

Ditlevsen & Ditlevsen (2023, Nature Communications 14:4254, AMOC) compute
variance and lag-1 autocorrelation in running 50-year windows, compare
against 95% confidence bands of a baseline, and — notably — *also provide a
time-of-tipping estimator* (OU pseudo-likelihood; 2057 with CI 2034-2128).
The alarm claim is that the EWS "stay consistently above the upper limit of
the confidence interval"; this is a qualitative persistence claim on a
single realisation, not a systematic K-step protocol with controlled FPR.
https://www.nature.com/articles/s41467-023-39810-w
(verified: running-window EWS vs baseline CI; tipping-time estimate)

Moinat, Kasparian & Brunetti (2024, arXiv:2407.18727, climate networks)
flag a warning when an indicator crosses a 3-sigma interval; they explicitly
report that the lag-1 autocorrelation "produces negative spikes exceeding
3-sigma" and "three contradicting low-level signals" — observed alarm
instability that is described but not quantified as a protocol.
https://arxiv.org/html/2407.18727
(verified: 3-sigma crossing; documented AC1 instability)

### 2.4 Thresholded dynamical measures

Grziwotz et al. (2023, Science Advances 9(1):eabq4558) introduce the
dynamical eigenvalue (DEV), estimated by state-space reconstruction, with a
*theoretical* threshold |DEV| = 1 at the bifurcation. Evaluation is by trend
of |DEV| toward 1 and by application to real datasets; no alarm protocol.
https://www.science.org/doi/10.1126/sciadv.abq4558
(verified: DEV, threshold |DEV|=1, trend-based evaluation)

### 2.5 Sequential / online detection (the adjacent field)

The statistical process-control tradition formalises *sequential* alarms:
CUSUM/Shiryaev-Roberts statistics with average-run-length (ARL) and
average-detection-delay (ADD) operating characteristics (Shiryaev 1963;
Tartakovsky & Veeravalli 2008 — standard texts). In ML streaming, Kalinke
& Gavioli-Akilagun (2025, arXiv:2505.17789, NeurIPS 2025) give a fully
online kernel-MMD change detector (RFF-MMD) with minimax-optimal detection
delay and run-length calibration. These methods define alarms sequentially
and calibrate false alarms by expected run time — the closest existing
formalism to our protocol — but they are evaluated for *abrupt* changes,
not for slow bifurcation ramps, and none of them is used in the EWS
literature for critical transitions.
https://arxiv.org/abs/2505.17789
(verified: online RFF-MMD, ARL-style calibration, minimax delay)

### 2.6 Applied fields

The applied EWS literature shows the same evaluation gap, and documents
its cost:

Delecroix et al. (2023, PLOS Global Public Health) — a systematic review of
resilience indicators for infectious-disease outbreaks — find that "the AUC
was almost always used to quantify the performance", the false-positive rate
was "poorly documented (only reported in one study)", and lead time was
quantified in only ten studies. This is a verified, direct statement that
the field's dominant metric is a single-shot AUC and that FPR and lead time
are rarely reported.
https://doi.org/10.1371/journal.pgph.0002253
(verified: AUC-dominant evaluation, FPR and lead time rarely reported)

Van Beers et al. (2026, arXiv:2607.25370, "Critical slowing down for
predicting controller induced loss of control in quadrotors") apply CSD
indicators to real quadrotor loss-of-control data (91 LOC events) and
produce time-to-LOC forecasts up to 0.9 s. The applied problem and the
forecast dimension are real; the evaluation again uses detection windows
and thresholds without a persistence protocol.
https://arxiv.org/abs/2607.25370
(verified: CSD + quadrotor LOC, 91 events, time-to-LOC forecast)

Seizure-prediction studies (e.g., the prospective protocol of Iasemidis et
al.) report lead time against a false-prediction rate — the only applied
field that systematically reports the (lead, FPR) pair, and the template we
adopt in spirit.

### 2.7 Time-to-tipping forecasts (recent, already taken)

The forecast dimension — already claimed by recent work — is *not* our
contribution; we cite it and move on:

- Ditlevsen & Ditlevsen (2023): tipping-time estimate for the AMOC (2.3).
- Masuda (2026, arXiv:2602.10817, "Detecting and forecasting tipping points
  from sample variance alone", TIPMOC): fits the power-law divergence of
  the sample variance, adjudicates power-law vs linear at each step, and
  forecasts the bifurcation position when the power law is favoured; it
  explicitly aims to "avoid false positives". Conceptually identical to the
  inverse-rate (Voight) forecast structure we explored; published before us.
  https://arxiv.org/abs/2602.10817
- Li et al. (2026, arXiv:2603.14944, "Ultra-Early Prediction of Tipping
  Points: Integrating Dynamical Measures with Reservoir Computing", RCDyM):
  windowed dynamical measures (dominant eigenvalue, Floquet multiplier,
  Lyapunov exponent) extrapolated by regression for "ultra-early" tipping
  prediction.
  https://arxiv.org/abs/2603.14944

==================================================================
## 3. The gap: no persistence-aware sequential evaluation
==================================================================

Across the verified corpus above, no paper:

1. Defines an alarm as a K-step *sustained* crossing (the indicator must
   remain above threshold for K consecutive steps);
2. Reports trajectory-level *coverage* (fraction of approaching trajectories
   that ever produce a sustained alarm) at a *controlled* per-step
   false-positive rate on explicit null trajectories;
3. Documents or analyses the *dip failure mode*: an indicator that crosses
   the threshold early (a "warning") and then falls back below it and never
   returns before the transition.

The closest precedents are partial and qualitative: Ditlevsen & Ditlevsen's
"stays consistently above the confidence band" (single realisation, no FPR
control, no trajectory statistics); TIPMOC's stepwise power-law adjudication
(no coverage statistics); the spatial smoothing post-process in the 2024
high-dimensional bifurcation paper (an ad-hoc persistence hack); and the
SPC/quickest-detection literature (abrupt-change setting, not slow ramps,
not applied to EWS).

==================================================================
## 4. Why the gap matters (mechanism)
==================================================================

Windowed EWS statistics (variance, AC1, eigenvalue estimates) are noisy.
On a slowly ramping system their expected value rises smoothly, but the
per-step estimate fluctuates around it. Consequences:

- A threshold calibrated on null statistics is crossed by *transient
  fluctuations* long before the true signal rises — the "early lead" that
  single-crossing metrics report.
- An indicator can cross early and then dip below the threshold for the
  remainder of the approach — an alarm that is technically "early" and
  operationally useless (no sustained warning).
- ROC/AUC over trajectory summaries rewards exactly these transient
  crossings, because they separate signal from null at the summary level.

Our benchmark demonstrates all three effects quantitatively (section 5);
the protocol quantifies the third directly.

==================================================================
## 5. Preliminary evidence from our benchmark (ours)
==================================================================

Protocol (ours): threshold = 95th percentile of the per-step score on
validation *null* trajectories; DT = first crossing before the bifurcation
time tau; K0 = any crossing; K10 = crossing sustained for 10 consecutive
steps; coverage = fraction of test trajectories with an alarm; FPR =
per-step false-alarm rate on test nulls. Systems: fold (tau ~ 133),
Hopf (tau ~ 100), logistic map (tau ~ 67), classic generators, seeds
101/202.

Ranking inversion (the headline result, to be reproduced in the paper):

  Hopf system (tau = 100):
    method                AUC   K0 coverage/lead    K10 coverage/lead
    observer (0.03,1e-4)  0.997 0.85 / 80.1         0.25 / 46.4   <- AUC lies
    AC1 w=30              0.783 0.75 / 57.5         0.35 / 35.0
    VAR-CSD w=30          0.610 0.80 / 60.5         0.60 / 41.2
    latched-var w=20       --    0.85 / ~46         0.70 / 55.9   <- best sustained

  Fold system (tau = 133):
    observer trend        1.000 1.00 / 107.6        1.00 / 87.7   (trend channel)
    AC1 raw w=30          0.718 0.80 / 68.0         0.55 / 32.5
    SRATIO w=50           0.665 0.85 / 33.7         0.55 / 21.0
    RETRATE w=50          0.667 0.90 / 45.7         0.55 / 20.9

  Logistic (tau = 67): every method K10 coverage <= 0.15 (no sustained
  detection; theory-consistent).

The dip phenomenon (ours): on the Hopf system, 100% of observer
trajectories cross the threshold early (mean crossing ~ t=20) and then
fall below it and never return before tau — the AUC of 0.997 is a rank
separation of mostly-below-threshold scores. This is the failure mode the
field's metrics cannot see.

Note on the fold system: the fold's tau is defined by the equilibrium ramp,
so the level channel trivially achieves 1.00 sustained coverage — a sanity
check of the protocol, not a CSD result.

These numbers come from our experiment scripts (persistence battery,
fusion/latch experiments, seeds 101/202) and must be re-run through the
final repository harness before publication.

==================================================================
## 6. Positioned contribution
==================================================================

RQ1 (main): Does persistence-aware evaluation change which detectors we
think are best? Answer (ours, preliminary): yes — observer AUC 0.997 drops
to 0.25 sustained coverage; AC1 0.783 AUC drops to 0.35; headline leads of
54-100 steps collapse to 21-33 sustained. The protocol, not a new detector,
is the contribution.

RQ3 (benchmark): Subject the standard suite — lag-1 AC, windowed variance,
AR(1) likelihood (Boettiger-Hastings style), DEV-type eigenvalue estimates,
deep-EWS classifiers (Bury et al. 2021), TIPMOC-style power-law fits — to
the K-step sustained protocol on a common benchmark, reporting (coverage,
sustained lead, FPR) instead of AUC.

RQ2 (separate track, optional): the model-side question of separating
variance and autocorrelation so that a detector can measure the fold's
true CSD (autocorrelation rise) rather than its equilibrium trend. This is
independent of the protocol contribution.

==================================================================
## 7. Verified references
==================================================================

1. Dakos, V. et al. (2008). Slowing down as an early warning signal for
   abrupt climate change. PNAS 105(38):14308-14312.
   https://www.pnas.org/doi/abs/10.1073/pnas.0802430105
2. Boettiger, C. & Hastings, A. (2012). Quantifying limits to detection of
   early warning for critical transitions. J. R. Soc. Interface
   9(75):2527-2539. https://royalsocietypublishing.org/doi/10.1098/rsif.2012.0125
3. Bury, T.M. et al. (2021). Deep learning for early warning signals of
   tipping points. PNAS 118(39):e2106140118.
   https://www.pnas.org/doi/10.1073/pnas.2106140118
4. Ditlevsen, P. & Ditlevsen, S. (2023). Warning of a forthcoming collapse
   of the Atlantic meridional overturning circulation. Nat. Commun.
   14:4254. https://www.nature.com/articles/s41467-023-39810-w
5. Grziwotz, F. et al. (2023). Anticipating the occurrence and type of
   critical transitions. Sci. Adv. 9(1):eabq4558.
   https://www.science.org/doi/10.1126/sciadv.abq4558
6. (2024). Early warning signals of complex critical transitions in
   deterministic dynamics. Nonlinear Dynamics 10.1007/s11071-024-10023-0.
   https://link.springer.com/article/10.1007/s11071-024-10023-0
7. Moinat, L., Kasparian, J. & Brunetti, M. (2024). Tipping detection
   using climate networks. arXiv:2407.18727.
   https://arxiv.org/abs/2407.18727
8. Delecroix, C. et al. (2023). The potential of resilience indicators to
   anticipate infectious disease outbreaks, a systematic review and guide.
   PLOS Glob. Public Health 3(10):e0002253.
   https://doi.org/10.1371/journal.pgph.0002253
9. Kalinke, F. & Gavioli-Akilagun, S. (2025). Optimal Online Change
   Detection via Random Fourier Features. NeurIPS 2025; arXiv:2505.17789.
   https://arxiv.org/abs/2505.17789
10. Van Beers, J.J. et al. (2026). Critical slowing down for predicting
    controller induced loss of control in quadrotors. arXiv:2607.25370.
    https://arxiv.org/abs/2607.25370
11. Masuda, N. (2026). Detecting and forecasting tipping points from
    sample variance alone (TIPMOC). arXiv:2602.10817.
    https://arxiv.org/abs/2602.10817
12. Li et al. (2026). Ultra-Early Prediction of Tipping Points:
    Integrating Dynamical Measures with Reservoir Computing (RCDyM).
    arXiv:2603.14944. https://arxiv.org/abs/2603.14944
