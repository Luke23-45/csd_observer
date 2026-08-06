# Persistence-Aware Evaluation of Early Warning Signals: Literature Review

Status: new research direction (the previous spectral-drift observer direction is closed).
Finalized 2026-08-06 after supervisor review round 2: sourcing re-verified
(Champ & Woodall 1987; NIST ARL0 = 91.75; Wheeler; Delecroix 2024 disambiguation);
ERDA links corrected; the August 2025 AMOC Author Correction is noted and its
corrected values used; section 6 now carries the closed-form Feller ARL0 for the
protocol's own (threshold, K) pairs.
Every citation below was verified against the primary source at the time of writing
(URLs provided). Our own numbers come from the project benchmark experiments
(seeds 101/202, classic generators, fixed-FPR calibration on validation nulls,
K0/K10 sustained-crossing protocol). Nothing in this document is claimed without
a verified source; where a claim is ours, it is labelled as such.

==================================================================
## 1. Purpose and scope
==================================================================

This review establishes the evidence base for the paper's core claim:

  Sustained-crossing alarm rules with controlled false-alarm rates are
  standard, 70-year-old practice in industrial process monitoring
  (Western Electric run rules, 1956), but they have not been imported
  into CSD/EWS benchmarking for bifurcation-type transitions. The EWS
  field evaluates detectors with single-shot, per-trajectory metrics
  (trend statistics, ROC/AUC, classification accuracy, single
  confidence-band crossings); no EWS benchmark defines an alarm as a
  K-step sustained event with trajectory-level coverage under a
  calibrated per-step false-positive rate, and none documents the
  failure mode in which an indicator crosses a threshold early and then
  never stays above it. This project supplies that missing evaluation
  protocol and demonstrates that its adoption changes the ranking of
  existing methods.

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
baseline significance — dominates the applied literature (see 2.6).

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
time-of-tipping estimator* (Strang-splitting pseudo-likelihood; corrected
version: 2065, CI 2037-2109 — see the corrigendum note in section 8,
ref. 4; the original 2057 / CI 2034-2128 values were revised by the 2025
Author Correction and must not be quoted).
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

### 2.5 Industrial process monitoring: run rules and sequential alarms

This is the provenance our protocol must acknowledge, and it is decisive
for the framing: *sustained-crossing alarm rules with calibrated false-alarm
rates are not new in monitoring generally.*

The Western Electric Company's *Statistical Quality Control Handbook*
(1956) codified the four Western Electric run rules for control charts:
(1) one point beyond the 3-sigma limit; (2) two of three consecutive points
beyond the 2-sigma limit; (3) four of five consecutive points beyond the
1-sigma limit; (4) eight consecutive points on one side of the centre line.
Each rule carries a known theoretical false-alarm probability under iid
normal observations (roughly 0.3% per point for the individual rules; NIST
Engineering Statistics Handbook), and the combined rule set has an exactly
computable in-control average run length: ARL0 = 91.75 points, obtained by
Markov-chain methods by Champ & Woodall (1987, Technometrics 29(4):393-399)
and confirmed in the NIST handbook ("adding the WECO rules increases the
frequency of false alarms to about once in every 91.75 points, on the
average"). Wheeler (SPC Press) gives the same ARL analysis in accessible
form. Two precision points matter here. First, 91.75 is the ARL of the
*four-rule WECO combination as a whole* (four different zone thresholds,
OR'd together) — it is not the ARL of any single run rule and must not be
quoted as the false-alarm rate of our protocol's K10 rule (section 6).
Second, our K10 rule is simpler than the general multi-zone case Champ &
Woodall solve: it is a single threshold with K consecutive exceedances —
the classical waiting time for a run of k successes, which has a
closed-form mean older than the Markov-chain treatment (Feller 1968,
Vol. I, XIII.7): ARL0 = (1 - p^K) / (p^K (1 - p)) for iid per-step
exceedance probability p. The WE rule set is the K-of-N run-rule ancestor
of the K0/K10 protocol in this project: alarm = a run of points crossing a
threshold, with a calibrated false-alarm rate, and with exact run-length
theory available for analytical validation.

The complementary sequential formalism is the CUSUM/Shiryaev-Roberts
family with average-run-length (ARL) and average-detection-delay (ADD)
operating characteristics (Shiryaev 1963; Tartakovsky & Veeravalli 2008 —
standard texts). In ML streaming, Kalinke & Gavioli-Akilagun (2025,
arXiv:2505.17789, NeurIPS 2025) give a fully online kernel-MMD change
detector (RFF-MMD) with minimax-optimal detection delay and run-length
calibration. These methods define alarms sequentially and calibrate false
alarms by expected run time — the closest existing formalism to our
protocol — but they are evaluated for *abrupt* changes, not for slow
bifurcation ramps, and none of them is used in the EWS literature for
critical transitions.
https://arxiv.org/abs/2505.17789
(verified: online RFF-MMD, ARL-style calibration, minimax delay)

References for this subsection:
- Western Electric Company (1956). Statistical Quality Control Handbook.
  Indianapolis: Western Electric Co. OCLC 33858387.
- Champ, C.W. & Woodall, W.H. (1987). Exact Results for Shewhart Control
  Charts With Supplementary Runs Rules. Technometrics 29(4):393-399.
  https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488266
  (paper PDF: https://www.stat.cmu.edu/technometrics/80-89/VOL-29-04/v2904393.pdf)
- NIST/SEMATECH e-Handbook of Statistical Methods, section 6.3.2
  ("WECO rules increase false alarms... once in every 91.75 points,
  on the average (see Champ and Woodall, 1987)").
  https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm
- Wheeler, D.J. Contra Two Sigma. SPC Press.
  https://spcpress.com/pdf/DJW255.pdf

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

Disambiguation: a later paper by the same group — Delecroix, van Nes,
Scheffer & van de Leemput, "Monitoring resilience in bursts" (PNAS
121(31):e2407148121, 2024) — is *not* related to alarm persistence. It is
a sampling-design study comparing short high-resolution monitoring bursts
against continuous time series for estimating resilience change. We do not
cite it as related work on the protocol; it is noted here only to preempt
the association by title.
https://www.pnas.org/doi/10.1073/pnas.2407148121
(verified: sampling-design study, bursts vs continuous series)

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
## 3. The gap (reframed)
==================================================================

It is NOT claimed that sustained-crossing alarms with controlled FPR are
new: they are the Western Electric run rules (1956), standard industrial
practice for seven decades, with exact run-length theory (Champ & Woodall
1987). What is claimed — and what the verified corpus supports — is the
specific, checkable statement:

  Sustained-crossing, controlled-FPR alarm rules have not been imported
  into CSD/EWS benchmarking for bifurcation-type transitions.

Concretely, no paper in the verified corpus above:

1. Defines the alarm of an EWS detector as a K-step *sustained* crossing
   (the run-rule framing) inside a benchmarking protocol;
2. Reports trajectory-level *coverage* (fraction of approaching
   trajectories that ever produce a sustained alarm) at a *controlled*
   per-step false-positive rate on explicit null trajectories;
3. Documents or analyses the *dip failure mode*: an indicator that crosses
   the threshold early (a "warning") and then falls back below it and never
   returns before the transition.

The closest precedents are partial and qualitative: Ditlevsen & Ditlevsen's
"stays consistently above the confidence band" (single realisation, no FPR
control, no trajectory statistics); TIPMOC's stepwise power-law adjudication
(no coverage statistics); the spatial smoothing post-process in the 2024
high-dimensional bifurcation paper (an ad-hoc persistence hack); and the
SPC/quickest-detection literature (abrupt-change setting, not slow ramps,
not applied to EWS benchmarking).

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
## 5. Preliminary evidence from our benchmark (ours — pilot stage)
==================================================================

Protocol (ours): threshold = 95th percentile of the per-step score on
validation *null* trajectories; DT = first crossing before the bifurcation
time tau; K0 = any crossing; K10 = crossing sustained for 10 consecutive
steps; coverage = fraction of test trajectories with an alarm; FPR =
per-step false-alarm rate on test nulls. Systems: fold (tau ~ 133),
Hopf (tau ~ 100), logistic map (tau ~ 67), classic generators, seeds
101/202.

CAVEAT: these numbers are a TWO-SEED PILOT. They establish the phenomenon;
they are not yet table-ready. The publication version requires a seed sweep
with confidence intervals on every quantity.

Ranking inversion (the pilot result):

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
final repository harness with a proper seed sweep before publication.

==================================================================
## 6. Validation step: run-rule theory as an analytical check
==================================================================

The run-rule provenance gives a validation instrument, not just a citation.
Our alarm rules are single-threshold K-consecutive-exceedances tests —
simpler than the multi-zone WECO combination — so their in-control
run-length properties have a classical closed form, the mean waiting time
for a run of k successes (Feller 1968, Vol. I, XIII.7). For iid per-step
exceedance probability p and K consecutive exceedances:

    ARL0 = (1 - p^K) / (p^K (1 - p)),     per-step FPR_iid = p^K.

(NB: 91.75 is NOT this quantity for K10. It is the ARL of the four-rule
WECO combination OR'd together (section 2.5) and must not be quoted as our
rule's false-alarm rate. Champ & Woodall's Markov-chain machinery is needed
only for such multi-rule combinations; a single-threshold run rule needs no
chain.)

Values for the protocol's actual (threshold, K) pairs at the calibrated
95th-percentile threshold (p = 0.05 per step):

    (p, K)    ARL0 (iid)              per-step FPR_iid   P(false alarm on a
                                                          200-step null)
    (0.05, 1) 20 steps                0.05               ~ 1 (certain)
    (0.05,10) ~1.1e13 steps           ~ 9.8e-14          ~ 2e-11 (never)

The two rows make opposite points, and both do real work:

- K0 (K=1): under iid nulls a single crossing is a *certain* event over a
  200-step record (P = 1 - 0.95^200 ~ 1), which is exactly why the protocol
  calibrates per-step FPR and why K0 coverage without per-step control is
  meaningless. The iid per-step FPR equals the calibrated 0.05 by
  construction, so this row is a calibration consistency check (factor ~ 1).
- K10 (K=10): under iid nulls a sustained alarm is *essentially impossible*
  (per-step FPR ~ 9.8e-14; expected false alarms per 200-step record
  ~ 2e-11). The pilot nulls show per-step crossing FPRs of 0.05-0.1 for the
  persistence detectors (section 5). Any empirical *sustained*-alarm rate
  above the iid prediction is therefore, by definition, serial dependence
  inherited from window overlap — the gap between the iid baseline and the
  measured FPR is a measurement, not a caveat.

The validation step (run with the seed sweep) therefore reports, per
(threshold, K) pair used in the protocol:

  - ARL0 and per-step FPR under the iid assumption (closed form above);
  - the empirically measured per-step and per-trajectory FPR on test nulls;
  - the ratio empirical / iid, reported as the serial-dependence factor
    (equivalently: the effective number of independent samples per window
    implied by the FPR inflation).

Where the empirical rate exceeds the iid prediction, the comparison either
justifies the empirical calibration as the honest operating point or forces
a correction (e.g., block-based calibration or effective-sample-size
adjustment). As an optional cross-check, Champ & Woodall's Markov-chain
method reproduces the closed form and extends it to the multi-rule WECO
combination (ARL0 = 91.75), tying the analytical check back to the
industrial literature.

This turns the SPC lineage from a liability (a reviewer will know the
rules) into a methodological asset: the protocol inherits a checkable
theory, and the paper reports both the empirical FPR and its analytical
counterpart.

==================================================================
## 7. Positioned contribution and remaining work
==================================================================

RQ1 (main): Does persistence-aware evaluation change which detectors we
think are best? Pilot answer (ours): yes — observer AUC 0.997 drops to
0.25 sustained coverage; AC1 0.783 AUC drops to 0.35; headline leads of
54-100 steps collapse to 21-33 sustained. The protocol, not a new detector,
is the contribution.

RQ3 (benchmark — PLANNED, not yet run): subject the field's own leading
methods to the K-step sustained protocol on a common benchmark:
  - DEV-type dominant-eigenvalue estimates (Grziwotz et al. 2023);
  - a Bury-style deep-learning classifier (Bury et al. 2021; the public
    code at github.com/ThomasMBury/deep-early-warnings-pnas is the
    reference implementation);
  - Boettiger-Hastings-style AR(1) likelihood detection (2012);
  - TIPMOC-style power-law variance fits (Masuda 2026).
Reporting (coverage, sustained lead, FPR) instead of AUC. Until DEV and a
DL classifier are run through K0/K10 on our benchmark, the ranking-
inversion claim is unproven for the methods reviewers care about most.

System coverage (PLANNED): Bury et al. evaluate across fold, Hopf and
transcritical bifurcations, and RCDyM across equilibria, cycles and chaotic
dynamics. Our fold/Hopf/logistic trio is not directly comparable to their
reported scores; a transcritical generator must be added to the benchmark
(at minimum) for the related-work table to be matched on conditions.

Seeds (PLANNED): a real seed sweep with confidence intervals on all
quantities in section 5 (the two-seed pilot is not table-ready).

RQ2 (separate track, optional): the model-side question of separating
variance and autocorrelation so that a detector can measure the fold's
true CSD (autocorrelation rise) rather than its equilibrium trend. This is
independent of the protocol contribution.

Real-world case study (FOLLOW-UP, not blocking): the strongest form of an
evaluation-protocol paper has at least one applied re-analysis. Two
candidates are publicly available: the AMOC fingerprint series of
Ditlevsen & Ditlevsen (2023, data and code archived at ERDA
https://erda.ku.dk/archives/cb78329f209d8ff2b4dd810abe4780ae/published-archive.html),
and the quadrotor LOC data of Van Beers et al. (2026, 91 events). Either is a
strong addition to the methods paper or a fast follow-up paper; it should
not block submission.

Corrigendum note (AMOC dataset): the AMOC paper carries an Author Correction
of 21 August 2025 (Nat. Commun. 16:7794, doi:10.1038/s41467-025-63201-y)
fixing a Strang-splitting MLE coding error (the flow was evaluated at
t(i-1) instead of t(i)); the tipping-time estimate changes by 8 years and
Figs. 5-7, Table 1 and the text were revised. Any case study must use the
corrected version of the paper, its corrected estimates (2065, CI
2037-2109) and the corrected code (archived separately at ERDA
https://erda.ku.dk/archives/afce4e1c3ac6f0db61c27cb45c2e9b14/published-archive.html),
citing the correction.

Order of operations (as advised): (1) run-rule framing + disambiguation
[this document]; (2) seed sweep with CIs; (3) RQ3 baselines (DEV, DL
classifier, B&H-style likelihood, TIPMOC-style) + transcritical generator;
(4) decide the real-world case study after the core results are stable.

==================================================================
## 8. Verified references
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
   Data/code: https://erda.ku.dk/archives/cb78329f209d8ff2b4dd810abe4780ae/published-archive.html
   Corrigendum (21 Aug 2025): Author Correction, Nat. Commun. 16:7794,
   https://doi.org/10.1038/s41467-025-63201-y (Strang-splitting MLE error;
   tipping-time estimate revised by 8 years). Corrected code:
   https://erda.ku.dk/archives/afce4e1c3ac6f0db61c27cb45c2e9b14/published-archive.html
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
9. Delecroix, C., van Nes, E.H., Scheffer, M., van de Leemput, I.A. (2024).
   Monitoring resilience in bursts. PNAS 121(31):e2407148121. [cited only
   for disambiguation, section 2.6]
   https://www.pnas.org/doi/10.1073/pnas.2407148121
10. Kalinke, F. & Gavioli-Akilagun, S. (2025). Optimal Online Change
    Detection via Random Fourier Features. NeurIPS 2025; arXiv:2505.17789.
    https://arxiv.org/abs/2505.17789
11. Van Beers, J.J. et al. (2026). Critical slowing down for predicting
    controller induced loss of control in quadrotors. arXiv:2607.25370.
    https://arxiv.org/abs/2607.25370
12. Masuda, N. (2026). Detecting and forecasting tipping points from
    sample variance alone (TIPMOC). arXiv:2602.10817.
    https://arxiv.org/abs/2602.10817
13. Li et al. (2026). Ultra-Early Prediction of Tipping Points:
    Integrating Dynamical Measures with Reservoir Computing (RCDyM).
    arXiv:2603.14944. https://arxiv.org/abs/2603.14944
14. Western Electric Company (1956). Statistical Quality Control Handbook.
    Indianapolis: Western Electric Co. OCLC 33858387.
    https://search.worldcat.org/title/33858387
15. Champ, C.W. & Woodall, W.H. (1987). Exact Results for Shewhart Control
    Charts With Supplementary Runs Rules. Technometrics 29(4):393-399.
    https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488266
16. NIST/SEMATECH e-Handbook of Statistical Methods, 6.3.2 (WECO rules,
    ARL 91.75). https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm
17. Wheeler, D.J. Contra Two Sigma. SPC Press.
    https://spcpress.com/pdf/DJW255.pdf
18. Feller, W. (1968). An Introduction to Probability Theory and Its
    Applications, Vol. I, 3rd ed. Wiley. (Chapter XIII, section 7:
    waiting times for runs of successes — source of the closed-form
    ARL0 used in section 6.)
