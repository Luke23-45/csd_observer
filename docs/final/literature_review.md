# Persistence-Aware Evaluation of Early Warning Signals: Literature Review

Status: new research direction (the previous spectral-drift observer direction is closed).
Finalized 2026-08-06 after supervisor review round 2: sourcing re-verified
(Champ & Woodall 1987; NIST ARL0 = 91.75; Wheeler; Delecroix 2024 disambiguation);
ERDA links corrected; the August 2025 AMOC Author Correction is noted and its
corrected values used; section 6 now carries the closed-form Feller ARL0 for the
protocol's own (threshold, K) pairs.
Revised 2026-08-19 after external review round 3: the epidemic-EWS
consecutive-exceedance literature (refs 19-21) was added and the core claim
narrowed accordingly — the claim is no longer that sustained-crossing rules
are absent from EWS, but that no systematic persistence-aware benchmark at
matched false-alarm operating points exists; the pseudo-bifurcation result
(ref 22) is added as motivation for a hard-negative benchmark class; the
per-K matched-calibration redesign is flagged in sections 5-7 as planned
work.
Every citation below was verified against the primary source at the time of
writing (URLs provided). Negative claims ("no paper in the verified corpus
does X", section 3) are bounded by the corpus: they assert absence from the
sources surveyed here, not absence from the literature as a whole — the
round-3 additions (refs 19-22) are exactly such corpus-bounded corrections.
Our own numbers come from the project benchmark experiments (seeds 101/202,
classic generators, fixed-FPR calibration on validation nulls, K0/K10
sustained-crossing protocol); they are two-seed pilot evidence (section 5
caveat), not table-ready. Nothing in this document is claimed without
a verified source; where a claim is ours, it is labelled as such.

==================================================================
## 1. Purpose and scope
==================================================================

This review establishes the evidence base for the paper's core claim:

  Sustained-crossing alarm rules with controlled false-alarm rates are
  standard, 70-year-old practice in industrial process monitoring
  (Western Electric run rules, 1956), and consecutive-exceedance alarm
  rules have also been used in applied epidemic EWS (Southall et al.
  2022; Looker et al. 2025 — section 2.6, refs 20-21). What is missing
  is a *systematic* persistence-aware benchmark: no EWS study evaluates
  detectors on sustained alarm coverage, sustained lead time and
  false-alarm behaviour across several persistence lengths K at matched
  false-alarm operating points, and none documents the failure mode in
  which an indicator crosses a threshold early and then never stays
  above it — or shows that this failure mode mis-ranks detectors
  relative to trajectory-level AUC. This project supplies that missing
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

The applied EWS literature shows the same evaluation gap, documents its
cost — and, where consecutive-exceedance alarm rules do appear, uses them
as single-rule applied detection devices rather than as a benchmark
protocol:

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

Southall et al. (2020, PLOS Computational Biology) — a systematic
EWS treatment of *incidence* data (as opposed to prevalence) for epidemic
elimination and emergence transitions at R0 = 1, with analytical results
from a counting-process model and Gillespie simulations. The evaluation
unit is again the trajectory summary: Kendall-tau trends and ROC/AUC
against null ("Fix") simulations; no alarm event or alarm duration is
defined.
https://doi.org/10.1371/journal.pcbi.1007836
(verified: incidence-data EWS; trend + ROC/AUC evaluation; no alarm protocol)

Southall et al. (2022, medRxiv preprint) — a summary and validation of EWS
*detection methods* for epidemic critical transitions that adds an explicit
constraint: multiple consecutive time-series points must satisfy the
algorithm's conditions before a detection of an approaching transition is
flagged. This is the consecutive-points rule later adopted by Looker et al.
(2025, below). It is validated on a single simulated system (disease
elimination) with hyper-parameters selected per algorithm via ROC analysis;
it is not a multi-K benchmark.
https://doi.org/10.1101/2022.05.27.22275693
(verified: consecutive-points detection constraint; single-system validation)

Looker, Rock & Dyson (2025, PLOS Computational Biology) — EWS for UK
COVID-19 epidemic peaks on reported-case and hospitalisation data (LTLA
and NHS-region level) plus SEIR Gillespie simulations. Detection is
explicitly defined as at least *three consecutive* time points crossing
the 2-sigma threshold (the threshold is analogous to a 95% confidence
interval), following Southall et al. (2022). This is the closest precedent
to our alarm rule in the EWS literature; it is an applied detection study
of a single indicator family — there is no detector benchmark, no per-K
false-alarm calibration, and no trajectory-level coverage statistic.
https://doi.org/10.1371/journal.pcbi.1013524
(verified: three-consecutive-exceedance detection rule; applied study)

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
1987), and consecutive-exceedance alarm rules have been used in applied
epidemic EWS (section 2.6, refs 20-21). What is claimed — and what the
verified corpus supports — is the specific, checkable statement:

  No EWS study evaluates detectors on a persistence sweep at matched
  false-alarm operating points: none reports sustained alarm coverage,
  sustained lead time and false-alarm behaviour together, for several
  persistence lengths K each calibrated to the same false-alarm budget
  on explicit null trajectories.

Concretely, no paper in the verified corpus above:

1. Runs a persistence sweep across several values of K under matched
   false-alarm budgets — the epidemic work (refs 20-21) uses a
   consecutive-points constraint with a single chosen number of points
   (a per-algorithm hyper-parameter in Southall et al. 2022; fixed at
   three in Looker et al. 2025), never a multi-K sweep with per-K
   calibration;
2. Calibrates the alarm threshold separately per K so that every
   operating point carries the same per-trajectory false-alarm budget on
   explicit null trajectories;
3. Reports trajectory-level *coverage* (fraction of approaching
   trajectories that ever produce a sustained alarm) together with
   sustained lead time and the sustained-alarm false-positive rate;
4. Documents or analyses the *dip failure mode*: an indicator that crosses
   the threshold early (a "warning") and then falls back below it and never
   returns before the transition — or shows that this failure mode
   mis-ranks detectors relative to trajectory-level AUC.

The closest precedents are partial: the epidemic consecutive-point rules
(refs 20-21) — a single chosen K, applied detection studies, no coverage
statistics and no operating-point control; Ditlevsen & Ditlevsen's
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

A further reason persistence must be tested against *non-bifurcating*
changes is that canonical EWS signatures need not indicate an approaching
bifurcation at all: in stochastic non-normal systems, transient
amplification produces the same statistical signatures (rising variance
and autocorrelation) while the system remains stable and far from any
bifurcation — "pseudo-bifurcations" (Troude et al. 2026, ref 22). A
benchmark that only contrasts true transitions with stationary nulls
therefore cannot tell whether a persistent alarm tracks an approaching
bifurcation or merely a persistent non-bifurcation change; the benchmark
must include a hard-negative class of changing-but-not-bifurcating systems
(section 7, RQ3).

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

ROUND-3 NOTE (planned protocol change): the K0/K10 comparison above shares
a single per-step threshold (95th percentile of validation-null scores),
which gives K10 a vastly stricter false-alarm budget than K0 (iid per-step
FPR p^K, section 6) — the pilot ranking changes therefore confound
persistence with conservativeness. The publication protocol calibrates the
threshold separately for each K to a common per-trajectory false-alarm
target on validation nulls and reports the K-sweep at matched operating
points (sections 6-7). The pilot numbers above are retained as the
two-seed motivation only.

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

Values for the round-2 pilot's (threshold, K) pairs at the
per-step-calibrated 95th-percentile threshold (p = 0.05 per step) — the
*unmatched* operating point:

    (p, K)    ARL0 (iid)              per-step FPR_iid   P(false alarm on a
                                                          200-step null)
    (0.05, 1) 20 steps                0.05               ~ 1 (certain)
    (0.05,10) ~1.1e13 steps           ~ 9.8e-14          ~ 2e-11 (never)

The two rows are the *unmatched* operating point of the round-2 pilot, and
they make the confound explicit: the same per-step threshold gives K10 a
per-trajectory false-alarm probability of ~2e-11 over a 200-step null —
thirteen orders of magnitude stricter than K0's — so a K0-vs-K10 ranking
change at that operating point cannot be attributed to persistence alone.
The round-3 protocol (section 5 note) removes the confound by making the
false-alarm budget the calibration target: the threshold is calibrated
separately for each K to a common per-trajectory false-alarm probability on
validation nulls (e.g. 5% of null trajectories producing any sustained
alarm), with the FKG window bound below as the analytic bracket. The iid
rows then play a validation role:

- K0 (K=1): under iid nulls a single crossing is a *certain* event over a
  200-step record (P = 1 - 0.95^200 ~ 1), so the per-step-calibrated
  operating point provides no trajectory-level false-alarm control: a 5%
  per-trajectory target over the protocol's canonical window (W = 50)
  requires tightening the per-step threshold to p ~ 0.001 (inverting
  1 - (1 - p^K)^(W - K + 1)).
- K10 (K=10): under iid nulls a sustained alarm is *essentially impossible*
  at p = 0.05 (per-step FPR ~ 9.8e-14; expected false alarms per 200-step
  record ~ 2e-11), so a 5% per-trajectory target must *loosen* the K10
  threshold to p ~ 0.51 over the same W = 50 window. The iid/FKG anchor
  (1 - (1 - p^K)^(W - K + 1)) thus becomes the calibration instrument
  relating the per-step threshold to the per-trajectory budget: at each
  calibrated operating point the iid prediction is compared with the
  empirically measured per-trajectory FPR, and the empirical/iid ratio is
  reported as the serial-dependence factor (window-overlap inflation) —
  the gap between the iid baseline and the measured FPR is a measurement,
  not a caveat.

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

RQ1 (main): At matched false-alarm operating points, does persistence-aware
evaluation change the ranking of early-warning detectors for approaching
bifurcations? Pilot answer (ours, at the unmatched round-2 operating point,
section 6): yes — observer AUC 0.997 drops to 0.25 sustained coverage; AC1
0.783 AUC drops to 0.35; headline leads of 54-100 steps collapse to 21-33
sustained. The matched-operating-point version with the seed sweep is the
table-ready experiment (sections 5-6). The protocol, not a new detector, is
the contribution.

RQ1b (planned): how do the persistence length K and the threshold interact
with sustained coverage and lead time — the full K-sweep trade-off
(K = 1, 2, 3, 5, 10) at matched operating points, in search of a
persistence regime where false alarms fall sharply while useful lead time
is retained.

RQ3 (benchmark — PLANNED, not yet run): subject the field's own leading
methods to the K-step sustained protocol on a common benchmark:
  - DEV-type dominant-eigenvalue estimates (Grziwotz et al. 2023);
  - a Bury-style deep-learning classifier (Bury et al. 2021; the public
    code at github.com/ThomasMBury/deep-early-warnings-pnas is the
    reference implementation);
  - Boettiger-Hastings-style AR(1) likelihood detection (2012);
  - TIPMOC-style power-law variance fits (Masuda 2026).
Reporting (coverage, sustained lead, FPR) instead of AUC. Until DEV and a
DL classifier are run through the persistence sweep (K = 1..10) at matched
operating points on our benchmark, the ranking-inversion claim is unproven
for the methods reviewers care about most.

Benchmark conditions (PLANNED): alongside the true-transition systems
(fold, Hopf, transcritical) and stationary nulls, the benchmark adds a
*hard-negative* class — systems that change substantially but do not
undergo the target bifurcation (e.g. transient forcing, non-normal
transient amplification; Troude et al. 2026, ref 22) — so that persistence
is tested against genuine transient false alarms, not only random ones
(section 4).

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
[this document]; (2) per-K matched-operating-point calibration (protocol
redesign, section 6) with the seed sweep and CIs; (3) RQ3 baselines (DEV,
DL classifier, B&H-style likelihood, TIPMOC-style) + transcritical
generator + hard-negative class; (4) decide the real-world case study after
the core results are stable.

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
19. Southall, E., Tildesley, M.J. & Dyson, L. (2020). Prospects for detecting
    early warning signals in discrete event sequence data: Application to
    epidemiological incidence data. PLOS Computational Biology 16(9):e1007836.
    https://doi.org/10.1371/journal.pcbi.1007836
20. Southall, E., Tildesley, M.J. & Dyson, L. (2022). How early can an
    upcoming critical transition be detected? medRxiv preprint.
    https://doi.org/10.1101/2022.05.27.22275693
21. Looker, J., Rock, K.S. & Dyson, L. (2025). Identifying COVID-19 peaks
    using early warning signals. PLOS Computational Biology 21(9):e1013524.
    https://doi.org/10.1371/journal.pcbi.1013524
22. Troude, V. et al. (2026). Pseudo-bifurcations in stochastic non-normal
    systems and the limits of early-warning signals. Communications Physics
    (2026), published online 4 June 2026.
    https://doi.org/10.1038/s42005-026-02703-7
