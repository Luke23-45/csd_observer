# Literature Review — Spectral-Drift Bayesian Early Warning Observer

For use with `docs/z2/formal_defininition.md` (the Kalman-drift / spectral-gap observer).

Every reference below was retrieved and checked online at the time of writing. DOIs / arXiv IDs / exact titles are given. Anything not fully confirmed is flagged "verify at submission"; do not submit until each flagged item is checked at its publisher page.

## 1. Scope of this review

The formalisation in `formal_defininition.md` makes claims in five areas:

1. Reduction of a stochastic system near bifurcation to a scalar OU process on the "spectral gap" `c(t) = -λ(θ(t))`.
2. A conditionally-linear discrete state-space model for `(c_k, u_k)`.
3. A Bayesian / sequential observer (particle filter or EKF) emitting a "collapse probability" alarm.
4. A *provable* theoretical advantage over sliding-window heuristics (CRBs, delay bounds, "optimality").
5. Coverage of multiple bifurcation types (fold / Hopf / period-doubling).

Each area is checked below against the published literature.

## 2. Verified references (corpus)

### 2.1 Early-warning signals — foundations

- **Scheffer et al. 2009**, *Nature* 461:53–59, DOI 10.1038/nature08227. Canonical review: before a critical transition the variance and lag-1 autocorrelation of state fluctuations rise; discusses noise colour, trends, and indicator universality. *Supports* the phenomenology the observer targets; *caveat:* indicators are type-dependent, not universal.
- **Carpenter & Brock 2006**, *Ecology Letters* 9:311–318, DOI 10.1111/j.1461-0248.2005.00863.x. Variance increases as a fold is approached (lake model). *Supports* variance as a precursor.
- **Dakos et al. 2008**, *PNAS* 105:14308–14312, DOI 10.1073/pnas.0802430105. Lag-1 autocorrelation as an early-warning signal; robust with respect to detrending. *Supports* the lag-1 / spectral-gap proxy.
- **Held & Kleinen 2004**, *Geophysical Research Letters* 31, DOI 10.1029/2004GL019835 (verify exact article ID). "Degenerate fingerprinting": model-based change detection via an AR(1) coefficient, i.e. a state-estimation-style approach. *Supports*: tracking an AR(1) coefficient is a precedent, not a novelty.
- **Ditlevsen & Ditlevsen 2023**, arXiv:2304.09161 (published in *PNAS*). Early warning of AMOC collapse from the lag-1 autocorrelation of a ~150-year SST fingerprint; estimated collapse ~2057; discusses how window size, approach rate and record length trade off. *Supports* AC-gap detection on real data; confirms finite-window sensitivity.

### 2.2 Model-based / state-space detection (closest prior art)

- **Ives & Dakos 2012**, *Ecosphere* 3(6):art52, DOI 10.1890/ES11-00347.1. "Detecting dynamical changes in nonlinear time series using locally linear state-space models." **Closest existing precedent:** a time-varying AR/VAR model with a measurement-error state-space, fitted by Kalman filtering / state estimation, used to detect stability changes (rise of the dominant eigenvalue). Our novelty over it must be stated explicitly: an online posterior of the spectral gap, a direct collapse probability, and a sequential-change guarantee.
- **Doucet, Godsill & Andrieu 2000**, *Statistics and Computing* 10:197–208. Rao–Blackwellised particle filtering. *Supports the design of §3:* conditional linearisation (RB-PF) is the right tool for models that are linear given the parameter.
- **Li, Goodall & Kadirkamanathan 2004**, *IEE Proceedings – Control Theory and Applications*: Rao–Blackwellised particle filtering for a linear state-space model with an unknown parameter; outperforms a plain EKF in the reported tests (verify exact volume/pages at submission). *Supports* the recommended filter.

### 2.3 Sequential change-point / quickest-detection theory

- **Shiryaev 1963**, "On optimum methods in quickest detection problems", *Theory of Probability and its Applications* 8:22–46. Bayesian-optimal change detection: policies threshold a posterior probability of a change. **Analogy:** the "collapse probability" alarm is exactly a Shiryaev-style posterior-probability threshold — this is the correct justification to attach to §3, not a generic "our filter is optimal".
- **Pollak 1985**, *Annals of Statistics* 13:206–227. Optimality and asymptotic properties of sequential change-detection procedures.
- **Tartakovsky & Veeravalli**, "General Asymptotic Bayesian Theory of Quickest Change Detection" (SIAM/sequential-analysis chapters; also the Tartakovsky–Nikiforov–Basseville book). Shiryaev-type procedures are asymptotically Bayes-optimal for broad non-i.i.d. models. Actual, citable statement: "asymptotically Bayes-optimal *within the assumed generative model*".
- **Page 1954 (CUSUM)**, *Biometrika*; **Roberts 1966 (SR)**, *Technometrics*. Frequentist procedures are generally *not* optimal under the Bayesian criterion — cite to bound the "optimal" claim.

### 2.4 Deep-learning and spectral early-warning

- **Bury et al. 2021**, *PNAS* 118(39):e2106140118, DOI 10.1073/pnas.2106140118. Deep-learning early-warning signal trained on many models; reduced false alarms vs. generic indicators; *key for §5:* the network acts on the dynamical *class* — a single ubiquitous indicator is insufficient across fold / Hopf / other types.

### 2.5 Hopf / period-doubling — where the scalar-OU picture breaks

- **Bury et al. 2020**, spectral-indicator paper, *J. R. Soc. Interface* 17 (verify exact volume/ID at submission). Near Hopf the classical lag-1 pattern is partially different (oscillatory eigenvalue pair); spectral methods can distinguish fold vs Hopf.
- Recent literature on early warnings for period-doubling shows the dominant eigenvalue multiplier approaches **−1** (mirror / alternating coordinate), not 0; the classical lag-1/OU picture is a misspecification for period-doubling. Verify the specific source (recent arXiv / Nonlinear-Dynamics items) before citing.
- **Boxler 1989**, "A stochastic version of centre manifold theory", *Probab. Theory Related Fields* (verify exact pages/DOI at submission). The stochastic centre-manifold theorem is a published, standard tool — the reduction used in §1.1 is legitimate.

### 2.6 False positives and calibration (prosecutor's fallacy)

- **Boettiger & Hastings 2012**, *Proc. R. Soc. B* (verify DOI at submission). Many "null" time series display EWS by chance; sensitivity and specificity must be calibrated to a target false-positive rate. *Consequence:* the collapse probability needs a calibrated threshold (Youden / fixed FPR) against the same null machinery the repo already has.

## 3. Section-by-section findings on `formal_defininition.md`

### 3.1 §1 (reduction to scalar OU) — correct; keep, with a scope caveat

This is a genuine generic-fold result:
- a unique Jacobian eigenvalue crosses zero at theta_c; the others are strictly negative;
- the stochastic centre-manifold reduction is standard (Boxler 1989; Arnold);
- in the sub-critical regime the quadratic term is dominated by the linear one, so `ż = λz + a z²` becomes OU with `c(t) = -λ(θ(t))`.

**Scope caveat (must stay):** this is a fold result only. For **Hopf** the leading eigen-pair is complex, so the lag-1 story differs (2.5). For **period-doubling** the leading multiplier → −1 (a mirror coordinate), so "c → 0 with real +1" is the wrong limit. State this and instantiate only the fold case in the paper.

### 3.2 §2 (state-space model) — mathematically right, except one hard bug

- The exact OU transition `u_{k+1} = φ(c_k)u_k + ε_k`, `ε_k ~ N(0, σ²/(2c)(1−φ²))` is correct.
- **Bug (must fix):** the variance `σ²/(2c)(1−φ(c)²)` is undefined/negative for `c < 0`. Because `c_{k+1} = c_k + η_k` is an unbounded random walk, simulation will hit `c < 0` and the covariance breaks. Fix: `c_{k+1} = max(c_min, c_k + η_k)` (rectified / reflecting drift) or a positivity-preserving parameterisation (log-c), with prior mass on `c ≥ c_min > 0`.
- Linear observation `y_k = u_k + ν_k`: fine.
- The "ū = 0 / known equilibrium" preprocessing must be implemented and justified in the paper (near the fold it is the data mean; define the preprocessing module).

### 3.3 §3 (observer) — concept correct; implementation wording must change

- `P(c_k < δ | y_{1:k})` is a Shiryaev-style posterior-probability statistic — correct *within the model* (see 2.3).
- The **EKF Jacobian is correct** (`F = [[φ, −Δt·u·φ], [0,1]]`).
- **Wording change:** near `c → 0` the EKF degenerates (φ(c) ≈ 1); the robust choice from the literature is a Rao–Blackwellised particle filter over `c` with a Kalman step over `u`; EKF should be presented as the cheap linearised baseline, not "the optimal implementation" as written.
- In `Q_k`, the state-dependent Q11 should be evaluated at the *predicted* mean `c_{k+1|k}` (standard EKF practice); say so.

### 3.4 §4 (comparison / theorem) — reject as written

The §4 claims are not supported and are partly dimensionally wrong:

1. "CRB for a W-window is O(W^{−1/2}); the Bayes filter achieves O(√Q_drift)" — the CRB is a variance bound, not a delay bound; O(√Q_drift) is dimensionally mismatched for an "error" reading of the state. Not in the cited literature; will fail review.
2. "Delay scales like O(log(1/ε))" — mixes quickest-detection bounds with drift-rate assumptions; there is no citable theorem in this form. Replace with the real first-order bound of the surveillance stream: delay ≈ **|log α| / KL(f1 ∥ f0)**, where f0/f1 are the no-change / change densities of the observed gap statistic and KL is their divergence, for desired false-alarm level α (Shiryaev/Lorden/Pollak tradition; see 2.3).
3. The §4 theorem ("the window delay exceeds the Bayes filter by O(W), arbitrarily large") is **false in the limit**. What is defensible and documentation-ready:
   - a rolling window of length W has an intrinsic lag/truncation of about **W/2** when estimating the *current* coefficient, and
   - any filter, including the Bayes filter, has a residual delay set by the evidence-accumulation rate, of order `log(1/FAR)/KL`.

   Conclusion to use: "Within the assumed generative model, the observer's delay is governed by evidence accumulation, not by W; a windowed AR(1) additionally has an ≈W/2 lag-bias." No "arbitrarily larger O(W)" claim.

### 3.5 §5 (parameters and calibration)

The parameters are unspecified: `σ_u, R, Q_drift, c_0, δ`, and the prior on `c_1`. As `c → 0`, identifiability of `(σ_u, R, c)` is genuinely weak (although the variance blow-up at the crossing partially restores the signal). The paper must give a parameter table and a sensitivity study, and calibrate the threshold (fixed-FPR / Youden) against the null series (2.6).

## 4. Professor-ready summary

> Near a **generic fold**, the dominant mode reduces to a scalar OU with mean-reversion rate `c(t) = −λ(θ(t))` (stochastic centre-manifold, a standard tool). We track the gap c with a conditionally-linear state-space model using a Rao–Blackwellised particle filter and alarm when the posterior `P(c_k < δ | y_{1:k})` crosses a threshold — a Shiryaev-type sequential-decision statistic. Within the assumed generative model this is Bayes-optimal quickest detection; versus windowed summaries it uses the data more efficiently (precedent: Ives & Dakos 2012; Ditlevsen & Ditlevsen 2023), and, calibrated against null series, it is less prone to the "prosecutor's fallacy" (Boettiger & Hastings 2012). The earlier "O(√Q_drift)" / "arbitrary O(W)" superiority claims are replaced by the honest statement: windowed AR(1) carries an ≈W/2 lag-bias; the filter's residual delay is set by `log(1/FAR)/KL`. The OU reduction is fold-only; Hopf (complex pair) and period-doubling (multiplier → −1) need separate reductions left for future work.

This is the ground-truth basis the formalisation must match before claims of optimality/advantage are written in the paper.

## 5. Required changes to `formal_defininition.md` (priority order)

1. §2: constrain `c` so the OU variance never becomes negative (`c_min`).
2. §3: RB particle filter primary; EKF as baseline; eval Q11 at the predicted mean.
3. §4: drop the O(√Q)/"O(W) better" claims; adopt the lag-bias + `log(1/FAR)/KL` statement.
4. §1/§5: label the reduction "fold-type" and add a short Hopf / period-doubling caveat.
5. Add a parameter & prior block and a null-calibration/FPR step (Youden or fixed-FPR).
6. Add a Related-Work paragraph positioning against **Ives & Dakos 2012** and **Ditlevsen & Ditlevsen 2023** (fixed-λ AR fitting vs online filtering).
7. Remove the draft-style first lines and "I-..." phrasing so the document reads as a clean paper section.

## 6. Reference list (final; re-verify flagged items at submission)

- [schee-09] Scheffer, M., et al. (2009). Early warning signals for critical transitions. *Nature* 461:53–59. DOI 10.1038/nature08227.
- [carp-06] Carpenter, S. R. & Brock, W. A. (2006). Rising variance: a leading indicator of ecological transition. *Ecology Letters* 9:311–318.
- [dakos-08] Dakos, V., et al. (2008). Slowness as an early warning for abrupt climate change. *PNAS* 105:14308–14312.
- [held-04] Held, H. & Kleinen, T. (2004). Detection of climate system changes through degenerate fingerprinting. *Geophys. Res. Lett.* (verify article ID).
- [ditl-23] Ditlevsen, P. & Ditlevsen, S. (2023). Warning of a forthcoming collapse of the Atlantic Ocean circulation. arXiv:2304.09161 (published in *PNAS*).
- [ives-12] Ives, A. R. & Dakos, V. (2012). Detecting dynamic changes using locally linear state-space models. *Ecosphere* 3(6):art52. DOI 10.1890/ES11-00347.1.
- [doucet-00] Doucet, A., Godsill, S. J. & Andrieu, C.M. (2000). On sequential Monte Carlo sampling for Bayesian filtering. *Statistics and Computing* 10:197–208.
- [li-04] Li, P., Goodall, R. & Kadirkamanathan, V. (2004). Estimation of parameters in a linear state-space model using a Rao–Blackwellised particle filter. *IEE Proceedings – Control Theory and Applications* (verify volume/pages at submission).
- [shir-63] Shiryaev, A. N. (1963). On optimum methods in quickest detection. *Th. Probab. Appl.* 8:22–46.
- [pollak-85] Pollak, M. (1985). Optimal detection of a change in distribution. *Annals of Statistics* 13:206–227.
- [tarta-...] Tartakovsky, Nikiforov & Basseville; Tartakovsky & Veeravalli (general asymptotic Bayesian quickest change online/sequential books).
- [page-54] Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41:100–115. Roberts, S. W. (1966). A comparison of some control chart procedures. *Technometrics* 8:411–430 (verify pages at submission).
- [bury-21] Bury, T., et al. (2021). Deep learning for early-warning signals of tipping points. *PNAS* 118(39):e2106140118. DOI 10.1073/pnas.2106140118.
- [bury-20] Bury, T., et al. (2020). Spectral indicators for early warning. *J. R. Soc. Interface* 17.
- [box-89] Boxler, P. (1989). A stochastic version of center manifold theory. *Probability Theory and Related Fields* 83:509–545 (verify exact pagination/DOI at submission).
- [boe-2012] Boettiger, C. & Hastings, A. (2012). Early warning signals and the prosecutor's fallacy. *Proc. R. Soc. B* (verify).

Each item ending with "(verify)" or "(verify …)" must be confirmed on the publisher page before the marks, the paper, or any citation is sent out.