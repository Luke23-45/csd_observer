# Spectral-Drift Bayesian Early Warning Observer — Formal Definition

**Scope.** This document formalises the spectral-drift Bayesian early warning observer for **fold (saddle-node) bifurcations**: the setting in which a single real eigenvalue of the linearisation crosses zero at the transition. Hopf and period-doubling bifurcations require separate reductions and are discussed in §6. The development is linear-level: the dominant mode is modelled as a scalar Ornstein–Uhlenbeck (OU) process with a slowly drifting mean-reversion rate; we derive the exact discrete-time state-space model, define the observer (a Rao–Blackwellised particle filter with a conditional Kalman filter over the mode), and give the alarm rule as a Shiryaev-type posterior-probability threshold.

---

## 1. Reduction to a Scalar OU Process (Fold Regime)

### 1.1 Centre-Manifold Reduction

Assume the deterministic system \(\dot{x} = f(x,\theta)\) has a stable equilibrium \(x^*(\theta)\) for \(\theta < \theta_c\), undergoing a generic **fold** (saddle-node) bifurcation at \(\theta_c\). By the implicit function theorem, for \(\theta\) near \(\theta_c\) the Jacobian \(J(\theta) = \nabla_x f(x^*(\theta),\theta)\) has a simple real eigenvalue \(\lambda(\theta)\) that crosses zero at \(\theta_c\), while all other eigenvalues have strictly negative real parts uniformly bounded away from zero.

Let \(e(\theta)\) and \(e^*(\theta)\) be the right and left eigenvectors of \(J(\theta)\) corresponding to \(\lambda(\theta)\), normalised so that \(e^* \cdot e = 1\). The **centre-manifold theorem** guarantees a one-dimensional attracting invariant manifold tangent to \(e(\theta)\) at the equilibrium. The dynamics on this manifold, up to cubic order, are the scalar ODE

\[
\dot{z} = \lambda(\theta) z + a(\theta) z^2 + \text{higher order},
\]

with \(a(\theta) \neq 0\) for a generic fold. The **stochastic centre-manifold reduction** (Boxler 1989; normal-form treatment of Arnold and collaborators) yields the scalar SDE for the dominant mode \(z_t\):

\[
dz_t = \bigl( \lambda(\theta(t)) z_t + a z_t^2 \bigr) dt + \sigma_{\text{eff}}\, dW_t,
\]

where \(\sigma_{\text{eff}}^2 = e^* \cdot (\sigma \sigma^\top) e\) is the projection of the noise onto the critical mode.

At the early warning stage the system is in the **sub-critical** regime, where the quadratic term is negligible compared with the linear stability. Thus the **Ornstein–Uhlenbeck approximation**

\[
dz_t = -c(t)\, z_t dt + \sigma_{\text{eff}}\, dW_t, \qquad c(t) := -\lambda(\theta(t)) > 0,
\]

holds with high accuracy, and the spectral gap \(c(t)\) is exactly the negative of the dominant eigenvalue. This is a rigorous, local, linear-level justification for the scalar OU model. The Lyapunov function \(V\) can be taken simply as \(z^2\) (or any monotone quadratic function of \(z\)).

> **Scope caveat.** The reduction above is specific to the **fold class**. For **Hopf** bifurcations the leading eigenpair is complex conjugate; there is no real eigenvalue crossing zero, and the lag-1 autocorrelation behaviour differs from the fold pattern (Bury et al. 2021; spectral analysis of Bury et al. 2020). For **period-doubling**, the dominant (discrete-time) multiplier tends to **−1** — a mirror/alternating coordinate — not to +1, so the "c → 0 with real +1" limit is the wrong limit. The framework in this document therefore instantiates the fold case only; the other two classes are deferred (§6).

### 1.2 Continuous-Time Generative Model

We adopt as the canonical model for the **system stability indicator** a real-valued process \(u_t\) (the dominant mode amplitude) satisfying

\[
du_t = -c(t) \bigl( u_t - \bar{u} \bigr) dt + \sigma_u \, dW_t,
\]

where \(\bar{u} = 0\) for the centred mode (the equilibrium value, estimated by preprocessing, e.g. subtraction of a running mean; near the fold the equilibrium is the data mean). **When generating synthetic data for validation, the observations must be centred by subtracting the equilibrium before the model is applied; otherwise the \(\bar{u} = 0\) assumption is misspecified.** The spectral gap \(c(t)\) is a slowly varying function, **bounded below by \(c_{\min} > 0\)**, that tends to \(c_{\min}\) as the system approaches the critical threshold (in the continuous-time picture \(c(t) \downarrow 0\); in the discrete implementation we clip at \(c_{\min}\), §2). We additionally approximate \(c(t)\) as piecewise constant on the sampling intervals \([k\Delta t, (k+1)\Delta t]\).

---

## 2. Discrete-Time State-Space Model

Sampling \(u_t\) at \(t_k = k\Delta t\) gives \(u_k\). The exact transition density of the OU process is

\[
u_{k+1} = e^{-c_k \Delta t} u_k + \bar{u}(1 - e^{-c_k \Delta t}) + \varepsilon_k,
\]
\[
\varepsilon_k \sim \mathcal{N}\!\left(0, \frac{\sigma_u^2}{2c_k}\bigl(1 - e^{-2c_k \Delta t}\bigr)\right),
\]

where \(c(t)\) is treated as constant over the interval \([k\Delta t, (k+1)\Delta t]\) (legitimate for a slowly varying parameter). With \(\bar{u}=0\) (centred, preprocessed mode) the state-space model is:

**State equation (latent spectral gap drift).**
\[
c_{k+1} = \max\bigl(c_{\min}, \; c_k + \eta_k\bigr), \qquad \eta_k \sim \mathcal{N}(0, Q_{\text{drift}}), \quad c_{\min} > 0.
\]

The lower bound is **mandatory**: the observation variance \(\sigma_u^2/(2c_k)(1 - e^{-2c_k\Delta t})\) is undefined or negative for \(c_k < 0\), which would collapse the filter. An equivalent positivity-preserving reparameterisation is \(c_k = \exp(\theta_k)\) with \(\theta_k\) drifting. The prior is a truncated distribution over \(c_1\) with support \([c_{\min}, \infty)\).

**Observation equation (dominant mode amplitude).**
\[
u_{k+1} = \phi(c_k) u_k + \varepsilon_k, \quad \phi(c) = e^{-c \Delta t},
\]
\[
\varepsilon_k \sim \mathcal{N}\!\left(0, \frac{\sigma_u^2}{2c_k}\bigl(1 - \phi(c_k)^2\bigr)\right),
\]
\[
y_k = u_k + \nu_k, \qquad \nu_k \sim \mathcal{N}(0, R).
\]

The model is **conditionally linear**: given the parameter path \(c_{1:k}\), the mode \(u_k\) is linear Gaussian. This is exactly the structure for which Rao–Blackwellised particle filtering is the appropriate estimator (Doucet, Godsill & Andrieu 2000; Li, Goodall & Kadirkamanathan 2004).

---

## 3. The Spectral-Drift Bayesian Observer

**Definition 1 (Spectral Early Warning Observer).** Given the state-space model of §2, the observer is a causal filter that recursively computes the posterior distribution \(p(c_k \mid y_{1:k})\) and outputs the **collapse probability**

\[
p_{\text{collapse},k} = \Pr(c_k < \delta \mid y_{1:k}) \quad \text{for a pre-specified threshold } \delta > 0.
\]

**Reference implementation: Rao–Blackwellised particle filter (RB-PF).** Particles \(\{c_k^{(i)}\}\) are propagated through the state equation, and for each particle the conditional Kalman filter over \(u_k\) is updated exactly; the particle weights use the predictive likelihood of the observation. The RB-PF is preferred over the plain EKF because, as \(c \to 0\), \(\phi(c) \to 1\) and the EKF linearisation of the near-unit-root mode degenerates; the RB-PF does not rely on this linearisation.

**Baseline implementation: extended Kalman filter (EKF).** Let \(\mathbf{x}_k = [u_k, c_k]^\top\). The prediction step uses

\[
\mathbf{x}_{k+1|k} = \begin{bmatrix} \phi(c_k) u_k \\ c_k \end{bmatrix},
\]

with Jacobian evaluated at the mean \((\hat{u}_{k|k}, \hat{c}_{k|k})\)

\[
\mathbf{F}_k = \begin{bmatrix}
\phi(\hat{c}_{k|k}) & -\Delta t \, \hat{u}_{k|k} \, \phi(\hat{c}_{k|k}) \\
0 & 1
\end{bmatrix}.
\]

The process-noise covariance is

\[
\mathbf{Q}_k = \begin{bmatrix}
\dfrac{\sigma_u^2}{2\hat{c}_{k|k}}\Bigl(1 - \phi(\hat{c}_{k|k})^2\Bigr) & 0 \\
0 & Q_{\text{drift}}
\end{bmatrix},
\]

The process noises \(\varepsilon_k\) and \(\eta_k\) are independent, so \(\mathbf{Q}_k\) is block-diagonal. The state-dependence of \(\mathrm{Var}(\varepsilon_k)\) is handled by evaluating it at the current state estimate \((\hat{u}_{k|k}, \hat{c}_{k|k})\) — the same linearisation point as \(\mathbf{F}_k\) — a standard EKF procedure. The measurement Jacobian is \(\mathbf{H} = [1, 0]\) with measurement noise \(R\). The EKF yields the Gaussian approximation \(p(c_k \mid y_{1:k}) \approx \mathcal{N}(\hat{c}_{k|k}, P_{cc,k|k})\), and

\[
p_{\text{collapse},k} = \Phi\!\left( \frac{\delta - \hat{c}_{k|k}}{\sqrt{P_{cc,k|k}}} \right).
\]

---

## 4. Comparison with Heuristic EWS (Theoretical Advantage)

The classical approach estimates the lag-1 autocorrelation (or an AR(1) coefficient) from a moving window of length \(W\). Three honest statements replace the earlier, overextended claims:

**1. Windowed AR(1) carries a systematic lag.** A rolling window of length \(W\) estimates the *average* coefficient over the window. Reported at the window's centre, the estimate carries a lag of approximately \(W/2\) sampling intervals relative to the current value; reported at the window's end, it trails the true current coefficient by the drift accumulated over the window, which also scales with \(W\) for a slowly varying \(c(t)\). In either convention the systematic lag is of order \(W\).

**2. Residual delay is governed by evidence accumulation.** For any sequential test of "gap is degrading" versus "gap is stationary", the first-order detection delay is

\[
\text{delay} \approx \frac{\log(1/\alpha)}{D(f_1 \,\|\, f_0)},
\]

where \(\alpha\) is the target false-alarm rate and \(D(f_1 \| f_0)\) the Kullback–Leibler divergence between the change and no-change densities of the observed gap statistic (Shiryaev 1963; Pollak 1985; Tartakovsky & Veeravalli). Hence, **within the assumed generative model**, the observer's delay is governed by the evidence-accumulation rate, not by a window length. More precisely, for a given model of the drift (e.g., a constant drift rate), the RB-PF's detection delay is bounded by an evidence-accumulation expression; in the slow-drift limit this expression can approximate the optimal abrupt-change bound.

**3. Scoped optimality of the alarm rule.** The rule "stop when \(P(c_k < \delta \mid y_{1:k}) \geq \theta\)" is a Shiryaev-type posterior-probability threshold. Classical quickest-detection theory establishes Bayes optimality for **abrupt changes** of a parameter. Our spectral gap is a *slowly drifting* parameter rather than an abrupt change; consequently we claim **asymptotic Bayesian optimality within the assumed generative model under a slow-enough drift** (the posterior-probability rule is then an asymptotic approximation of the optimal rule), not exact optimality for arbitrary drift profiles.

**Proposition (scoped).** Under the state-space model of §2, for a fixed false-alarm rate \(\alpha\) the expected detection delay of the RB-PF observer is of order \(\log(1/\alpha)/D(f_1\|f_0)\), whereas a windowed AR(1) estimator of length \(W\) additionally carries a systematic lag of order \(W\). (Statement of intent; empirical verification in the experiments section.)

---

## 5. Model Parameters and Calibration

The model is fully specified by the following parameters, fixed per system in the experiments:

| Parameter | Meaning | Typical value (per system) |
|---|---|---|
| \(\Delta t\) | sampling interval | fixed by data cadence |
| \(\sigma_u\) | process noise of the mode | system-dependent |
| \(R\) | observation noise variance | system-dependent |
| \(Q_{\text{drift}}\) | drift noise of the spectral gap | small, controls tracking speed |
| \(c_{\min}\) | lower bound on the gap | > 0, e.g. \(10^{-3}\) |
| \(\delta\) | collapse threshold on \(c\) | system-dependent |
| \(\theta\) | alarm threshold on \(p_{\text{collapse}}\) | calibrated (below) |
| \(p(c_1)\) | prior | truncated normal with support \([c_{\min}, \infty)\) |

**Calibration.** The alarm threshold \(\theta\) is calibrated to a target false-positive rate against null (no-transition) trajectories (Youden index or fixed-FPR rule), following the caution that many null time series exhibit early-warning-looking statistics by chance (Boettiger & Hastings 2012). A sensitivity study over \((Q_{\text{drift}}, \delta, c_{\min}, \theta)\) is reported alongside the main results.

---

## 6. Scope and Limitations

- **Fold-type only.** The scalar-OU reduction holds for generic fold bifurcations. For **Hopf** bifurcations the leading eigenpair is complex and the lag-1 autocorrelation does not follow the fold pattern (Bury et al. 2020, 2021); a separate reduction over the complex pair (equivalently, the radial modulus) is required. For **period-doubling**, the dominant multiplier tends to **−1** (mirror coordinate), requiring a sign-alternating formulation rather than the \(c \to 0\) limit. Both are future work.
- **Identifiability.** As \(c \to 0\) the parameters \((\sigma_u, R, c)\) become jointly harder to identify; the variance blow-up of the OU at the crossing partially restores the signal. The sensitivity study quantifies this.
- **Optimality scope.** Exact Shiryaev optimality applies to abrupt changes; for a drifting gap we claim asymptotic optimality under slow-enough drift (§4.3).

---

## 7. Related Work

The closest precedent is **Ives & Dakos (2012)**, who fit time-varying AR/VAR models in a state-space form to detect changes in stability (rise of the dominant eigenvalue); our contribution relative to them is an online posterior over the spectral gap itself, a direct collapse probability, and a sequential-decision (quickest-detection) framing. **Ditlevsen & Ditlevsen (2023)** build an early warning of an AMOC collapse on the lag-1 autocorrelation of an SST fingerprint with a fixed-coefficient model; we replace the single autocorrelation statistic with full Bayesian tracking of the gap. **Bury et al. (2021)** show with deep-learning EWS that indicators are type-dependent (fold vs Hopf vs other), supporting our fold-scoped development. The alarm rule follows the Bayesian quickest-detection tradition of **Shiryaev (1963)** and subsequent theory (**Pollak 1985; Tartakovsky & Veeravalli**).

---

## 8. Final Formal Package (Summary)

1. **Generative SDE (fold type):** Near a generic fold, the dominant mode evolves as \( du_t = -c(t) u_t dt + \sigma_u dW_t \), with \(c(t) \geq c_{\min} > 0\) and \(c(t) \downarrow c_{\min}\) approaching the tipping point.
2. **Discrete state-space:** \( c_{k+1} = \max(c_{\min}, c_k + \eta_k) \), \(u_{k+1} = e^{-c_k \Delta t} u_k + \varepsilon_k(c_k)\), \(y_k = u_k + \nu_k\).
3. **Observer:** Rao–Blackwellised particle filter (particles over \(c_k\), conditional Kalman filter over \(u_k\)) estimates \(p(c_k \mid y_{1:k})\) and outputs the collapse probability; plain EKF retained as a linearised baseline.
4. **Optimality (scoped):** The alarm rule is a Shiryaev-type posterior-probability threshold; within the assumed generative model it is asymptotically Bayes-optimal for slow-enough drift, and its residual delay is of order \(\log(1/\alpha)/D(f_1\|f_0)\), independent of any window length.
5. **Calibration:** Threshold \(\theta\) is calibrated to a target FPR against null trajectories; parameters fixed per system with a sensitivity study.

This is the foundation for the experiments and the paper. The full verified literature basis is in `literature_review.md`.

---

## References

- Boxler, P. (1989). A stochastic version of center manifold theory. *Probability Theory and Related Fields* 83:509–545 (verify pagination at submission).
- Shiryaev, A. N. (1963). On optimum methods in quickest detection problems. *Theory of Probability and its Applications* 8:22–46.
- Pollak, M. (1985). Optimal detection of a change in distribution. *Annals of Statistics* 13:206–227.
- Tartakovsky, A. & Veeravalli, V. General asymptotic Bayesian theory of quickest change detection (SIAM/sequential-analysis; also Tartakovsky–Nikiforov–Basseville monograph).
- Doucet, A., Godsill, S. & Andrieu, C. (2000). On sequential Monte Carlo sampling methods for Bayesian filtering. *Statistics and Computing* 10:197–208.
- Li, P., Goodall, R. & Kadirkamanathan, V. (2004). Estimation of parameters in a linear state-space model using a Rao–Blackwellised particle filter. *IEE Proceedings – Control Theory and Applications* (verify volume/pages at submission).
- Ives, A. R. & Dakos, V. (2012). Detecting dynamical changes in nonlinear time series using locally linear state-space models. *Ecosphere* 3(6):art52. DOI 10.1890/ES11-00347.1.
- Ditlevsen, P. & Ditlevsen, S. (2023). Warning of a forthcoming collapse of the Atlantic Ocean circulation. arXiv:2304.09161 (published in *PNAS*).
- Bury, T. et al. (2021). Deep learning for early-warning signals of tipping points. *PNAS* 118(39):e2106140118. DOI 10.1073/pnas.2106140118.
- Bury, T. et al. (2020). Spectral indicators for early warning. *J. R. Soc. Interface* 17 (verify exact ID at submission).
- Boettiger, C. & Hastings, A. (2012). Early warning signals and the prosecutor's fallacy. *Proc. R. Soc. B* (verify DOI at submission).
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41:100–115. Roberts, S. W. (1966). A comparison of some control chart procedures. *Technometrics* 8:411–430 (verify pages at submission).
- Scheffer, M. et al. (2009). Early warning signals for critical transitions. *Nature* 461:53–59. DOI 10.1038/nature08227.
