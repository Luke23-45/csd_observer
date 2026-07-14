Now I have everything I need. Let me deliver the triple-verified, corrected formal definition. I found **four real errors** — one of which is a silent index bug that would crash an implementation on the very first valid time step.I've now verified everything against the literature (CWNA model derivation, period-doubling eigenvalue theory, standard EWS lag formulas, Kalman recursion). Here is the complete corrected specification, with every error annotated.

---

## Bug Report First — Four Issues Found

**Bug 1 (Critical — implementation crash):** The "causal" formula requires `x_{t−W−1}`, meaning data from index `t−W−1` onward. At `t = W` (the stated start), that index is `x_{−1}` — undefined. The series cannot start at `t = W` using that formula; it would start at `t = W+2`. Fix: use the standard within-window formula, which uses `W−2` pairs and is valid exactly from `t = W`.

**Bug 2 (Notation):** Curly braces `{…}` denote sets (unordered), but Pearson correlation requires ordered pairs. Using them conflates vectors with sets and breaks the pairing semantics.

**Bug 3 (Numerical stability):** The simple covariance update `P_{t|t} = (I − K_t H) P_{t|t−1}` loses symmetry and positive-definiteness after many iterations due to floating-point cancellation. The Joseph form is mandatory for robust long-run operation.

**Bug 4 (Missing standard step):** Within-window linear detrending before computing correlation is the universal convention in the EWS/CSD literature (Dakos 2008, Bury 2020, etc.) and prevents a trend inside the window from inflating the autocorrelation estimate. The document omits it entirely.

**Verified correct:** The `Q` matrix is the *Continuous White Noise Acceleration* (CWNA) model, yielding `Q = q·[[Δt³/3, Δt²/2],[Δt²/2, Δt]]` with `Δt = 1` — confirmed analytically by multiple sources. The `[[1/4,…]]` variant from some texts is the discrete DWPA model (different physical assumption). The document's choice is correct and should be labeled explicitly.

---

## Corrected Formal Definition

**Kalman-Filtered Lag-2 Criticality Index for Period-Doubling Early Warning**

---

### 1. Input Signal and Rolling Lag-2 Autocorrelation

Let `{x_t}_{t=1}^T` be the inter-beat interval (IBI) time series, `x_t ∈ ℝ⁺`. Fix window length `W = 30`.

For each beat index `t ≥ W`, define the **within-window detrended lag-2 Pearson autocorrelation**. Let the window residuals be obtained by removing the within-window linear trend from `{x_{t−W+1}, …, x_t}`, yielding `{x̃_{t−W+1}, …, x̃_t}`. Then define the two **ordered** vectors:

```
a(t) = (x̃_{t−W+1}, x̃_{t−W+2}, …, x̃_{t−2})  ∈ ℝ^{W−2}
b(t) = (x̃_{t−W+3}, x̃_{t−W+4}, …, x̃_t    )  ∈ ℝ^{W−2}
```

Both vectors have `W−2` components; the `i`-th element of `b` is the `i`-th element of `a` shifted forward by exactly 2 beats. This uses only data up to and including time `t`, making the estimate strictly causal. The raw lag-2 autocorrelation observation is:

```
        Σᵢ (aᵢ − ā)(bᵢ − b̄)
ρ₂(t) = ───────────────────────────────────────
         √[Σᵢ(aᵢ−ā)²] · √[Σᵢ(bᵢ−b̄)²]
```

where `ā`, `b̄` are the respective sample means over the `W−2` pairs, and the sum runs from `i = 1` to `W−2`. This produces the observation sequence `y_t := ρ₂(t) ∈ [−1, 1]` for `t = W, W+1, …, T`.

> **Theoretical grounding.** Near a period-doubling bifurcation the linearized return map is `x_{t+1} = λ x_t + σ εt` with `λ → −1`. The lag-τ autocorrelation satisfies `ACF(τ) = λ^|τ|`, so `ACF(2) = λ² → +1`. Monitoring `y_t` approaching `+1` is therefore the correct scalar early-warning signal for this bifurcation type.

---

### 2. State-Space Model

We track the time-varying "true" lag-2 autocorrelation `μ_t` and its discrete-time derivative `δ_t` using the **Continuous White Noise Acceleration (CWNA) constant-velocity model**, which assumes the drift `δ_t` evolves as an integrated Wiener process.

**State vector** (2 × 1):

```
z_t = (μ_t, δ_t)ᵀ
```

**State transition:**

```
z_t = F z_{t−1} + w_t,    F = [[1, 1], [0, 1]],    w_t ~ N(0, Q)
```

**Process noise covariance** (CWNA model, sampling interval Δt = 1, scalar intensity `q > 0`):

```
Q = q · [[1/3,  1/2],
          [1/2,  1  ]]
```

*Derivation:* integrating `E[a(t)a(s)] = q · δ(t−s)` through the transition matrix gives `Q_{11} = q Δt³/3`, `Q_{12} = Q_{21} = q Δt²/2`, `Q_{22} = q Δt`. For Δt = 1 these reduce to the values above. Note: the alternative DWPA model (discrete random-acceleration impulse) yields `Q_{11} = q Δt⁴/4`; the two differ only in their upper-left entry. The CWNA form is used here.

**Observation equation** (scalar):

```
y_t = H z_t + v_t,    H = (1, 0),    v_t ~ N(0, r)
```

Fix `r = 1`. Scaling both `q` and `r` by a common factor α leaves the ratio `q/r` — and thus the steady-state Kalman gain — invariant, so this normalisation is without loss of generality.

---

### 3. Kalman Filter Recursion

**Initialisation** at `t = W`:

```
ẑ_{W|W} = (y_W, 0)ᵀ

P_{W|W} = [[r,    0   ],
            [0,    r/10]]
```

The initial position uncertainty equals the measurement noise variance `r = 1`; the initial velocity uncertainty `r/10 = 0.1` reflects the prior that the rate of change starts near zero.

**For each subsequent step `t = W+1, …, T`:**

**Step 1 — Predict:**

```
ẑ_{t|t−1} = F ẑ_{t−1|t−1}

P_{t|t−1} = F P_{t−1|t−1} Fᵀ + Q
```

**Step 2 — Innovation:**

```
ỹ_t = y_t − H ẑ_{t|t−1}
```

**Step 3 — Innovation variance** (scalar, since H is 1 × 2):

```
S_t = H P_{t|t−1} Hᵀ + r  =  P_{t|t−1}[1,1] + r
```

**Step 4 — Kalman gain** (2 × 1 vector):

```
K_t = P_{t|t−1} Hᵀ S_t⁻¹
```

**Step 5 — Update (Joseph form — numerically stable):**

```
ẑ_{t|t} = ẑ_{t|t−1} + K_t ỹ_t

L_t := I − K_t H    (2 × 2)

P_{t|t} = L_t P_{t|t−1} L_tᵀ + r · K_t K_tᵀ
```

After the update, enforce symmetry to prevent floating-point drift:

```
P_{t|t} ← (P_{t|t} + P_{t|t}ᵀ) / 2
```

> **Why Joseph form over the simple form:** `P_{t|t} = L_t P_{t|t−1}` is algebraically equivalent only when `K_t` is exactly optimal. The Joseph form is correct for any gain and is guaranteed to yield a symmetric positive-semi-definite result even when finite precision corrupts the recursion.

---

### 4. Criticality Score

**Primary score** (filtered estimate of lag-2 autocorrelation only):

```
s_t^(0) = μ̂_{t|t} = (1, 0) ẑ_{t|t}
```

**Optional boosted score** (penalizes rising autocorrelation more strongly):

```
s_t^(1) = μ̂_{t|t} + β · max(0, δ̂_{t|t}),    δ̂_{t|t} = (0, 1) ẑ_{t|t}
```

with `β ≥ 0` tuned identically to `q` (see §5). Use `s_t^(0)` unless there is an explicit reason to boost trend sensitivity.

The score `s_t` is defined for `t ≥ W`; set `s_t = 0` for `t < W`.

---

### 5. Parameter Tuning

Fix `r = 1`. Select `q` from the log-spaced grid:

```
Q = {10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹}
```

If `s_t^(1)` is used, extend the search to all combinations with `β ∈ {0, 0.5, 1.0, 2.0}`.

All tuning is performed **within the training folds** of a cross-validation split:

1. For each candidate `(q, β)`, compute scores on training trajectories.
2. Sweep a decision threshold `θ` over the empirical range of `s_t` and maximise **Youden's J statistic** `J = sensitivity + specificity − 1`.
3. Select the `(q, β, θ)` triple with highest balanced accuracy (or the pre-specified criterion, e.g. sensitivity ≥ 0.95 subject to minimum specificity).
4. Apply the chosen triple to the held-out test fold without further modification.

---

### 6. Remarks

- The formula in §1 uses `W−2 = 28` pairs per window (for `W = 30`), which is ample for a stable Pearson estimate and begins at `t = W` without requiring any look-back beyond the available data.
- Within-window linear detrending (§1) prevents slowly drifting IBI baselines from biasing `ρ₂(t)` upward, a known confound documented in the EWS literature.
- The CWNA `Q` matrix is derived from continuous-time first principles and is the standard choice when the underlying signal (here, the lag-2 autocorrelation) is expected to vary smoothly.
- All computations are deterministic given `(q, β)` and the initial state; no gradient-based learning is used at any step.
- Behavior at extremes: as `q → 0`, `s_t` approaches a simple exponential smoother of `y_t`; as `q → ∞`, `s_t → y_t` (raw lag-2 values). The grid search selects the optimal noise-suppression/responsiveness tradeoff.