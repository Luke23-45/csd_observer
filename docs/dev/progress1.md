Yes, be strict here.

**Honest verdict:** most of your current `Kalman-*` rows are **not published baseline methods**. They are internal prototype heads / training variants of your own pipeline, so they should **not** appear as main-paper baselines unless you explicitly frame them as ablations.

**What is actually standard and published**

These are the baseline families that are defensible in the early-warning / tipping-point literature:

- **Variance / running variance**  
  Standard critical-slowing-down indicator. Source: [Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)

- **Lag-1 autocorrelation / AR(1) / AC1**  
  Standard indicator. Source: [Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)

- **Skewness**  
  Standard indicator in the canonical EWS suite. Source: [Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)

- **Spectral ratio / low-frequency power**  
  Standard indicator in the EWS suite. Source: [Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)

- **DFA**  
  Standard indicator in the same family. Source: [Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)

- **Return rate / recovery time**  
  Standard indicator in the same family. Source: [Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)

- **Locally linear state-space / time-varying AR state-space**  
  Published model-based baseline. Source: [Ives & Dakos 2012](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/ES11-00347.1)

- **Bayesian/state-space model-comparison early warning**  
  Published model-based baseline. Source: [Boettiger & Hastings 2012](https://pubmed.ncbi.nlm.nih.gov/23055060/)

**Published but optional, not “standard canonical”**
- **Deep learning EWS**: [Bury et al. 2021](https://pubmed.ncbi.nlm.nih.gov/34544867/)  
- **Reservoir computing tipping-point detection**: [Li et al. 2023](https://spj.science.org/doi/10.34133/research.0174)

Those are real published baselines, but I would treat them as **secondary/appendix baselines**, not the core standard set, unless your paper is explicitly about beating modern ML methods.

---

## What to remove from the main baseline table

These are **not standard published baselines** in the literature:

- `Kalman-Lag2`
- `Kalman-BCE`
- `Kalman-LSTM`
- `Kalman-LSTM-Spec`
- `Kalman-Lag2-Net`
- `Kalman-ACKO` unless you can point to a specific published paper and identical implementation
- `Kalman-LSTM-Aug`
- `Kalman-BCE-Spec`

They are your own internal model variants, not literature baselines.

---

## What to do with your current non-Kalman indicator rows

These are **closer to standard**, but they need cleanup:

- `RunningVar`  
  Keep only if it is clearly labeled as **variance**.

- `Raw-CSD`  
  This is ambiguous. If it is really AC1 / AR(1), rename it to **AC1** or **AR1**. If not, drop it.

- `Lag2-CSD` and `Lag2-CSD-detrended`  
  I would **not** use these as main baselines. Lag-2 is not the canonical early-warning indicator; the standard literature is lag-1 autocorrelation, not lag-2.

So: if you want a clean paper, keep **Variance, AC1, Skewness, Spectral Ratio, DFA, Return Rate**, and drop the custom lag-2 family from the main table.

---

## Best baseline set for your paper

If you want the strongest defensible comparison for `Kalman-Spectral-Drift`, I would use:

1. **Variance**
2. **AC1 / AR(1)**
3. **Skewness**
4. **Spectral ratio / low-frequency power**
5. **DFA**
6. **Return rate / recovery time**
7. **Ives & Dakos 2012** state-space observer
8. **Boettiger & Hastings 2012** Bayesian state-space / model comparison

Optional appendix-only:
9. **Bury 2021 deep learning EWS**
10. **Reservoir computing 2023**
11. **DMD / DEV** if you want extra dynamical-systems baselines, but I would not call them standard core baselines

---

## Bottom line

- Your current `Kalman-*` family is **not** a valid “published baseline suite.”
- Your **proposed method** is `Kalman-Spectral-Drift`.
- Your **main baselines** should be the published indicator suite + the two published state-space/Bayesian baselines.
- Keep the neural / reservoir methods only if you want a stronger appendix or a “modern ML comparison” section.

If you want, I can next give you a **final baseline table** with exact names you should use in the paper, so it is publication-safe and internally consistent.