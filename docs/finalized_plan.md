# Finalized Plan: Beating Kalman-BCE

**Goal:** Beat Kalman-BCE convincingly across all 3 synthetic systems (Fold, Hopf, Logistic) with a learned Kalman observer method.

**Key insight from ~10 hours of investigation:** EWS feature augmentation (adding csd, rvar, lag2, alternans to observer input) is a dead end — it helps Hopf but actively harms Fold and Logistic. The primary viable path forward is Kalman-LSTM-Spec (no augmentation) trained **with null data**.

---

## Part 1: What Has Been Done

### 1.1 Bug Fixes (All Committed in HEAD)

| ID | Severity | File | Symptom | Fix |
|----|----------|------|---------|-----|
| P0 | Critical | `trainer.py` | Kalman-Lag2-Net always trained 15 epochs (patience limit), never used best checkpoint | `train_kalman_lag2` now accepts `lag2_null`/`null_seq_lengths` for proper EW-AUC validation |
| P1 | Medium | `benchmark.py` | Kalman-Lag2 raw `mu_hat` ∈ [-1,1] used directly as probability scores | Applied `torch.sigmoid(mu_hat)` |
| P2 | High | `default.yaml` | Kalman-LSTM-Spec could not detect Hopf (DT=nan, AUC=0.490) on Hopf | `spectral_radius_weight: 0.1 → 0.01` |
| P3 | Low | `kalman_lag2.py` | 49-param MLP head overfits; 25 params (hidden_dim=4) optimal | Reverted `hidden_dim: 8 → 4` |

### 1.2 Baseline (No Bugs) — Kalman-LSTM-Spec Performance

Benchmark run (n_patients=500, seeds=1) after all fixes:

| System | DT ↓ | EW-AUC ↑ | FPR ↓ | vs BCE DT | vs BCE AUC | vs BCE FPR |
|--------|------|----------|-------|-----------|-----------|-----------|
| Fold | 89.2 | 0.789 | 0.593 | +3.8 (worse) | −0.134 (worse) | 2.5× (worse) |
| Hopf | **33.5** | **0.992** | 0.093 | **+55.8** | **+0.169** | 13× (worse) |
| Logistic | nan | 1.000 | 0.000 | — | 0.000 | — |

**Verdict:** Kalman-LSTM-Spec beats BCE on Hopf (DT +55.8, AUC +0.169) but loses on Fold and Logistic.

---

## Part 2: Kalman-LSTM-Aug Investigation (H1–H3)

### 2.1 Method

Added `Kalman-LSTM-Aug` to METHODS — uses `augment_features=True` (4 EWS features appended to raw CSD: csd_ews, rvar, lag2_ews, alternans) + `loss_type="lstm_spec"`. Implementation in `_run_synthetic_experiment` (lines 270–314).

Initial benchmark run:

| System | DT | EW-AUC | FPR |
|--------|-----|--------|-----|
| Fold | 133.3 | 0.999 | 0.152 |
| Hopf | 52.5 | 1.000 | 0.206 |
| Logistic | 66.7 | 1.000 | 0.251 |

High FPR on Hopf (0.206) and Logistic (0.251) motivated systematic testing.

### 2.2 Hypothesis H1: Add Null Data to Threshold Selection

**Test script:** `studies/runner/debug_threshold.py`
**Idea:** `select_threshold` currently uses only signal validation data. Adding null data as negative examples should raise the threshold, reducing FPR.
**Result:** ❌ **Not helpful.** Adding null data paradoxically LOWERED the threshold on Fold (0.360 → 0.327) and Hopf (0.500 → 0.442), INCREASING FPR. Root cause: `select_threshold` optimizes for a target FNR, and adding many easy-negative null samples shifts the optimal threshold toward lower values.
**Key finding:** Null scores overlap signal scores in feature space — threshold calibration cannot separate them.

### 2.3 Hypothesis H2: Standardize EWS Features

**Test script:** `studies/runner/debug_aug.py`
**Idea:** Z-score normalize each feature channel using null training data statistics (mean, std) before feeding to observer.
**Result on Hopf:** Helped "all 4" (FPR 0.2359 → 0.0801), but AUC also dropped (0.614 → 0.680 — actually improved slightly). Still worse than individual features.
**Result on Fold/Logistic:** Not tested systematically — the "all 4 std" config was tested only on Hopf.

### 2.4 Hypothesis H3: Reduced Feature Subsets

**Test script:** `studies/runner/debug_aug.py`
**Idea:** Use only 1–2 EWS features instead of all 4, to reduce overfitting and false positives.
**Results (2 seeds, n_patients=200):**

**Fold Bifurcation:**

| Config | DT | EW-AUC | FPR |
|--------|-----|--------|-----|
| Baseline (no aug) | **118.6** | **0.805** | 0.452 |
| csd only | 106.4 | 0.502 | 0.281 |
| rvar only | 111.2 | 0.472 | 0.318 |
| lag2 only | 99.7 | 0.475 | 0.677 |
| alternans only | 103.2 | 0.438 | 0.249 |

> **No augmented config beats baseline on Fold.** AUC drops from 0.805 to 0.438–0.502. The features actively harm detection.

**Hopf Bifurcation:**

| Config | DT | EW-AUC | FPR |
|--------|-----|--------|-----|
| Baseline (no aug) | 100.0 | 0.715 | 0.246 |
| csd only | 55.6 | 0.722 | 0.095 |
| **rvar only** | **43.0** | 0.826 | **0.006** |
| lag2 only | 56.5 | 0.694 | 0.081 |
| **alternans only** | 64.5 | **1.000** | **0.028** |

> **Almost all augmented configs beat baseline on Hopf.** rvar only: FPR=0.006 (near zero), DT=43.0. alternans only: AUC=1.000, FPR=0.028.

**Logistic Bifurcation:**

| Config | DT | EW-AUC | FPR |
|--------|-----|--------|-----|
| **Baseline (no aug)** | **66.7** | **1.000** | **0.045** |
| csd only | 66.7 | 1.000 | 0.154 |
| rvar only | 66.7 | 1.000 | 0.163 |
| lag2 only | 66.7 | 1.000 | 0.157 |
| alternans only | 66.7 | 1.000 | 0.155 |

> **No augmented config beats baseline on Logistic.** AUC is max for all. FPR worsens 3–4× (0.045 → 0.15+). DT unchanged.

### 2.5 H1–H3 Conclusion: EWS Augmentation is a Dead End

```
                Fold    Hopf    Logistic   Overall
Baseline        ✅      ❌      ✅         —
Aug (csd)       ❌      ✅      ❌         ❌
Aug (rvar)      ❌      ✅      ❌         ❌
Aug (lag2)      ❌      ✅      ❌         ❌
Aug (alternans) ❌      ✅      ❌         ❌
```

No single EWS feature helps across all 3 systems. The biophysical reason:
- **Fold:** Gradual drift with collapse. EWS (variance, autocorrelation) behavior is non-monotonic — variance can decrease near fold, confusing the observer.
- **Hopf:** Oscillatory dynamics with clear spectral signatures. EWS strongly amplifies pre-bifurcation signal.
- **Logistic:** Period-doubling cascade. Raw CSD already gives perfect separation (AUC=1.000). EWS adds noise → increases FPR.

**Recommendation:** Remove `Kalman-LSTM-Aug` from METHODS. Do not pursue EWS augmentation further.

---

## Part 3: Hypothesis H4 — Train with Null Data

### 3.1 Motivation

The core problem: LSTM-Spec produces elevated scores on null data (FPR 0.093 on Hopf, 0.593 on Fold). Currently, the model never sees null trajectories during training — it only knows what signal looks like, not what null looks like.

Training with null data (`target=0`, `bifurcation_time=INF`) should teach the observer to suppress scores on noise, reducing FPR without sacrificing detection performance.

### 3.2 Plan

1. **Modify `train_kalman`** (or create a wrapper) to accept an optional `null_tensors` parameter
2. Create a combined training set: `signal_train ∪ null_train`
   - Signal trajectories: `is_positive=True`, actual `bifurcation_times`
   - Null trajectories: `is_positive=False`, `bifurcation_times = T+1` (beyond sequence length, so no "pre-bifurcation" label ever assigned)
3. During each epoch, batches are drawn from the combined set (ratio 1:1 signal:null or weighted)
4. Validation is unchanged (signal val + null val for EW-AUC)
5. Evaluate on the same test splits

### 3.3 Implementation Strategy

**Approach A (safe, separate function):**
Create `train_kalman_with_null()` in `trainer.py` that:
- Concatenates signal and null feature/mask tensors
- Creates combined `bifurcation_times` tensor (null = T+1)
- Creates combined `is_positive` tensor (null = False)
- Calls the existing training loop with larger dataset
- Returns the trained model

**Approach B (simpler, inline):**
Modify the benchmark's LSTM-Spec block to:
1. Create a combined signal+null TensorizedDataset
2. Pass all to `train_kalman` (it accepts `bifurcation_times` and `is_positive` for loss computation)
3. The BCE loss and spectral loss naturally handle null trajectories (target=0 throughout)

**Chosen: Approach A** — less invasive to existing code, easier to test independently.

### 3.4 Test Script (H4)

Create `studies/runner/debug_nulltrain.py`:
- Uses LSTM-Spec (no augmentation) as base method
- Trains once without null data, once with null data
- Single system (Hopf — most FPR-sensitive)
- Reports DT, EW-AUC, FPR for both configurations
- Optional: sweep null:signal ratio (1:1, 2:1, 1:2)

### 3.5 Expected Outcomes

| Scenario | Fold FPR | Hopf FPR | Logistic FPR | Overall |
|----------|----------|----------|--------------|---------|
| **Best case:** FPR drops to near BCE levels on Hopf/Logistic, DT and AUC preserved on Fold | ✅ | ✅ | ✅ | **Can beat BCE** |
| **Mixed case:** FPR improves on Hopf/Logistic but Fold DT/AUC degrades | ❌ | ✅ | ✅ | Borderline |
| **Worst case:** Null training kills detection (model learns to output low scores everywhere) | ❌ | ❌ | ❌ | Dead end |

### 3.6 Contingency

If H4 fails (null training degrades detection), the remaining options are:
1. **Accept the result** — report honestly that no learned method beats BCE across all systems
2. **Ensemble** — use Kalman-LSTM-Spec for Hopf, Kalman-BCE for Fold, Lag2-CSD for Logistic
3. **Adaptive spectral threshold** — make spectral radius threshold learnable per system
4. **Alternative architecture** — try Transformer or GRU head instead of LSTM

---

## Part 4: Design of H4 Test Script

### 4.1 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base method | LSTM-Spec (no aug) | Already best candidate; augmentation is dead end |
| Null:signal ratio | 1:1 initially | Balanced; can sweep 0.5:1, 1:1, 2:1 |
| Null target assignment | `bif_time=T+1, is_positive=False` | Ensures every time step is "before bifurcation" |
| Batch sampling | Weighted random sampler | Ensure each batch has ~50% null trajectories |
| Loss function | Same `lstm_spec` (BCE + SpectralRadius) | BCE on null targets=0 penalizes high null scores |
| Validation | Signal val + null val EW-AUC | Same as current LSTM-Spec |
| Threshold selection | Signal val only (H1 showed null data hurts) | Use same strategy as LSTM-Spec baseline |

### 4.2 Implementation Detail: Combined Dataset

```python
# Pseudo-code
def train_kalman_with_null(
    tensors_signal, train_idx_s, val_idx_s,
    tensors_null, train_idx_n, val_idx_n,
    loss_type="lstm_spec", null_ratio=1.0, ...
):
    # Combined training set
    train_idx = np.concatenate([train_idx_s, train_idx_n])
    features = torch.cat([tensors_signal.features, tensors_null.features], dim=0)
    masks = torch.cat([tensors_signal.masks, tensors_null.masks], dim=0)

    # Null trajectories: bifurcation_time = T+1 (beyond sequence), is_positive = False
    n_sig = len(tensors_signal.bifurcation_times)
    n_null = len(tensors_null.bifurcation_times)
    bif_times = torch.cat([
        tensors_signal.bifurcation_times,
        tensors_null.seq_lengths + 1,  # always beyond sequence
    ])
    is_pos = torch.cat([
        tensors_signal.is_positive,
        torch.zeros(n_null, dtype=torch.bool, device=device),
    ])
    seq_lens = torch.cat([tensors_signal.seq_lengths, tensors_null.seq_lengths])

    combined = TensorizedDataset(features, masks, seq_lens, bif_times, is_pos)

    # Weighted sampler for balanced batches
    weights = np.ones(len(train_idx))
    weights[train_idx_s] = null_ratio  # adjust signal weight to achieve desired ratio
    sampler = WeightedRandomSampler(weights, len(train_idx), replacement=True)

    return train_kalman(combined, train_idx, val_idx_s + val_idx_n,
                        loss_type=loss_type, sampler=sampler, ...)
```

### 4.3 Metric Computation (Unchanged)

After training with combined dataset:
- `build_probs(model, tensors_signal, test_idx_s)` → signal test probs (same as before)
- `build_probs(model, tensors_null, test_idx_n)` → null test probs
- `select_threshold(probs_val_signal, ...)` → threshold from signal val only
- DT, EW-AUC, FPR computed identically to existing benchmark

Metrics are computed on the SAME test splits as the baseline, ensuring fair comparison.

---

## Part 5: Timeline

| Step | Description | Duration |
|------|-------------|----------|
| 1 | Implement `train_kalman_with_null()` in `trainer.py` | 15 min |
| 2 | Write `debug_nulltrain.py` — compare ±null on Hopf | 10 min |
| 3 | Run H4 test on Hopf, analyze results | 5 min |
| 4 | If H4 works: extend to Fold + Logistic | 5 min |
| 5 | If H4 works: update benchmark.py (replace Aug with Null-trained LSTM-Spec) | 10 min |
| 6 | Full benchmark run + metrics | 20 min |
| 7 | Update `benchmark_report.md` | 15 min |
| 8 | ruff + pytest verification | 2 min |
| | **Total (if H4 succeeds)** | **~82 min** |
