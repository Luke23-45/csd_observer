# Formal Definition of the CSD Observer Benchmark

This document defines the benchmark used in the finalized experiments for this project. It is intended to support manuscript writing, table generation, and figure generation. It is deliberately written as a benchmark definition rather than a superiority claim: no method is assumed to dominate every dynamical system, every metric, or every patient-count setting.

## 1. Scope

The benchmark evaluates observer-based critical-transition detectors on three synthetic dynamical systems:

- `fold`
- `hopf`
- `logistic`

Each system is evaluated at five patient counts:

- `100`
- `200`
- `300`
- `400`
- `500`

The finalized reporting pipeline uses the following seed order when sorting and aggregating runs:

- `101`
- `202`
- `303`
- `404`
- `505`
- `606`
- `707`
- `808`
- `909`
- `1010`

For each patient count, the analysis uses the most complete batch available for that patient count. Partial reruns are retained in the experiment directory for traceability but are excluded from the main tables and figures.

## 2. Batch Selection Rule

Let `B_n` be the set of available benchmark batches for patient count `n`.

For each `n`, the selected batch `b*_n` is the batch that maximizes, in order:

1. the number of unique `(system, method, seed)` triples,
2. the number of systems covered,
3. the total number of records,
4. and, if needed, the most recent timestamp.

This rule selects the most complete batch in a deterministic way and prevents partial reruns from distorting the reported results.

## 3. Methods

The finalized visualization and reporting pipeline includes two observer variants:

- `Kalman-BCE`
- `Kalman-LSTM-Spec`

For each `(system, patient count, seed, method)` combination, the model produces a score sequence `p_{i,t} in [0, 1]` for each held-out test trajectory `i`. These scores are used consistently for thresholding, detection-time estimation, early-warning discrimination, and false-positive evaluation.

## 4. Metric Definitions

### 4.1 Detection Time

For a positive trajectory `i` with bifurcation time `tau_i` and score sequence `p_{i,t}`, define the first alert time as:

`a_i = min { t : 0 <= t < tau_i and p_{i,t} >= theta }`

where `theta` is the decision threshold.

The detection time for trajectory `i` is:

`DT_i = tau_i - a_i`

if an alert occurs before the bifurcation, and is undefined otherwise.

The reported detection time is the mean over all positive trajectories with at least one pre-bifurcation alert:

`DT = mean_i(DT_i)`

Trajectories that never cross the threshold before the bifurcation are undefined for this metric and are omitted from the mean. They are not counted as zero-lead detections.

### 4.2 Early-Warning AUC

The early-warning score for each positive signal trajectory is the maximum model score in a pre-bifurcation window:

`e_i = max { p_{i,t} : max(0, tau_i - 50) <= t < tau_i - 5 }`

For each null trajectory `j`, the corresponding score is computed over the final portion of the sequence:

`n_j = max { p_{j,t} : max(0, T_j - 50) <= t < T_j - 5 }`

where `T_j` is the null sequence length.

The reported early-warning AUC is the ROC AUC computed from the pooled set `{e_i}` and `{n_j}` with labels `1` for signal and `0` for null:

`EW-AUC = ROC-AUC({e_i}, {n_j})`

If only one class is present, the quantity is undefined.

### 4.3 False Positive Rate

The false positive rate is the fraction of null time steps whose score exceeds the threshold:

`FPR = (sum_j sum_t 1[p_{j,t} >= theta]) / (sum_j T_j)`

This is a per-time-step alert rate on null sequences, not a per-trajectory error rate.

### 4.4 Threshold Selection

For learned models, the threshold is selected on the validation split before test evaluation. The validation labeling rule uses:

- positive validation scores from `max(0, tau_i - 60) <= t < tau_i`, and
- negative validation scores from the early pre-bifurcation region `0 <= t < max(max(0, tau_i - 120) - 1, 0)`.

If optional null validation scores are supplied, their first `120` time steps are also labeled as negative. In the finalized benchmark runs for the two reported learned methods, threshold selection uses the signal validation trajectories.

The threshold is selected from the ROC curve as follows. First, the method chooses the threshold that maximizes Youden's J statistic:

`J = TPR - FPR`

If the corresponding sensitivity is at least `0.80`, that threshold is used. Otherwise, the threshold whose sensitivity is closest to `0.80` is used. If the validation labels contain only one class, or if the validation scores are effectively constant, the fallback threshold is `0.5`.

This keeps the threshold fixed before testing and avoids using the test set for calibration.

## 5. Reporting Convention

All reported summary values are aggregated across seeds within the selected batch for each patient count, system, and method. The manuscript should interpret the tables and figures locally:

- compare methods within the same system and patient count,
- compare metrics separately,
- report trade-offs instead of declaring a single universally best method.

This reporting rule is intentional. The benchmark is designed to show where each method is strong, where it is weak, and how the trade-off changes across systems and patient-count settings.

## 6. Manuscript Note

The final manuscript should describe these results as benchmark comparisons, not as proof of universal superiority. A method may improve detection time on one system while losing EW-AUC or false-positive performance on another. The correct interpretation is conditional performance under the evaluated systems, patient counts, seeds, and metrics.

The appropriate claim is therefore:

`The proposed observer variants exhibit system- and metric-dependent trade-offs in early-warning performance.`

The inappropriate claim is:

`The proposed method is universally best or state of the art across all settings.`
