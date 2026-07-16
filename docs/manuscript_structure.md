# Manuscript Structure

This document defines the working structure for the manuscript. It follows the standard scientific-paper organization recommended by major publishers while adapting it to this project's actual benchmark: learned Kalman observer variants evaluated across fold, Hopf, and logistic systems with patient-count sweeps.

The manuscript should be written as a careful empirical and methodological study. It should not claim that the proposed method is universally best or state of the art across all settings.

## Source Guidance Used

The structure below is based on common manuscript guidance from:

- Elsevier: scientific articles commonly use Title, Abstract, Keywords, followed by the IMRAD structure: Introduction, Methods, Results, and Discussion. Source: `https://www.elsevier.com/connect/11-steps-to-structuring-a-science-paper-editors-will-take-seriously`
- Elsevier condensed structure guidance: a typical paper includes Title, Abstract, Keywords, Introduction, Methods, Results, Discussion, Conclusion, Acknowledgements, References, and Supporting Materials. Source: `https://www.elsevier.com/connect/the-condensed-read-how-to-structure-a-science-paper`
- IEEE Author Center: the abstract should summarize the research, conclusions, and implications, and the article should clearly separate problem, method, results, and interpretation. Source: `https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/`
- Nature-family article guidance: the abstract should be concise, the main text should communicate the central result clearly, and methods/results should be organized so readers can evaluate the evidence. Sources: `https://www.nature.com/srep/author-instructions/submission-guidelines`, `https://www.nature.com/ncomms/submit/article`
- Springer Nature author guidance: each manuscript section has a specific role, and structure affects how easily the work can be read and cited. Source: `https://www.springernature.com/gp/researchers/the-researchers-source/publish-with-impact-blogpost/writing-a-manuscript-and-mastering-abstracts/25261924`

## Final Recommended Structure

### Title

Purpose: state the paper's core contribution without overclaiming.

Working title:

`Regime-Dependent Performance of Learned Kalman Observers for Early-Warning Detection`

Alternative title:

`When Do Learned Kalman Observers Help? A Cross-Bifurcation Study of Early-Warning Detection`

Avoid titles that imply universal superiority, such as "A State-of-the-Art Early-Warning Detector".

### Abstract

Purpose: summarize the problem, method, benchmark, main result, and interpretation in one compact paragraph.

Required content:

- early-warning detection is regime-dependent,
- the study evaluates learned Kalman observer variants across fold, Hopf, and logistic systems,
- the benchmark uses patient-count sweeps and multiple seeds,
- the results show system- and metric-dependent trade-offs,
- the conclusion is conditional performance, not universal dominance.

Do not write the abstract until the Results and Discussion sections are stable.

### Keywords

Recommended keywords:

- Critical transitions
- Early-warning signals
- Kalman observer
- Bifurcation detection
- Spectral regularization
- Dynamical systems

## 1. Introduction

Purpose: motivate the scientific problem and state the contribution.

Recommended flow:

1. Critical transitions require early-warning detection because late detection can miss the actionable pre-transition window.
2. Many early-warning methods are evaluated on isolated systems, but different bifurcations produce different warning signatures.
3. A fold transition, a Hopf transition, and a logistic or period-doubling transition should not be assumed to favor the same observer.
4. Learned observers are attractive because they can adapt score trajectories from data, but their inductive biases may help one regime and hurt another.
5. This paper evaluates learned Kalman observer variants across multiple dynamical regimes and patient-count settings.
6. The central finding is that observer performance is regime- and metric-dependent.

Contribution statement:

`We provide a controlled cross-bifurcation benchmark of learned Kalman observer variants for early-warning detection and show that performance depends on the interaction between dynamical regime, data depth, and metric choice.`

Safe contribution bullets:

- We formalize a benchmark across fold, Hopf, and logistic systems.
- We compare learned Kalman observer variants across patient-count settings.
- We evaluate detection time, early-warning AUC, and false-positive rate.
- We show that no single observer variant dominates all systems and metrics.
- We identify where spectral or recurrent observer structure is useful and where simpler observers remain competitive.

### 1.1 Background and Motivation

Discuss early-warning detection, critical slowing down, and why observer-based scoring is useful.

Keep this focused. Do not turn the introduction into a full textbook section.

### 1.2 Problem Gap

State the gap clearly:

`The missing question is not whether a learned observer can win on one system, but whether the same observer behavior transfers across qualitatively different bifurcation mechanisms.`

### 1.3 Contributions

Use the contribution bullets above. Keep the claims tied to the evaluated systems and metrics.

## 2. Related Work

Purpose: place the study in the existing literature.

Recommended subsections:

### 2.1 Classical Early-Warning Signals

Cover variance, autocorrelation, lag-based indicators, and critical slowing down.

Role in this paper:

- establishes the non-learned baseline context,
- motivates why simple indicators can remain competitive,
- helps explain why learned methods should be compared against strong simple baselines.

### 2.2 Observer and Kalman-Based Approaches

Cover Kalman filtering, observer dynamics, and learned observer variants.

Role in this paper:

- motivates the use of a Kalman observer,
- explains why stability and filtering behavior matter,
- prepares the reader for the learned observer design.

### 2.3 Learning Across Dynamical Regimes

Discuss why methods may not transfer uniformly across regimes.

Role in this paper:

- supports the honest framing that method performance can be conditional,
- prepares reviewers for a trade-off result rather than a universal-best claim.

If targeting a venue with strict IMRAD formatting, this section can be merged into the Introduction.

## 3. Problem Setting and Benchmark Definition

Purpose: define exactly what is being evaluated.

This section should be aligned with `docs/definition/formal_definination.md`.

Recommended subsections:

### 3.1 Dynamical Systems

Define the three systems:

- fold,
- Hopf,
- logistic.

For each system, state the qualitative transition mechanism and why it is a distinct test case.

### 3.2 Trajectory Data and Patient Counts

Define:

- patient counts: `100`, `200`, `300`, `400`, `500`,
- seed order: `101`, `202`, `303`, `404`, `505`, `606`, `707`, `808`, `909`, `1010`,
- signal and null trajectories,
- train/validation/test split logic if needed.

### 3.3 Batch Selection

State the deterministic rule for selecting the most complete batch per patient count.

This prevents reviewers from thinking partial reruns were hand-selected.

### 3.4 Evaluation Metrics

Define:

- detection time,
- early-warning AUC,
- false-positive rate,
- threshold selection.

Use the formal definitions already finalized. Do not redefine them differently in the manuscript.

## 4. Methods

Purpose: describe the observer variants and training procedure clearly enough for reproduction.

Recommended subsections:

### 4.1 Learned Kalman Observer

Describe the observer model at the level needed for the paper:

- input trajectory,
- latent observer state,
- score sequence,
- learned components,
- output probability or warning score.

Avoid excessive implementation detail in the main text. Put exact architecture settings in the Methods details or supplement.

### 4.2 Observer Variants

Main reported variants:

- `Kalman-BCE`
- `Kalman-LSTM-Spec`

Describe why these two are the main comparison:

- `Kalman-BCE` is the strong learned baseline,
- `Kalman-LSTM-Spec` represents the proposed or focal spectral/recurrent observer variant.

### 4.3 Training Objective and Calibration

Define:

- training loss,
- validation criterion,
- threshold selection,
- early stopping if used,
- random seed handling.

Keep calibration separate from test evaluation.

### 4.4 Implementation and Reproducibility

State where the code and generated artifacts live:

- model and training code,
- benchmark runner,
- experiment data directory,
- visualization outputs.

## 5. Experimental Design

Purpose: explain the experimental comparisons before showing results.

Recommended subsections:

### 5.1 Main Benchmark

Compare methods across:

- systems,
- patient counts,
- seeds,
- metrics.

Main artifacts:

- Table 1: 500-patient benchmark summary,
- Table 2: patient-count sweep,
- Figure 1: patient-depth response.

### 5.2 Trajectory-Level Behavior

Use representative trajectories to show how score sequences behave before bifurcation.

Main artifact:

- Figure 2: representative trajectory panels.

### 5.3 Training Behavior

Use training curves only if they support a concrete point about stability, convergence, or regime dependence.

Main artifact:

- Figure 3: representative training curves.

### 5.4 Supplementary or Exploratory Comparisons

Older exploratory methods and ablations should be treated carefully.

Suitable supplement content:

- classical indicators,
- EWS augmentation experiments,
- null-training experiments,
- additional observer variants,
- high-noise or stress-test runs if they are not part of the finalized main benchmark.

Do not mix exploratory single-seed results into the main claims unless they match the finalized data protocol.

## 6. Results

Purpose: report the evidence in the same order as the experimental design.

Recommended subsections:

### 6.1 No Single Observer Dominates Across Systems and Metrics

Start with the full benchmark result. This is the central claim and should appear before discussing method-specific wins.

Use:

- Table 1,
- Figure 1.

Interpretation:

`The benchmark shows system- and metric-dependent trade-offs rather than a universal ordering of methods.`

### 6.2 Patient Count Changes the Method Ranking

Discuss the patient-count sweep.

Use:

- Table 2,
- Figure 1.

Focus on whether performance trends are stable or sensitive to data depth.

### 6.3 Score Trajectories Reveal Regime-Specific Behavior

Use representative score trajectories to explain why aggregate metrics differ.

Use:

- Figure 2.

This section should connect model behavior to the underlying dynamics.

### 6.4 Training Curves and Optimization Behavior

Use this only if the training curves reveal something meaningful.

Use:

- Figure 3.

If the curves are not scientifically informative, move them to supplement.

### 6.5 Summary of Main Findings

End Results with a compact summary:

- one method may be strong on one system,
- another may be preferable on a different metric,
- the results support regime-aware observer selection.

Avoid discussion-style speculation here; save interpretation for Discussion.

## 7. Discussion

Purpose: interpret the results and explain why they matter.

Recommended subsections:

### 7.1 Regime Dependence Is the Main Finding

Explain that different bifurcation mechanisms create different detection requirements.

### 7.2 Why Spectral or Recurrent Structure Helps Only in Some Settings

Discuss the inductive bias carefully:

- useful when the dynamics match it,
- not expected to dominate every transition type,
- not a failure if it loses where the bias is mismatched.

### 7.3 Why Simple Baselines Remain Important

Explain that simple indicators can be strong because they encode direct dynamical signatures.

This strengthens the paper because it shows the comparison is not weak.

### 7.4 Practical Implications

State that observer choice should depend on:

- dynamical regime,
- tolerance for false positives,
- desired warning lead time,
- available data depth.

## 8. Limitations

Purpose: show reviewers that the study is honest and bounded.

Recommended points:

- synthetic systems only,
- limited set of observer variants in the finalized main benchmark,
- threshold calibration may affect detection-time conclusions,
- patient-count sweeps do not cover all possible data regimes,
- results should not be generalized to all domains without additional validation.

Do not hide limitations. Use them to make the claims precise.

## 9. Conclusion

Purpose: close with the central scientific message.

Recommended conclusion:

`Learned Kalman observers show useful but conditional early-warning behavior. Across fold, Hopf, and logistic systems, the preferred observer depends on the dynamical regime, the metric, and the available data. The study therefore supports regime-aware observer design rather than a universal-detector interpretation.`

## Back Matter

Include:

- Data Availability
- Code Availability
- Acknowledgements
- Author Contributions
- Conflicts of Interest
- References
- Supplementary Material

## Figure and Table Map

Main paper:

- Figure 1: patient-depth response across systems and metrics.
- Figure 2: representative score trajectories.
- Figure 3: representative training curves, if scientifically useful.
- Table 1: 500-patient benchmark summary.
- Table 2: patient-count sweep summary.
- Table 3: benchmark batch inventory, likely supplement unless journal space allows.

Supplement:

- additional exploratory methods,
- alternate thresholds or calibration checks,
- older ablations that are not part of the finalized main benchmark,
- full batch inventory if not in the main text.

## Writing Order

Recommended writing order:

1. Methods and benchmark definition.
2. Results.
3. Discussion.
4. Introduction.
5. Abstract.
6. Title and keywords.

This order is more reliable than writing the abstract first, because the exact claims should be determined by the finalized results.

## Claims to Use

Safe claims:

- `Performance is regime-dependent under the evaluated settings.`
- `No single reported observer variant dominates all systems and metrics.`
- `The learned observer variants expose different detection-time, EW-AUC, and false-positive trade-offs.`
- `Observer selection should be tied to dynamical structure and evaluation priority.`

Claims to avoid:

- `The proposed method is universally best.`
- `The proposed method is state of the art.`
- `The method is robust across domains.`
- `The method solves early-warning detection.`
- `The method generalizes to all bifurcations.`
