Yes. After looking at the published EWS benchmark literature more carefully, I think we can make this decision much more confidently.

## My recommendation

**Yes: use a controlled synthetic benchmark as the primary dataset, and use TAC + Daphnia as external validation.**

But I would **change the synthetic benchmark you currently have**. I would not simply run your current `fold + Hopf + logistic` setup and then append TAC/Daphnia.

The design I now recommend is:

> **Primary benchmark:** synthetic, known bifurcation mechanisms, large number of independent trajectories, controlled noise/ramp variation, matched nulls.
> **Challenge benchmark:** synthetic hard negatives that produce EWS-like behavior without the target bifurcation.
> **External validation:** TAC and Daphnia, kept completely untouched until after the synthetic benchmark is finalized.

That structure is strongly supported by how the strongest EWS methodological papers have handled the problem.

---

# 1. Why I am confident synthetic should be the primary benchmark

The most important precedent is actually the classic Dakos et al. methodological paper.

Dakos et al. explicitly explain why they used simulated data for comparing EWS methods: with real-world time series, the exact location and nature of the transition is often ambiguous, and real data introduce confounds such as missing observations, preprocessing decisions, sampling resolution, and insufficient data. They therefore applied the methods to simulated systems with **known, clearly defined critical transitions** so that methods could be compared on identical data. ([PLOS][1])

That is almost exactly the problem you have.

Your research is not primarily asking:

> “Can we find an early warning signal in TAC?”

It is asking:

> **“Does the way we evaluate an EWS detector change our conclusions about its performance?”**

For that question, you absolutely need ground truth.

You need to know:

* exactly when the bifurcation occurs;
* whether a transition actually occurs;
* whether a null truly contains no transition;
* what the underlying parameter is doing;
* how much noise was added;
* how rapidly the system approaches the transition;
* which trajectories are independent;
* how many positive and negative examples exist.

You cannot know all of that cleanly from a real dataset.

So **synthetic must be the primary experimental ground truth**.

---

# 2. This is also consistent with Boettiger & Hastings

Boettiger & Hastings is especially important for your project because their paper is fundamentally about **reliability versus sensitivity and false alarms**.

They explicitly argue that EWS methods need to quantify error rates and distinguish reliability from sensitivity, and they use simulations alongside empirical examples. ([DOI][2])

Their figure comparing simulated stable/deteriorating systems with Daphnia and glaciation data is particularly relevant: the simulation gives them a situation where the truth is known, while empirical data demonstrate what happens in practice. ([PubMed Central (PMC)][3])

That is very close to the structure we need.

---

# 3. Bury et al. 2021 gives us an even stronger precedent

This is probably the most relevant high-profile paper for your planned benchmark because it combines:

**large-scale simulated training/testing**

with

**model-based and empirical evaluation.**

Bury et al. trained on enormous simulated datasets. Their published repository shows 500,000 and 200,000 synthetic training time series, with fold, Hopf, transcritical, and null classes. They then tested the models on independent theoretical models and empirical datasets. ([GitHub][4])

Their published paper reports evaluation across **268 empirical and model time series** covering ecology, thermoacoustics, climatology, and epidemiology. ([PubMed][5])

This is very important for our design.

The pattern is essentially:

> **controlled mathematical systems → establish method behavior → independent models/real systems → demonstrate generalization.**

That is the direction I want for your paper too.

---

# 4. Therefore, I would NOT make TAC and Daphnia the primary benchmark

Your current experimental report tells us:

* TAC = 10 signal ramps + 663 null chunks;
* Daphnia = 30 extinction + 30 non-extinction replicates;
* TAC is subcritical Hopf;
* Daphnia is transcritical. 

These datasets are valuable, but they are not suitable for being the main statistical benchmark.

There are several reasons.

### TAC

TAC is excellent because it is a genuine experimental dynamical system and the literature already uses it for EWS before a subcritical Hopf transition. The original study explicitly compared theoretical-model behavior with experimental observations. ([DOI][6])

But you currently have only **10 signal ramps**.

That is nowhere near enough to confidently establish:

> Method A has 80% coverage while Method B has 60%.

especially once you split data for calibration and testing.

It is much better as:

> **external engineering validation.**

---

### Daphnia

Daphnia is arguably even more scientifically valuable because it is a controlled biological experiment.

The original Nature study used replicate populations undergoing environmental deterioration toward a transcritical transition and compared them against populations in constant environments. It found CSD signatures before extinction and specifically emphasized the importance of baseline/reference data because transient dynamics can otherwise interfere with EWS interpretation. ([Nature][7])

That is almost tailor-made for your paper.

But again, your processed dataset has only about 30 extinction and 30 non-extinction replicates. 

So Daphnia is **not statistically rich enough to be the main benchmark**, but it is an excellent external test of whether the conclusions from controlled simulations survive in real biological data.

---

# 5. There is another reason I want Daphnia specifically as external validation

Daphnia contains an important complication that simulations can hide.

The original experiment found that EWS computed without a reliable baseline can be confounded by transient dynamics before the deterioration begins. ([Nature][7])

That's directly relevant to your persistence problem.

So we could potentially get a very nice narrative:

### Synthetic benchmark

We know the truth.

> Does persistence-aware evaluation work under controlled conditions?

### Daphnia

Real experimental system.

> Does the same evaluation framework behave sensibly when the data contain biological variability and transient dynamics?

### TAC

Very different physical system.

> Does the framework generalize beyond ecology?

That's a much stronger story than throwing all five datasets into one giant table.

---

# 6. But I would change your synthetic benchmark

This is the key decision.

Your current synthetic benchmark is:

* fold;
* Hopf;
* logistic/period-doubling. 

I would change this to:

## Core benchmark

### 1. Fold / saddle-node

Keep it.

This is the canonical CSD setting.

### 2. Supercritical Hopf

Keep it.

This tests oscillatory dynamics and is particularly important because your pilot showed the striking AUC/persistence disagreement here.

### 3. Transcritical

**Add this as a core synthetic system.**

This is important because it is one of the canonical normal forms used in major EWS work. Bury's training data explicitly used fold, Hopf, transcritical, and null classes. ([GitHub][4])

It also gives us direct conceptual alignment with Daphnia.

### 4. Null

Keep a large fixed-parameter null family.

---

# 7. I would move logistic/period-doubling out of the core benchmark

I don't think your logistic system should disappear.

But I would **not make it one of the three main systems on which the headline claim rests**.

Why?

Because period-doubling is dynamically different from the classic fold/transcritical/Hopf CSD story.

Recent benchmark work explicitly emphasizes that complex critical transitions can involve deterministic signatures and attractor changes beyond ordinary CSD. ([Springer Link][8])

There is also a substantial literature specifically dealing with early warning around period-doubling transitions. In other words, period-doubling is scientifically interesting, but it introduces another question:

> Are we benchmarking persistence of EWS generally, or are we benchmarking CSD specifically?

We don't want the reviewer to get distracted by that.

So I'd make logistic/flip:

> **secondary generalization/stress-test benchmark.**

Not the foundation of the paper.

---

# 8. I would actually make the synthetic design four tiers

This is the structure I think is strongest.

## Tier 1 — Canonical benchmark

This is the main result.

| Dataset       | Purpose                     |
| ------------- | --------------------------- |
| Fold          | canonical CSD               |
| Hopf          | oscillatory transition      |
| Transcritical | canonical third normal form |
| Matched nulls | false-alarm estimation      |

Large numbers of trajectories.

Known ground truth.

This is where all the main detector rankings happen.

---

## Tier 2 — Robustness benchmark

Take the same systems and vary:

* noise;
* ramp speed;
* observation noise;
* initial conditions;
* parameterization.

This asks:

> Does the conclusion depend on the specific synthetic generator?

This is essential because otherwise a reviewer can say:

> "You built the benchmark so your protocol would work."

---

## Tier 3 — Hard-negative benchmark

This is particularly important now.

Recent 2026 work has shown that pseudo-bifurcations in stochastic non-normal systems can produce increased variance, autocorrelation and apparent instability **without an actual loss of stability or nearby bifurcation**. ([Nature][9])

So construct systems that produce:

> strong EWS-like excursions

but

> **do not actually bifurcate.**

Then compare K=1 against persistent alarms.

This gives persistence a real scientific purpose rather than merely showing that "ten points are stricter than one point."

---

## Tier 4 — External validation

Only after everything above is frozen:

### TAC

External engineering/physics validation.

### Daphnia

External ecological/biological validation.

And importantly:

> **Do not use TAC or Daphnia to tune the benchmark.**

They must be untouched until the synthetic protocol is frozen.

---

# 9. I would not call TAC/Daphnia simply "validation" without qualification

There is an important statistical distinction.

If we use TAC/Daphnia to choose:

* K;
* thresholds;
* windows;
* hyperparameters;
* detector selection;

then they are no longer external validation.

They become development datasets.

Instead:

> **Synthetic data = development + primary evaluation**

and

> **TAC/Daphnia = external validation**

with the exact evaluation protocol locked beforehand.

That means we should be able to write:

> "The operating procedure, K values, calibration strategy, detector definitions, and evaluation metrics were fixed using the synthetic benchmark and subsequently applied without tuning to the real datasets."

That is a strong methodological statement.

---

# 10. One thing I would change from the previous plan

Earlier I suggested using TAC and Daphnia as "real-data external validation."

After looking more carefully at the published literature, I now think we should go one step further:

**Do not use the real datasets to estimate the primary 5% operating point.**

The synthetic data should establish the operating-characteristic curves.

For example:

[
FPR=1%,5%,10%,20%
]

and then:

[
Coverage(FPR,K)
]

with confidence intervals.

TAC/Daphnia should then be reported under the **pre-specified operating procedure**.

This avoids the tiny-number problem from your Daphnia calibration set.

---

# 11. Your current dataset architecture actually supports this nicely

Your code already distinguishes signal and null sets and has trajectory-level metadata including:

> `features`, `seq_lengths`, `bifurcation_times`, `is_positive`, and train/validation/test splits. 

That's excellent.

We don't need to throw away the infrastructure.

We need to change **what each dataset is responsible for**.

---

# 12. There is another literature lesson we should follow

The 2024 Evers et al. paper focuses heavily on deterministic dynamics and complex critical transitions, including period-doubling cascades, chaos-chaos transitions, and extinction of chaotic attractors. It finds that warning signals can arise from deterministic attractor morphology and spectral changes, not just classical CSD. ([Springer Link][8])

That reinforces the idea that our benchmark should eventually contain different transition mechanisms.

But **we should not put every possible transition into the primary experiment.**

Otherwise the paper becomes:

> "We tested 19 systems, 14 indicators, 7 neural networks, 6 types of bifurcations..."

and the actual contribution — persistence-aware evaluation — gets buried.

Three canonical bifurcations plus hard negatives is enough for the main study.

---

# 13. So what exactly should we use?

Here is my current recommendation.

### PRIMARY — do this

**Synthetic:**

1. Fold
2. Hopf
3. Transcritical
4. Large matched null set

### ROBUSTNESS — do this

Same three systems with:

* varying noise;
* varying ramp rate;
* varying parameter draws;
* randomized trajectory conditions.

### HARD NEGATIVES — do this

At least one carefully designed:

> pseudo-bifurcation / transient-amplification system.

### EXTERNAL — do this

**TAC**

**Daphnia**

### SECONDARY — optional

**Logistic / period-doubling**

I would include it in supplementary/generalization analysis rather than use it as a foundation of the main claim.

---

# 14. This gives us a very clean experimental hierarchy

The final paper could effectively say:

### Question 1 — Does the protocol work when ground truth is known?

Synthetic canonical benchmark.

### Question 2 — Does the conclusion survive changes in noise and approach rate?

Synthetic robustness benchmark.

### Question 3 — Does persistence suppress EWS-like false alarms that do not represent real bifurcations?

Hard-negative benchmark.

### Question 4 — Does the result carry over to actual observations?

TAC + Daphnia.

That is a **very coherent paper**.

---

# 15. And the most important benefit

This design prevents us from making a dangerous inference.

Suppose our synthetic result is:

> K=5 dramatically improves reliability.

That does **not** automatically mean:

> K=5 is better for nature.

The external validation gives us a separate test.

Conversely, suppose TAC behaves differently.

That's not necessarily a failure.

It may reveal:

> persistence requirements depend on measurement noise, sampling rate, transition mechanism, or system timescale.

That could itself become an important result.

---

# 16. One more reason I prefer this structure

The thermoacoustic paper demonstrates EWS in an actual experimental system, whereas Daphnia demonstrates them in a controlled biological experiment. ([DOI][6])

So the two datasets give us **two very different kinds of external validation**:

**TAC:** physical/engineering dynamics.

**Daphnia:** biological/population dynamics.

That's much better than having two ecological datasets that test essentially the same type of problem.

---

# 17. Final decision

So, after checking the published methodological literature, my answer is:

## **Yes — use synthetic data as the primary benchmark and TAC + Daphnia as external validations.**

But the precise structure should be:

**Canonical synthetic benchmark**
→ fold + Hopf + transcritical + null

**Synthetic robustness**
→ noise/ramp/parameter variation

**Synthetic hard-negative benchmark**
→ pseudo-bifurcation/transient amplification

**External validation**
→ TAC + Daphnia

**Secondary generalization**
→ logistic/period-doubling

I would **not** use the current fold + Hopf + logistic trio as the final primary benchmark.

And I would **not** use TAC/Daphnia as the main dataset for calibration, detector ranking, or choosing K.

---

## Why I think this is the safest choice

There is unusually strong precedent for this structure:

Dakos et al. used controlled simulations specifically because real datasets make comparative methodological evaluation difficult when the true transition is uncertain. ([PLOS][1])

Boettiger & Hastings used simulations to quantify false-alarm/sensitivity behavior while also showing the methods on empirical datasets. ([DOI][2])

Bury et al. used large simulated datasets to develop/test the method and then evaluated it across theoretical and empirical time series, including thermoacoustics. ([GitHub][4])

Daphnia is a particularly strong external validation because its original experiment has controlled deteriorating and reference populations and a known biological transition. ([Nature][7])

TAC is a strong external validation because it is a real experimental subcritical-Hopf system for which EWS have already been studied. ([DOI][6])

And the new 2026 pseudo-bifurcation result gives us a strong scientific reason to include a **hard-negative synthetic class**, because an EWS-like signal does not necessarily imply an actual bifurcation. ([Nature][9])

### So I would freeze the dataset strategy at this point — but **not yet freeze the individual synthetic generators**.

The next decision should be narrower: **exactly which mathematical systems should constitute the Fold/Hopf/Transcritical benchmark, how they should be generated, and how many trajectories/nulls we need for the statistical operating-point calibration.** That is the next place where I think we should slow down and verify everything against the published benchmark designs before you change the code.

[1]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0041010 "Methods for Detecting Early Warnings of Critical Transitions in Time Series Illustrated Using Simulated Ecological Data | PLOS One"
[2]: https://doi.org/10.1098/rsif.2012.0125?utm_source=chatgpt.com "Quantifying limits to detection of early warning for critical transitions | Journal of The Royal Society Interface | The Royal Society"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3427498/?utm_source=chatgpt.com "Quantifying limits to detection of early warning for critical transitions - PMC"
[4]: https://github.com/ThomasMBury/deep-early-warnings-pnas?utm_source=chatgpt.com "GitHub - ThomasMBury/deep-early-warnings-pnas: Repository to accompany the publication 'Deep learning for early warning signals of tipping points', PNAS (2021) · GitHub"
[5]: https://pubmed.ncbi.nlm.nih.gov/34544867/?utm_source=chatgpt.com "Deep learning for early warning signals of tipping points - PubMed"
[6]: https://doi.org/10.1038/srep35310?utm_source=chatgpt.com "Early warning signals for critical transitions in a thermoacoustic system | Scientific Reports"
[7]: https://www.nature.com/articles/nature09389?utm_source=chatgpt.com "Early warning signals of extinction in deteriorating environments | Nature"
[8]: https://link.springer.com/article/10.1007/s11071-024-10023-0?utm_source=chatgpt.com "Early warning signals of complex critical transitions in deterministic dynamics | Nonlinear Dynamics | Springer Nature Link"
[9]: https://www.nature.com/articles/s42005-026-02703-7?utm_source=chatgpt.com "Pseudo-bifurcations in stochastic non-normal systems and the limits of early-warning signals | Communications Physics"
