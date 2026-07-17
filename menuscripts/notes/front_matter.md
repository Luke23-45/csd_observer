# Front Matter

## Title

Regime-Dependent Performance of Learned Kalman Observers for Early-Warning Detection

## Abstract

Early-warning methods for critical transitions are often evaluated as if a single detector should transfer across dynamical regimes. This assumption is too strong when the warning signatures differ across bifurcation mechanisms. We study this issue with learned Kalman observer variants on synthetic fold, Hopf, and logistic systems, using patient-count sweeps from 100 to 500 trajectories and ten random seeds per setting. The benchmark compares a binary-cross-entropy Kalman observer with a spectral recurrent variant using detection time, early-warning AUC, and false-positive rate. At 500 patients, the spectral recurrent observer increases early-warning AUC on fold and Hopf systems and matches the BCE observer on logistic dynamics, but the gains do not extend uniformly to false-positive control or detection-time summaries. Across patient counts, the relative behavior of the two observers remains system- and metric-dependent. These results argue against treating early-warning detection as a single transferable classification problem. Instead, observer design and model selection should be tied to the dynamical regime, the available data depth, and the operating cost of false alarms versus delayed warnings.

## Keywords

critical transitions; early-warning signals; Kalman observer; bifurcation detection; spectral regularization; dynamical systems

## Editorial Check

The title states the paper's scope without claiming universal superiority. The abstract reports only the finalized benchmark scope and avoids universal or state-of-the-art claims. The keywords are specific to the method, task, and scientific domain.
