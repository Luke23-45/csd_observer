# Neural baseline specification

The benchmark includes two learned baselines. Both emit a causal per-step
alarm probability and enter the same validation-null calibration,
persistence, censoring, and FPR governance path as non-learned methods.

## LSTM-AlarmNet

This is a one-way recurrent encoder with a per-step alarm head. It follows
the deep early-warning family evaluated by Bury et al., “Deep learning for
early warning signals of tipping points”, PNAS 118 (2021),
doi:10.1073/pnas.2106140118. The implementation is intentionally causal:
the recurrent state at `t` depends only on observations through `t`.

## TCN-AlarmNet

This is a dilated causal temporal-convolution encoder. Every convolution is
left-padded by its receptive-field width, so no future value enters a score.
The family is motivated by Bai, Kolter & Koltun, “An Empirical Evaluation
of Generic Convolutional and Recurrent Networks for Sequence Modeling”,
arXiv:1803.01271. It is used as a long-series architectural baseline, not
as a claim of state-of-the-art performance.

## Fairness and capacity

Both models use the registry bundle, replicate-level split, train-only
normalization, identical seeds, and fixed validation-null thresholding.
Hyperparameters are configuration values and are never selected from test
metrics. The default capacity is one 64-unit LSTM layer or three 32-channel
TCN levels. Checkpoints and training fingerprints are written through the
`OutputWriter`; no model constructs output paths itself.

Neither baseline is assigned fabricated results. Results exist only after a
configured run has produced them.
