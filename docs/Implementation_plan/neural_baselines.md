# Neural baseline specification

The benchmark includes three learned baselines. All emit a causal per-step
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

## PatchTST-AlarmNet

This is the patch-based transformer baseline (plan §6.4, family b3). It
follows the patch-tokenisation scheme of Nie et al., “A Time Series is
Worth 64 Words: Long-term Forecasting with Transformers”, ICLR 2023
(PatchTST): the series is cut into fixed-length patches, each patch is
linearly embedded, and a transformer encoder processes the patch sequence.

Two adaptations keep the head *strictly causal* (the persistence protocol
rejects future leakage):

* self-attention over patches is causally masked (patch `i` attends only
  to patches `j ≤ i`);
* a score at step `t` is emitted only from the representation of the
  *latest patch completed at or before* `t`; steps before the first patch
  completes emit the neutral logit 0 (p = 0.5).

The alarm signal is therefore piecewise-constant between patch
completions and carries an architectural latency of up to one patch
window. That latency is an honest property of the representation, not a
calibration artifact, and must be kept in mind when comparing
detection-time numbers across methods (LSTM/TCN emit per-step).

The positional encoding is sinusoidal and computed for the actual patch
count, so sequence length is not capped by a learned position table.
Default capacity (`d_model=64`, `n_heads=2`, `n_layers=2`, `patch_len=16`,
non-overlapping) sits in the same budget as the LSTM/TCN baselines.

## Fairness and capacity

All three models use the registry bundle, replicate-level split,
train-only normalization, identical seeds, and fixed validation-null
thresholding. Hyperparameters are configuration values and are never
selected from test metrics. The default capacity is one 64-unit LSTM
layer, three 32-channel TCN levels, or a two-layer 64-unit patch
transformer. Checkpoints and training fingerprints are written through
the `OutputWriter`; no model constructs output paths itself.

Neither baseline is assigned fabricated results. Results exist only after a
configured run has produced them.
