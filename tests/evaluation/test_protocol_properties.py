"""L2.7: persistence-protocol mathematical properties.

The protocol's guarantees, checked on synthetic arrays:

1. FKG anchor bracket: for i.i.d. null steps with per-step alarm
   probability p, the empirical per-window persistent-FPR stays at or
   below the analytic upper bound (positive association ⇒ bound).
2. Monotonicity: persistent-FPR and the anchors are non-increasing in
   ``k_persist``.
3. Ablation identity: at ``k_persist=1`` the persistence pipeline
   reduces exactly to the classic step-alarm pipeline.
4. Censoring: trajectories without a persistent pre-transition alarm
   are flagged, never averaged into the DT.
5. Causality: the persistent-alarm stream at time t depends only on
   scores ``s_{<=t}`` — perturbing the future changes nothing earlier.
"""

from __future__ import annotations

import numpy as np
import pytest

from csd_observer.evaluation.common.metrics import (
    compute_early_warning_auc,
    compute_false_positive_rate,
)
from csd_observer.evaluation.persistence.protocol import (
    compute_persistent_dts,
    compute_persistent_fpr,
    compute_persistent_trajectory_fpr,
    null_anchor_upper_bound,
    persistent_alarm_stream,
    trajectory_fpr_anchor,
)


def _iid_null_scores(n_traj: int, length: int, p: float, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.random((n_traj, length), dtype=np.float64)
    return scores, np.full(n_traj, length, dtype=np.int64)


def test_fkg_anchor_brackets_empirical_persistent_fpr() -> None:
    p = 0.05
    k = 5
    window = 50
    for seed in range(3):
        scores, lengths = _iid_null_scores(200, 120, p, seed=seed)
        threshold = float(np.quantile(scores, 1.0 - p))
        anchor = null_anchor_upper_bound(p, k, window)
        assert 0.0 < anchor < 1.0
        # The per-window FKG bound applies to trajectory-level
        # persistent-FPRs (unit = window/trajectory), not step rates.
        traj_fpr = compute_persistent_trajectory_fpr(scores, lengths, threshold, k)
        traj_anchor = trajectory_fpr_anchor(p, k, window, seq_lengths=lengths)
        assert 0.0 <= traj_fpr <= traj_anchor + 1e-9


def test_persistent_fpr_and_anchors_monotone_in_k() -> None:
    p = 0.05
    window = 50
    scores, lengths = _iid_null_scores(150, 100, p, seed=3)
    threshold = float(np.quantile(scores, 1.0 - p))
    rates = [compute_persistent_fpr(scores, lengths, threshold, k) for k in range(1, 8)]
    assert all(b <= a + 1e-9 for a, b in zip(rates, rates[1:], strict=False))
    anchors = [null_anchor_upper_bound(p, k, window) for k in range(1, 8)]
    assert anchors == sorted(anchors, reverse=True)


def test_k1_reduces_to_classic_pipeline() -> None:
    """At k_persist=1 the persistent stream equals the alarm stream, the
    persistent FPR equals the classic step FPR, and the persistent
    detection time equals the classic detection time."""
    rng = np.random.default_rng(11)
    scores = rng.random((40, 90), dtype=np.float64)
    lengths = np.full(40, 90, dtype=np.int64)
    threshold = 0.6
    stream = persistent_alarm_stream(scores, threshold, 1, seq_lengths=lengths)
    np.testing.assert_array_equal(stream, scores >= threshold)
    assert compute_persistent_fpr(scores, lengths, threshold, 1) == pytest.approx(
        compute_false_positive_rate(scores, lengths, threshold)
    )

    bif = np.full(40, 60.0)
    pos = np.ones(40, dtype=bool)
    leads, censored = compute_persistent_dts(scores, bif, pos, lengths, threshold, 1)
    from csd_observer.evaluation.common.metrics import compute_per_traj_dts

    classic = compute_per_traj_dts(scores, bif, pos, lengths, threshold)
    for lead, c, cl in zip(leads, censored, classic, strict=False):
        if np.isfinite(cl):
            assert lead == pytest.approx(cl) and c == 0.0
        else:
            assert c == 1.0


def test_censored_trajectories_are_flagged_not_averaged() -> None:
    scores = np.array([[0.1, 0.1, 0.1, 0.9, 0.9],   # persistent alarm at t=3
                       [0.1, 0.1, 0.1, 0.1, 0.1]])  # never alarms
    lengths = np.array([5, 5], dtype=np.int64)
    bif = np.array([5.0, 5.0])
    pos = np.ones(2, dtype=bool)
    leads, censored = compute_persistent_dts(scores, bif, pos, lengths, threshold=0.5, k_persist=2)
    assert leads[0] == pytest.approx(1.0)   # tau - t_first_persistent = 5 - 4
    assert censored[0] == 0.0
    assert np.isnan(leads[1]) and censored[1] == 1.0


def test_causality_future_perturbation_does_not_change_past_stream() -> None:
    scores = np.array([[0.9, 0.1, 0.9, 0.9, 0.1, 0.1],
                       [0.9, 0.1, 0.9, 0.9, 0.1, 0.1]], dtype=np.float64)
    lengths = np.full(2, 6, dtype=np.int64)
    base = persistent_alarm_stream(scores, 0.5, 2, seq_lengths=lengths)
    perturbed = scores.copy()
    perturbed[:, 4:] = 1.0  # future flips to alarm
    after = persistent_alarm_stream(perturbed, 0.5, 2, seq_lengths=lengths)
    np.testing.assert_array_equal(base[:, :4], after[:, :4])


def test_anchor_window_matches_early_start_delta_parameterization() -> None:
    """R0.2: the anchor window is the early-window length, so the anchor
    and the empirical EW window are directly comparable."""
    p, k = 0.05, 5
    short = null_anchor_upper_bound(p, k, 50)
    long = null_anchor_upper_bound(p, k, 100)
    assert long > short
    assert null_anchor_upper_bound(p, k, 100) == trajectory_fpr_anchor(
        p, k, 100, seq_lengths=np.array([80, 120])
    )  # max length 120 clipped to window 100


def test_early_start_delta_changes_ew_window() -> None:
    """R0.2/R0.6: a wider early window sees an early alarm that the
    default window misses — the delta knobs are wired through."""
    tau = 150.0
    early_alarm_t = 60  # inside [tau-200, tau-50), outside [tau-50, tau-145)
    sig = np.full((1, 200), 0.1)
    sig[0, early_alarm_t] = 0.9
    lens = np.array([200], dtype=np.int64)
    pos = np.array([True], dtype=bool)
    bifs = np.array([tau], dtype=np.float64)
    nulls = np.full((1, 200), 0.1)
    null_lens = np.array([200], dtype=np.int64)
    auc_narrow = compute_early_warning_auc(
        sig, bifs, pos, lens, nulls, null_lens,
        early_start_delta=50.0, early_end_delta=5.0,
    )
    auc_wide = compute_early_warning_auc(
        sig, bifs, pos, lens, nulls, null_lens,
        early_start_delta=200.0, early_end_delta=5.0,
    )
    assert auc_wide == pytest.approx(1.0)  # alarm in window -> max=0.9 > 0.1
    assert auc_narrow == pytest.approx(0.5)  # alarm outside -> max=0.1 == null
