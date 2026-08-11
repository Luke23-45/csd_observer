"""Evaluation package: metric primitives + persistence-aware protocol."""

from csd_observer.evaluation.common.calibration import calibrate_threshold
from csd_observer.evaluation.common.metrics import (
    compute_detection_time,
    compute_early_warning_auc,
    compute_false_positive_rate,
    compute_null_metrics,
    compute_per_traj_dts,
)
from csd_observer.evaluation.persistence.governance import evaluate_method
from csd_observer.evaluation.persistence.protocol import (
    compute_persistent_detection_metrics,
    compute_persistent_dts,
    compute_persistent_ew_auc,
    compute_persistent_fpr,
    compute_persistent_trajectory_fpr,
    null_anchor_upper_bound,
    persistent_alarm_stream,
    run_lengths,
)

__all__ = [
    "calibrate_threshold",
    "compute_detection_time",
    "compute_early_warning_auc",
    "compute_false_positive_rate",
    "compute_null_metrics",
    "compute_per_traj_dts",
    "evaluate_method",
    "run_lengths",
    "persistent_alarm_stream",
    "compute_persistent_dts",
    "compute_persistent_detection_metrics",
    "compute_persistent_fpr",
    "compute_persistent_trajectory_fpr",
    "compute_persistent_ew_auc",
    "null_anchor_upper_bound",
]
