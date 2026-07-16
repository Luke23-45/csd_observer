"""
Canonical constants for the CSD observer benchmark visualization.
"""

from __future__ import annotations

from typing import Dict, Tuple

PATIENT_COUNTS: Tuple[int, ...] = (100, 200, 300, 400, 500)
SYSTEMS: Tuple[str, ...] = ("fold", "hopf", "logistic")
METHODS: Tuple[str, ...] = ("Kalman-BCE", "Kalman-LSTM-Spec")

SEED_ORDER: Tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)

SYSTEM_DISPLAY_NAMES: Dict[str, str] = {
    "fold": "Fold",
    "hopf": "Hopf",
    "logistic": "Logistic",
}

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "Kalman-BCE": "Kalman-BCE",
    "Kalman-LSTM-Spec": "Kalman-LSTM-Spec",
}

METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "detection_time": "Detection Time",
    "ew_auc": "EW-AUC",
    "fpr": "FPR",
    "threshold": "Threshold",
    "n_epochs_trained": "Epochs Trained",
}

METHOD_COLORS: Dict[str, str] = {
    "Kalman-BCE": "#332288",       # indigo
    "Kalman-LSTM-Spec": "#44AA99", # teal
}

SYSTEM_COLORS: Dict[str, str] = {
    "fold": "#117733",     # green
    "hopf": "#882255",     # wine
    "logistic": "#999933", # olive
}

SUPPLEMENTAL_COLORS: Dict[str, str] = {
    "highlight": "#CC6677",
    "neutral": "#BBBBBB",
    "dark": "#333333",
    "sky": "#88CCEE",
}
