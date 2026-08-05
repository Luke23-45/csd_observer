"""
Canonical constants for the CSD observer benchmark visualization.
"""

from __future__ import annotations

from typing import Dict, Tuple

PATIENT_COUNTS: Tuple[int, ...] = (100, 200, 300, 400, 500)
SYSTEMS: Tuple[str, ...] = ("fold", "hopf", "logistic")
METHODS: Tuple[str, ...] = (
    "Kalman-Spectral-Drift",
    "VAR-CSD",
    "AC1-CSD",
    "SKEW-CSD",
    "SRATIO-CSD",
    "DFA-CSD",
    "RETRATE-CSD",
    "DMD-CSD",
)

# New benchmark suite seeds are 0..n_seeds-1; legacy batches used 101..1010.
# Both are kept so mixed data roots still sort deterministically.
SEED_ORDER: Tuple[int, ...] = (0, 1, 2, 3, 4, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)

SYSTEM_DISPLAY_NAMES: Dict[str, str] = {
    "fold": "Fold",
    "hopf": "Hopf",
    "logistic": "Logistic",
}

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "Kalman-Spectral-Drift": "Kalman-Spectral-Drift",
    "VAR-CSD": "VAR-CSD",
    "AC1-CSD": "AC1-CSD",
    "SKEW-CSD": "SKEW-CSD",
    "SRATIO-CSD": "SRATIO-CSD",
    "DFA-CSD": "DFA-CSD",
    "RETRATE-CSD": "RETRATE-CSD",
    "DMD-CSD": "DMD-CSD",
}

METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "detection_time": "Detection Time",
    "ew_auc": "EW-AUC",
    "fpr": "FPR",
    "threshold": "Threshold",
    "n_epochs_trained": "Epochs Trained",
}

METHOD_COLORS: Dict[str, str] = {
    "Kalman-Spectral-Drift": "#332288", # indigo
    "VAR-CSD": "#117733",               # green
    "AC1-CSD": "#44AA99",               # teal
    "SKEW-CSD": "#88CCEE",              # sky
    "SRATIO-CSD": "#CC6677",            # rose
    "DFA-CSD": "#999933",               # olive
    "RETRATE-CSD": "#DDCC77",           # sand
    "DMD-CSD": "#882255",               # wine
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
