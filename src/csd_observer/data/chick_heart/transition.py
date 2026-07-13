from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression


def _gaussian_detrend(signal: np.ndarray, bandwidth: float) -> np.ndarray:
    trend = gaussian_filter1d(signal, sigma=bandwidth, mode="reflect")
    return signal - trend


def _get_return_map_slope(x: np.ndarray) -> float:
    y = x[1:].reshape(-1, 1)
    X = x[:-1].reshape(-1, 1)
    if len(y) < 2:
        return 0.0
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0, 0])


def detect_transitions(
    df: pd.DataFrame,
    *,
    bandwidth: int = 60,
    rolling_window: int = 10,
    slope_threshold: float = -0.95,
    consecutive_beats: int = 10,
) -> Dict[Tuple[int, str], int]:
    results: Dict[Tuple[int, str], int] = {}

    for (tsid, typ), group in df.groupby(["tsid", "type"]):
        group = group.sort_values("Beat number")
        ibi = group["IBI (s)"].values.astype(np.float64)

        residuals = _gaussian_detrend(ibi, bandwidth)

        slopes = np.full(len(residuals), np.nan)
        for i in range(len(residuals) - rolling_window + 1):
            window = residuals[i : i + rolling_window]
            slopes[i + rolling_window - 1] = _get_return_map_slope(window)

        below = slopes < slope_threshold
        count_consecutive = 0
        transition_beat = -1
        for i in range(len(below)):
            if below[i]:
                count_consecutive += 1
                if count_consecutive >= consecutive_beats:
                    transition_beat = i - rolling_window
                    break
            else:
                count_consecutive = 0

        results[(tsid, typ)] = transition_beat

    return results
