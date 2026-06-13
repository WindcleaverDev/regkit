"""Nonlinearity detection via LOWESS vs linear fit residual comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def detect_nonlinearity(x: pd.Series, y: pd.Series, *, frac: float = 0.6) -> bool:
    """Return True if LOWESS substantially beats a linear fit.

    Criterion: LOWESS RSS < 0.7 × linear RSS on the same data.
    Returns False for short series (n < 10) or degenerate inputs.
    """
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    if len(x_arr) < 10:
        return False

    coeffs = np.polyfit(x_arr, y_arr, 1)
    linear_pred = np.polyval(coeffs, x_arr)
    linear_rss = float(np.sum((y_arr - linear_pred) ** 2))
    if linear_rss == 0:
        return False

    smoothed = lowess(y_arr, x_arr, frac=frac, return_sorted=True)
    lowess_pred = np.interp(x_arr, smoothed[:, 0], smoothed[:, 1])
    lowess_rss = float(np.sum((y_arr - lowess_pred) ** 2))

    return lowess_rss < 0.7 * linear_rss


def nonlinearity_plot_data(x: pd.Series, y: pd.Series, *, frac: float = 0.6) -> dict:
    """Return raw points and LOWESS curve data for render.py."""
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean, y_clean = x_arr[mask], y_arr[mask]
    smoothed = lowess(y_clean, x_clean, frac=frac, return_sorted=True)
    return {
        "x": x_clean.tolist(),
        "y": y_clean.tolist(),
        "lowess_x": smoothed[:, 0].tolist(),
        "lowess_y": smoothed[:, 1].tolist(),
    }
