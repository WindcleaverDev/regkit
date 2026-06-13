"""Univariate outlier detection via the 1.5×IQR rule."""

from __future__ import annotations

import numpy as np
import pandas as pd


def flag_outliers(col: pd.Series) -> int:
    """Return the count of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    arr = col.dropna().to_numpy(dtype=float)
    if len(arr) < 4:
        return 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return int(((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)).sum())
