"""Pairwise Pearson correlation and Variance Inflation Factor computation."""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(X: pd.DataFrame) -> dict[str, float]:
    """VIF per column of X (must be numeric, already one-hot encoded).

    Requires at least 2 features; returns 1.0 for each feature when X has
    only one column (no collinearity possible).
    """
    if X.shape[1] < 2:
        return {col: 1.0 for col in X.columns}
    exog = X.to_numpy(dtype=float)
    result = {}
    for i, col in enumerate(X.columns):
        try:
            v = float(variance_inflation_factor(exog, i))
        except Exception:
            v = float("inf")
        result[col] = v
    return result


def correlation_matrix(X: pd.DataFrame) -> tuple[list[str], list[list[float]]]:
    """Pairwise Pearson correlation on numeric columns of X.

    Returns (column_names, matrix) where matrix[i][j] is the correlation
    between column i and column j, rounded to 4 decimal places.
    """
    numeric = X.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        names = list(numeric.columns)
        return names, [[1.0]] * len(names)
    corr = numeric.corr(numeric_only=True).to_numpy(dtype=float)
    names = list(numeric.columns)
    return names, [[round(float(v), 4) for v in row] for row in corr]
