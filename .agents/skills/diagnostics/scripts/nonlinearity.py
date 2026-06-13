"""Partial residual data for spotting feature-level nonlinearity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def partial_residual_plots(model, X: pd.DataFrame) -> dict[str, dict]:
    """One entry per continuous feature: partial residual = residual + β·x.

    A feature has potential nonlinearity if a LOWESS line through these
    deviates visibly from linear; render.py plots them, the triage layer
    relies on the RESET test for the formal check.
    """
    resid = np.asarray(model.resid, dtype=float)
    out: dict[str, dict] = {}
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        # Skip dummies / constant columns — partial residuals are uninformative
        if len(np.unique(values)) <= 2:
            continue
        beta = float(model.params.get(col, 0.0))
        out[col] = {
            "x": values.tolist(),
            "partial_resid": (resid + beta * values).tolist(),
        }
    return out
