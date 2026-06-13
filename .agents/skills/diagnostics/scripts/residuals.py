"""Residual data extraction for diagnostic plots."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.outliers_influence import OLSInfluence


def residual_data(model) -> dict:
    """Return fitted values, residuals, standardized residuals, leverage and
    Cook's distance — everything render.py needs for the diagnostic plots.
    """
    infl = OLSInfluence(model)
    resid = np.asarray(model.resid, dtype=float)
    return {
        "fitted": np.asarray(model.fittedvalues, dtype=float).tolist(),
        "residuals": resid.tolist(),
        "std_residuals": np.asarray(infl.resid_studentized_internal, dtype=float).tolist(),
        "leverage": np.asarray(infl.hat_matrix_diag, dtype=float).tolist(),
        "cooks_d": np.asarray(infl.cooks_distance[0], dtype=float).tolist(),
    }
