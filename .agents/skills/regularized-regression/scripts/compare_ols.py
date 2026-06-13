"""Comparison of regularised model vs OLS on the same data.

OLS is used as a reference only — not a competitor. The dict is embedded in
RegularizedRegressionReport.comparison_to_ols.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score


def compare_to_ols(
    X_orig: np.ndarray,
    y: np.ndarray,
    coef_reg: np.ndarray,
    intercept_reg: float,
    feature_names: list[str],
) -> dict:
    """Fit OLS and compare key statistics.

    Returns a dict with:
      - ols_r2: OLS in-sample R²
      - reg_r2: regularised in-sample R²
      - ols_coef: {feature: ols_coef} mapping (original scale)
      - reg_coef: {feature: reg_coef} mapping
      - coef_shrinkage: mean absolute shrinkage |β_ols - β_reg|
      - ols_n_features: number of OLS features (always = n columns)
      - notes: list of string observations
    """
    X_with_const = sm.add_constant(X_orig, has_constant="add")
    try:
        ols = sm.OLS(y, X_with_const).fit()
        ols_coef_vals = ols.params[1:]  # exclude intercept
        ols_r2 = float(ols.rsquared)
    except Exception:
        # OLS fails on singular matrix — no comparison possible
        return {"error": "OLS failed (singular matrix — data likely collinear)"}

    y_pred_reg = X_orig @ coef_reg + intercept_reg
    reg_r2 = float(r2_score(y, y_pred_reg))

    ols_coef_dict = {name: float(v) for name, v in zip(feature_names, ols_coef_vals, strict=True)}
    reg_coef_dict = {name: float(v) for name, v in zip(feature_names, coef_reg, strict=True)}

    shrinkage = float(np.mean(np.abs(ols_coef_vals - coef_reg)))

    notes: list[str] = []
    if reg_r2 > ols_r2 + 0.01:
        notes.append("Regularised model has higher R² than OLS (data may be overfit by OLS).")
    elif ols_r2 > reg_r2 + 0.02:
        notes.append(
            "OLS has higher in-sample R²; this is expected — regularisation trades fit for "
            "generalisation. Compare out-of-sample scores."
        )
    else:
        notes.append("Regularised and OLS in-sample R² are close.")

    return {
        "ols_r2": round(ols_r2, 6),
        "reg_r2": round(reg_r2, 6),
        "ols_coef": ols_coef_dict,
        "reg_coef": reg_coef_dict,
        "coef_shrinkage_mean": round(shrinkage, 6),
        "notes": notes,
    }
