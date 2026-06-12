"""Build the CoefficientRow table from a fitted statsmodels OLS model."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from regression_pack_core.schemas import CoefficientRow


def build_coefficient_table(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    *,
    standardize: bool = False,
    y: pd.Series | None = None,  # required if standardize=True
) -> list[CoefficientRow]:
    """Build a CoefficientRow per parameter. The intercept ('const') appears
    first. p-values from model.pvalues, CIs from model.conf_int() (95% default).
    If standardize, set standardized_coefficient = coef * X[f].std() / y.std().
    """
    if standardize and y is None:
        raise ValueError("y is required when standardize=True")

    conf_int = model.conf_int(alpha=0.05)
    rows: list[CoefficientRow] = []

    params = model.params
    ordered = ["const", *[p for p in params.index if p != "const"]] if "const" in params.index else list(params.index)

    for name in ordered:
        std_coef = None
        if standardize and name != "const" and name in X.columns:
            x_std = float(X[name].std(ddof=1))
            y_std = float(y.std(ddof=1))
            if y_std > 0:
                std_coef = float(params[name]) * x_std / y_std
        rows.append(
            CoefficientRow(
                feature=name,
                coefficient=float(params[name]),
                std_error=float(model.bse[name]),
                t_stat=float(model.tvalues[name]),
                p_value=float(model.pvalues[name]),
                ci_lower=float(conf_int.loc[name, 0]),
                ci_upper=float(conf_int.loc[name, 1]),
                standardized_coefficient=std_coef,
            )
        )
    return rows
