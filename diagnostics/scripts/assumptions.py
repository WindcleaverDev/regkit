"""Assumption tests for a fitted OLS model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from regression_pack_core.schemas import AssumptionCheck, Status


def linearity(model, X: pd.DataFrame, y: pd.Series) -> AssumptionCheck:
    """Ramsey RESET test. p < 0.05 → WARN ('missed nonlinearity'), p < 0.01 → FAIL."""
    reset = linear_reset(model, power=2, use_f=True)
    p = float(reset.pvalue)
    if p < 0.01:
        status, evidence = Status.FAIL, (
            f"Ramsey RESET rejects linearity strongly (p = {p:.3g}) — the model likely "
            "misses a nonlinear relationship."
        )
    elif p < 0.05:
        status, evidence = Status.WARN, (
            f"Ramsey RESET suggests possible missed nonlinearity (p = {p:.3g})."
        )
    else:
        status, evidence = Status.OK, (
            f"Ramsey RESET finds no evidence of missed nonlinearity (p = {p:.3g})."
        )
    return AssumptionCheck(
        name="linearity",
        status=status,
        test_name="Ramsey RESET",
        statistic=float(reset.fvalue),
        p_value=p,
        evidence=evidence,
    )


def homoscedasticity(model) -> AssumptionCheck:
    """Breusch-Pagan test. p < 0.05 → WARN, p < 0.01 → FAIL."""
    lm_stat, lm_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    p = float(lm_p)
    if p < 0.01:
        status, evidence = Status.FAIL, (
            f"Breusch-Pagan strongly rejects constant error variance (p = {p:.3g}) — "
            "residual spread changes with the predictors."
        )
    elif p < 0.05:
        status, evidence = Status.WARN, (
            f"Breusch-Pagan suggests possible heteroscedasticity (p = {p:.3g})."
        )
    else:
        status, evidence = Status.OK, (
            f"Breusch-Pagan finds no evidence of heteroscedasticity (p = {p:.3g})."
        )
    return AssumptionCheck(
        name="homoscedasticity",
        status=status,
        test_name="Breusch-Pagan",
        statistic=float(lm_stat),
        p_value=p,
        evidence=evidence,
    )


def normality_of_residuals(model) -> AssumptionCheck:
    """Shapiro-Wilk if n ≤ 5000, else Anderson-Darling. p < 0.05 → WARN.

    FAIL is reserved for extreme deviation — defer to the QQ plot for that
    judgment. Includes skewness and kurtosis in detail.
    """
    resid = np.asarray(model.resid, dtype=float)
    n = len(resid)
    skew = float(stats.skew(resid))
    kurt = float(stats.kurtosis(resid))  # excess kurtosis

    if n <= 5000:
        test_name = "Shapiro-Wilk"
        stat, p = stats.shapiro(resid)
        stat, p = float(stat), float(p)
    else:
        test_name = "Anderson-Darling"
        ad = stats.anderson(resid, dist="norm")
        stat = float(ad.statistic)
        # Approximate p from critical values (5% level index 2)
        p = 0.04 if stat > ad.critical_values[2] else 0.5

    if p < 0.05:
        status = Status.WARN
        evidence = (
            f"{test_name} rejects normality of residuals (p = {p:.3g}; "
            f"skewness {skew:.2f}, excess kurtosis {kurt:.2f}). Inspect the QQ plot — "
            "mild deviations matter little for large n."
        )
    else:
        status = Status.OK
        evidence = (
            f"{test_name} finds residuals consistent with normality (p = {p:.3g}; "
            f"skewness {skew:.2f}, excess kurtosis {kurt:.2f})."
        )
    return AssumptionCheck(
        name="normality_of_residuals",
        status=status,
        test_name=test_name,
        statistic=stat,
        p_value=p,
        evidence=evidence,
        detail={"skewness": skew, "kurtosis": kurt, "n": n},
    )


def independence(model) -> AssumptionCheck:
    """Durbin-Watson. < 1.5 or > 2.5 → WARN. < 1.0 or > 3.0 → FAIL.

    Only meaningful for time-ordered data — caveat included in evidence.
    """
    dw = float(durbin_watson(model.resid))
    caveat = "Note: only meaningful if rows are time-ordered."
    if dw < 1.0 or dw > 3.0:
        status = Status.FAIL
        evidence = f"Durbin-Watson = {dw:.2f} indicates strong autocorrelation in residuals. {caveat}"
    elif dw < 1.5 or dw > 2.5:
        status = Status.WARN
        evidence = f"Durbin-Watson = {dw:.2f} suggests possible autocorrelation. {caveat}"
    else:
        status = Status.OK
        evidence = f"Durbin-Watson = {dw:.2f} — no evidence of autocorrelation. {caveat}"
    return AssumptionCheck(
        name="independence",
        status=status,
        test_name="Durbin-Watson",
        statistic=dw,
        p_value=None,
        evidence=evidence,
    )


def no_multicollinearity(X: pd.DataFrame) -> AssumptionCheck:
    """VIF per feature (excluding intercept). max VIF > 5 → WARN, > 10 → FAIL."""
    if X.shape[1] < 2:
        return AssumptionCheck(
            name="no_multicollinearity",
            status=Status.OK,
            test_name="VIF",
            statistic=1.0,
            p_value=None,
            evidence="Single predictor — multicollinearity not applicable.",
            detail={"vif": {X.columns[0]: 1.0} if X.shape[1] == 1 else {}},
        )

    exog = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    vifs = {
        col: float(variance_inflation_factor(exog, i + 1)) for i, col in enumerate(X.columns)
    }
    max_vif = max(vifs.values())
    worst = max(vifs, key=vifs.get)

    if max_vif > 10:
        status = Status.FAIL
        evidence = f"Severe multicollinearity: max VIF = {max_vif:.1f} ({worst})."
    elif max_vif > 5:
        status = Status.WARN
        evidence = f"Moderate multicollinearity: max VIF = {max_vif:.1f} ({worst})."
    else:
        status = Status.OK
        evidence = f"No problematic multicollinearity (max VIF = {max_vif:.1f})."
    return AssumptionCheck(
        name="no_multicollinearity",
        status=status,
        test_name="VIF",
        statistic=max_vif,
        p_value=None,
        evidence=evidence,
        detail={"vif": vifs},
    )


def run_all(model, X: pd.DataFrame, y: pd.Series) -> list[AssumptionCheck]:
    return [
        linearity(model, X, y),
        homoscedasticity(model),
        normality_of_residuals(model),
        independence(model),
        no_multicollinearity(X),
    ]
