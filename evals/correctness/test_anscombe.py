"""Correctness eval: Anscombe's quartet.

Four datasets with (nearly) identical regression lines, R² and correlation —
but radically different diagnostics. A pack that reports only fit statistics
treats them as the same model; the diagnostics layer must tell them apart.

The quartet has n=11 (below the fit CLI's 30-row minimum), so this eval
exercises the diagnostic modules directly through the library layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "diagnostics" / "scripts"))

import assumptions as assumptions_mod  # noqa: E402
import influence as influence_mod  # noqa: E402

# Anscombe (1973), Graphs in Statistical Analysis, table 1
X_123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ANSCOMBE = {
    "I": (X_123, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (X_123, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (X_123, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": (
        [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
    ),
}


@pytest.fixture(scope="module")
def fits() -> dict:
    out = {}
    for key, (x, y) in ANSCOMBE.items():
        X = pd.DataFrame({"x": x}, dtype=float)
        ys = pd.Series(y, dtype=float, name="y")
        model = sm.OLS(ys, sm.add_constant(X)).fit()
        out[key] = (model, X, ys)
    return out


def test_same_regression_line(fits):
    """All four datasets produce ≈ the same slope, intercept and R²."""
    for key, (model, _, _) in fits.items():
        assert model.params["const"] == pytest.approx(3.0, abs=0.01), key
        assert model.params["x"] == pytest.approx(0.5, abs=0.003), key
        assert model.rsquared == pytest.approx(0.666, abs=0.01), key


def test_dataset_ii_fails_linearity(fits):
    """II is a perfect parabola — RESET must reject linearity; I must pass."""
    model_i, X_i, y_i = fits["I"]
    model_ii, X_ii, y_ii = fits["II"]
    assert assumptions_mod.linearity(model_i, X_i, y_i).status.value == "ok"
    assert assumptions_mod.linearity(model_ii, X_ii, y_ii).status.value == "fail"


def test_dataset_iii_has_cooks_outlier(fits):
    """III is a tight line plus one gross outlier (row 2)."""
    model, X, _ = fits["III"]
    report = influence_mod.run(model, X)
    assert any(p.row_index == 2 for p in report.cooks_d_outliers)


def test_dataset_iv_has_extreme_leverage(fits):
    """IV's slope exists only because of the single x=19 point (row 7)."""
    model, X, _ = fits["IV"]
    report = influence_mod.run(model, X)
    assert any(p.row_index == 7 for p in report.high_leverage)
    assert report.max_leverage > 0.9
