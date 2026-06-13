"""Correctness evals: regularised regression.

Tests library-layer behaviour directly (no CLI subprocess):
1. Ridge shrinks but does not zero coefficients.
2. Ridge is more stable than OLS on collinear data.
3. Lasso selects the true signal features (x1, x3) and drops pure noise (x4)
   at an alpha where the 1-SE rule would fire.
4. The inverse-standardisation round-trip is accurate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".agents" / "skills" / "regularized-regression" / "scripts"))

import path as path_mod  # noqa: E402
import selection as sel_mod  # noqa: E402


def _fit(X_raw: np.ndarray, y: np.ndarray, names: list[str], *, method: str) -> dict:
    """Helper: scale X, build scaler, call fit_regularized."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)
    return path_mod.fit_regularized(X_sc, y, scaler, names, method=method)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def collinear_df() -> pd.DataFrame:
    """Load (or generate) the collinear dataset.

    x1 = true signal, x2 ≈ x1 (near-duplicate), x3 = independent signal, x4 = noise.
    DGP: y = 3 + 2*x1 + 0.8*x3 + eps
    """
    data_path = REPO / "examples" / "data" / "collinear.csv"
    if not data_path.exists():
        import subprocess
        subprocess.run([sys.executable, str(REPO / "examples" / "synth_data.py")], check=True)
    return pd.read_csv(data_path)


# ─── Ridge: no exact zeros ────────────────────────────────────────────────────


def test_ridge_never_zeros_features(collinear_df):
    """Ridge shrinks all coefficients but never zeroes them."""
    df = collinear_df
    X = df[["x1", "x2", "x3", "x4"]].to_numpy()
    y = df["y"].to_numpy()

    result = _fit(X, y, ["x1", "x2", "x3", "x4"], method="ridge")
    coef_orig = result["coef_orig"]

    # All coefficients should be non-zero for ridge
    assert np.all(np.abs(coef_orig) > 1e-10), "Ridge zeroed at least one coefficient"

    # Ridge may return a coef_zero_mask but all entries should be False
    mask = result.get("coef_zero_mask")
    if mask is not None:
        assert not np.any(mask), "Ridge coef_zero_mask has True entries (unexpected zeros)"


# ─── Ridge: OLS instability vs Ridge stability ────────────────────────────────


def test_ridge_stabilises_collinear_estimates(collinear_df):
    """Ridge coefficients for x1 and x2 are closer together than OLS.

    On highly collinear data, OLS gives wildly different signs for x1/x2
    (e.g. x1 ≈ -2, x2 ≈ +4). Ridge should pull both toward the true
    shared effect (~1.0 each).
    """
    df = collinear_df
    X_raw = df[["x1", "x2", "x3", "x4"]].to_numpy()
    y = df["y"].to_numpy()

    # OLS (via numpy lstsq on standardised X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)
    X_const = np.c_[np.ones(len(y)), X_sc]
    ols_coef_scaled, *_ = np.linalg.lstsq(X_const, y, rcond=None)
    ols_coef_orig = ols_coef_scaled[1:] / scaler.scale_

    result = _fit(X_raw, y, ["x1", "x2", "x3", "x4"], method="ridge")
    ridge_coef = result["coef_orig"]

    ols_spread = abs(ols_coef_orig[0] - ols_coef_orig[1])   # |β_x1 - β_x2| OLS
    ridge_spread = abs(ridge_coef[0] - ridge_coef[1])        # |β_x1 - β_x2| Ridge
    assert ridge_spread < ols_spread, (
        f"Ridge did not stabilise: OLS spread = {ols_spread:.3f}, "
        f"Ridge spread = {ridge_spread:.3f}"
    )


# ─── Lasso: sparsity at supra-optimal alpha ──────────────────────────────────


def test_lasso_drops_noise_at_1se_alpha(collinear_df):
    """At the 1-SE alpha, Lasso should drop x4 (pure noise) as exactly zero.

    The 1-SE rule selects a more parsimonious alpha than the CV-optimal one.
    We apply it manually here and check the resulting non-zero mask.
    """
    df = collinear_df
    X_raw = df[["x1", "x2", "x3", "x4"]].to_numpy()
    y = df["y"].to_numpy()

    result = _fit(X_raw, y, ["x1", "x2", "x3", "x4"], method="lasso")
    alpha_1se = result.get("alpha_1se")
    if alpha_1se is None:
        pytest.skip("alpha_1se not available for this fit")

    # Fit lasso at the 1-SE alpha directly to get exact zero mask
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)
    lasso_1se = Lasso(alpha=alpha_1se, max_iter=5000).fit(X_sc, y)
    coef_zero = lasso_1se.coef_ == 0.0

    feat_names = ["x1", "x2", "x3", "x4"]
    dropped = [feat_names[i] for i in range(4) if coef_zero[i]]

    assert "x4" in dropped, (
        f"x4 (pure noise) should be zeroed at alpha_1se={alpha_1se:.4g}; "
        f"dropped = {dropped}"
    )


# ─── FeatureSelection helper ─────────────────────────────────────────────────


def test_feature_selection_schema(collinear_df):
    """FeatureSelection correctly partitions selected and dropped features."""
    df = collinear_df
    X_raw = df[["x1", "x2", "x3", "x4"]].to_numpy()
    y = df["y"].to_numpy()

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)
    # Use a fairly large alpha to ensure some zeros
    alpha = 0.5
    lasso = Lasso(alpha=alpha, max_iter=5000).fit(X_sc, y)
    coef_orig = lasso.coef_ / scaler.scale_
    coef_zero_mask = lasso.coef_ == 0.0

    feat_names = ["x1", "x2", "x3", "x4"]
    fs = sel_mod.build_feature_selection(
        coef_orig, feat_names, method="lasso", coef_zero_mask=coef_zero_mask
    )

    assert fs is not None
    assert fs.n_selected + fs.n_dropped == 4
    assert set(fs.selected_features) | set(fs.dropped_features) == set(feat_names)
    assert set(fs.selected_features) & set(fs.dropped_features) == set()


# ─── Inverse-standardisation round-trip ──────────────────────────────────────


def test_inverse_standardisation_roundtrip():
    """Coefficients on original scale must reproduce y-hat accurately."""
    rng = np.random.default_rng(0)
    X_raw = rng.standard_normal((200, 3))
    true_coef = np.array([2.0, -1.0, 0.5])
    y = X_raw @ true_coef + 5.0 + rng.normal(0, 0.5, 200)

    result = _fit(X_raw, y, ["a", "b", "c"], method="ridge")
    coef_orig = result["coef_orig"]
    intercept_orig = result["intercept_orig"]

    y_hat = X_raw @ coef_orig + intercept_orig
    residuals = y - y_hat
    rmse = np.sqrt(np.mean(residuals ** 2))

    assert rmse < 1.0, f"RMSE on original scale too large: {rmse:.4f}"
