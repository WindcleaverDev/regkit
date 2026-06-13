"""Correctness evals: logistic regression.

Tests library-layer behaviour directly (no CLI subprocess):
1. AME signs match the planted DGP (x1+, x2-, x3+, x4≈0).
2. Odds ratios are consistent with log-odds coefficients.
3. AUC is above a meaningful threshold (> 0.70) for the planted DGP.
4. Calibration Brier score is finite and in [0, 1].
5. ROC thresholds are all in [0, 1] (no np.inf leakage).

DGP: log-odds = -0.5 + 1.5*x1 - 1.2*x2 + 0.6*x3, x4 = noise.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[2]
LOGISTIC_SCRIPTS = REPO / ".agents" / "skills" / "logistic-regression" / "scripts"
sys.path.insert(0, str(LOGISTIC_SCRIPTS))

from calibration import compute_calibration  # noqa: E402
from coefficients import build_odds_ratio_table  # noqa: E402
from marginal import compute_marginal_effects  # noqa: E402
from roc import compute_roc  # noqa: E402


@pytest.fixture(scope="module")
def binary_fit():
    """Fit statsmodels Logit on binary.csv and return (model, y, y_pred_prob)."""
    data_path = REPO / "examples" / "data" / "binary.csv"
    if not data_path.exists():
        import subprocess
        subprocess.run([sys.executable, str(REPO / "examples" / "synth_data.py")], check=True)

    df = pd.read_csv(data_path)
    feature_names = ["x1", "x2", "x3", "x4"]
    X_df = df[feature_names].copy()
    X_df.insert(0, "const", 1.0)
    y = df["y"].to_numpy()

    model = sm.Logit(y, X_df).fit(maxiter=200, disp=False)
    y_pred = model.predict()
    return model, y, y_pred


# ─── AME signs ───────────────────────────────────────────────────────────────


def test_ame_signs_match_dgp(binary_fit):
    """AMEs must have the correct sign: x1+, x2-, x3+."""
    model, y, _ = binary_fit
    mes = compute_marginal_effects(model)
    ame_by_feat = {m.feature: m.ame for m in mes}

    assert ame_by_feat["x1"] > 0, f"x1 AME should be positive; got {ame_by_feat['x1']:.4f}"
    assert ame_by_feat["x2"] < 0, f"x2 AME should be negative; got {ame_by_feat['x2']:.4f}"
    assert ame_by_feat["x3"] > 0, f"x3 AME should be positive; got {ame_by_feat['x3']:.4f}"


def test_x4_ame_small(binary_fit):
    """x4 is pure noise; its AME should be close to 0."""
    model, _, _ = binary_fit
    mes = compute_marginal_effects(model)
    x4_ame = next(m.ame for m in mes if m.feature == "x4")
    assert abs(x4_ame) < 0.10, f"x4 AME too large for noise variable: {x4_ame:.4f}"


# ─── Odds ratios ─────────────────────────────────────────────────────────────


def test_odds_ratios_consistent_with_log_odds(binary_fit):
    """exp(log_odds_coefficient) == odds_ratio within float tolerance."""
    model, _, _ = binary_fit
    coefs = build_odds_ratio_table(model)
    for c in coefs:
        expected_or = math.exp(c.log_odds_coefficient)
        assert c.odds_ratio == pytest.approx(expected_or, rel=1e-5), (
            f"Inconsistent OR for {c.feature}: "
            f"exp({c.log_odds_coefficient:.4f}) = {expected_or:.4f} but OR = {c.odds_ratio:.4f}"
        )


def test_x1_or_greater_than_one(binary_fit):
    """x1 has a positive effect in the DGP, so OR > 1."""
    model, _, _ = binary_fit
    coefs = build_odds_ratio_table(model)
    x1_or = next(c.odds_ratio for c in coefs if c.feature == "x1")
    assert x1_or > 1.0, f"x1 OR should be > 1; got {x1_or:.4f}"


# ─── AUC ─────────────────────────────────────────────────────────────────────


def test_auc_above_threshold(binary_fit):
    """AUC should be > 0.70 for the planted DGP with clear signal."""
    _, y, y_pred = binary_fit
    roc = compute_roc(y, y_pred)
    assert roc.auc > 0.70, f"AUC below threshold: {roc.auc:.4f}"


# ─── ROC thresholds ──────────────────────────────────────────────────────────


def test_roc_thresholds_finite(binary_fit):
    """All ROC thresholds must be in [0, 1] — no np.inf leakage."""
    _, y, y_pred = binary_fit
    roc = compute_roc(y, y_pred)
    assert all(0.0 <= t <= 1.0 for t in roc.thresholds), (
        f"ROC thresholds contain out-of-range value: "
        f"min={min(roc.thresholds):.4f}, max={max(roc.thresholds):.4f}"
    )


# ─── Calibration ─────────────────────────────────────────────────────────────


def test_calibration_brier_in_range(binary_fit):
    """Brier score must be in (0, 1)."""
    _, y, y_pred = binary_fit
    cal = compute_calibration(y, y_pred)
    assert 0.0 < cal.brier_score < 1.0, f"Brier score out of range: {cal.brier_score}"


def test_calibration_bins_populated(binary_fit):
    """At least 5 calibration bins should have positive observation counts."""
    _, y, y_pred = binary_fit
    cal = compute_calibration(y, y_pred)
    assert len(cal.bin_counts) >= 5, f"Expected >= 5 bins; got {len(cal.bin_counts)}"
    assert all(n > 0 for n in cal.bin_counts), "At least one bin is empty"
