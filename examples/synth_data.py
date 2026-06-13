"""Generate synthetic test datasets for the regression pack smoke tests.

Writes three CSVs to examples/data/:

- clean.csv          — all assumptions satisfied; should diagnose clean
- heteroscedastic.csv — planted heteroscedasticity; homoscedasticity FAIL
- influential.csv     — one obvious leverage/Cook's D outlier at row 0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "data"


def make_clean(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eps = rng.normal(0, 1, n)
    y = 2 + 1.5 * x1 - 0.8 * x2 + eps
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def make_heteroscedastic(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # Error scale must be MONOTONE in x1, not symmetric (|x1|+0.5): Breusch-Pagan
    # regresses squared residuals linearly on the predictors, so a variance
    # pattern symmetric in x1 is invisible to it.
    eps = rng.normal(0, 1, n) * (0.4 + np.exp(0.8 * x1))
    y = 2 + 1.5 * x1 - 0.8 * x2 + eps
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def make_influential(rng: np.random.Generator, n: int = 200) -> pd.DataFrame:
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eps = rng.normal(0, 1, n)
    y = 2 + 1.5 * x1 - 0.8 * x2 + eps
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    # Plant a clear high-leverage + Cook's D outlier
    df.loc[0, "x1"] = 10.0
    df.loc[0, "y"] = -50.0
    return df


def make_collinear(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    """Highly collinear predictors to test regularisation.

    x1 is the true signal; x2 ≈ x1 (VIF >> 10, zero additional information).
    x3 is an independent signal. x4 is pure noise.
    Expected results:
    - OLS: unstable split of the x1 coefficient across x1 and x2
    - Ridge: stabilises estimates by averaging x1 and x2 to ≈ 1.0 each
    - Lasso (1-SE rule): zeros x2 and/or x4 — retains x1 and x3
    """
    x1 = rng.standard_normal(n)
    # x2 has a tiny unique component so the design matrix is not exactly singular
    x2 = x1 + rng.normal(0, 0.005, n)  # near-exact duplicate of x1 (corr ≈ 0.9999)
    x3 = rng.standard_normal(n)          # genuine independent signal
    x4 = rng.standard_normal(n)          # pure noise — no true effect
    eps = rng.normal(0, 0.5, n)
    y = 3.0 + 2.0 * x1 + 0.8 * x3 + eps
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})


def make_skewed(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    """Log-normal target — all positive, strongly right-skewed (skewness > 1.5).
    Pre-analysis must recommend log_transform; fitting on raw y should show
    heteroscedasticity; fitting on log(y) should diagnose clean.
    """
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # Total variance in log space ≈ 0.64 + 0.09 + 0.36 = 1.09 → skewness ≈ 5+
    eps = rng.normal(0, 0.6, n)
    log_y = 2 + 0.8 * x1 - 0.3 * x2 + eps
    y = np.exp(log_y)  # log-normal, all positive, strongly right-skewed
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def make_tips(rng: np.random.Generator, n: int = 300) -> pd.DataFrame:
    """Synthetic tips dataset for model-comparison checkpoint.

    DGP: tip = 0.8 + 0.10*total_bill + 0.30*size + eps
    total_bill ~ Gamma(shape=8, scale=5), size ~ {1,2,3,4,5,6}.
    Mild heteroscedasticity to make HC3 comparison meaningful.
    """
    total_bill = rng.gamma(shape=8, scale=5, size=n).clip(3, 80)
    size = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.05, 0.40, 0.30, 0.15, 0.07, 0.03])
    eps = rng.normal(0, 0.5 + 0.025 * total_bill, size=n)
    tip = 0.8 + 0.10 * total_bill + 0.30 * size + eps
    return pd.DataFrame({"total_bill": total_bill.round(2), "size": size, "tip": tip.round(2)})


def make_binary(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    """Binary outcome with a clear signal — AUC should be ~0.85.

    DGP: log-odds = -0.5 + 1.5*x1 - 1.2*x2 + 0.6*x3  (x4 is pure noise)
    Expected marginal effect signs: x1 positive, x2 negative, x3 positive.
    """
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)  # noise — should be near-zero
    log_odds = -0.5 + 1.5 * x1 - 1.2 * x2 + 0.6 * x3
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})


def main() -> None:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "clean.csv": make_clean(rng),
        "heteroscedastic.csv": make_heteroscedastic(rng),
        "influential.csv": make_influential(rng),
        "skewed.csv": make_skewed(rng),
        "collinear.csv": make_collinear(rng),
        "tips.csv": make_tips(rng),
        "binary.csv": make_binary(rng),
    }
    for name, df in datasets.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        print(f"✓ wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
