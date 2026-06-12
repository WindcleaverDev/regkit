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


def main() -> None:
    np.random.seed(42)
    rng = np.random.default_rng(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "clean.csv": make_clean(rng),
        "heteroscedastic.csv": make_heteroscedastic(rng),
        "influential.csv": make_influential(rng),
    }
    for name, df in datasets.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        print(f"✓ wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
