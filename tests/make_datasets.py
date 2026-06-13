"""Export famous regression datasets to tests/data/ as CSVs.

All datasets ship inside packages this project already depends on
(scikit-learn, statsmodels, plotly), so this runs offline and is fully
reproducible — no downloads, no API keys.

Linear / diagnostic datasets:
- tips          — mixed numeric + categorical predictors (one-hot encoding)
- diabetes      — 10-feature continuous regression; canonical Lasso example
- engel         — textbook heteroscedasticity (Breusch-Pagan FAIL)
- gapminder     — strong nonlinearity (Ramsey RESET FAIL, log remediation)
- statecrime    — small-n influence analysis (named outliers)
- macro         — quarterly time series (Durbin-Watson autocorrelation)
- longley       — severe collinearity benchmark (7 features, n=16; theory only)

Binary classification datasets (for logistic-regression skill):
- breast_cancer — 30 tumour-measurement features, malignant/benign, n=569
- affairs       — fair (1978) marital-affairs survey binarised, n=6366
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer, load_diabetes

OUT_DIR = Path(__file__).parent / "data"


# ─── Linear / diagnostic ─────────────────────────────────────────────────────


def tips() -> pd.DataFrame:
    """Restaurant tips (Bryant & Smith 1995), n=244. Target: tip."""
    return px.data.tips()


def diabetes() -> pd.DataFrame:
    """Diabetes progression (Efron et al. 2004), n=442, raw units. Target: progression.

    Ten baseline variables (age, sex, bmi, bp, s1-s6 serum measures) predict
    disease progression one year later. The serum variables are strongly
    collinear (s1/s2 = total/LDL cholesterol; s3/s4 = HDL/TCH ratio).
    Canonical benchmark for Ridge and Lasso: Lasso typically selects bmi, s5,
    bp; Ridge gives all features non-zero weights.
    """
    bunch = load_diabetes(as_frame=True, scaled=False)
    return bunch.frame.rename(columns={"target": "progression"})


def engel() -> pd.DataFrame:
    """Engel's 1857 Belgian household food expenditure, n=235. Target: foodexp."""
    return sm.datasets.engel.load_pandas().data


def gapminder_2007() -> pd.DataFrame:
    """Gapminder 2007 cross-section, n=142. Target: lifeExp."""
    df = px.data.gapminder()
    df = df[df["year"] == 2007][["country", "continent", "lifeExp", "pop", "gdpPercap"]]
    return df.reset_index(drop=True)


def statecrime() -> pd.DataFrame:
    """US state crime (2009 ACS), n=51. Target: violent."""
    data = sm.datasets.statecrime.load_pandas().data
    return data.reset_index()  # keep the state name as a column


def macro() -> pd.DataFrame:
    """US quarterly macro data 1959-2009 (statsmodels), n=203. Target: realcons."""
    return sm.datasets.macrodata.load_pandas().data


def longley() -> pd.DataFrame:
    """Longley (1967) US macroeconomic data, n=16. Target: TOTEMP.

    Famously severe multicollinearity — every pair of predictors is nearly
    collinear. VIF values are in the hundreds. Too small for the fit CLI
    (MIN_ROWS=30) but useful for unit-testing regularisation path logic.
    """
    return sm.datasets.longley.load_pandas().data


# ─── Binary classification ────────────────────────────────────────────────────


def breast_cancer() -> pd.DataFrame:
    """Wisconsin Breast Cancer Diagnostic (UCI/sklearn), n=569. Target: malignant.

    30 numeric features derived from fine needle aspirate images: 10 cell-nucleus
    measurements (radius, texture, perimeter, area, smoothness, compactness,
    concavity, concave_points, symmetry, fractal_dimension) each summarised as
    mean, SE, and worst across the image.

    Target encoding: malignant=1, benign=0.
    sklearn's default encodes benign=1, malignant=0; we invert here so that
    the "positive class" is the clinically important one (cancer present).

    Positive rate: 37.3% (212 malignant, 357 benign).
    Expected AUC: ~0.99 with all 30 features (demonstrates the model is well-
    calibrated rather than an interesting selection problem). More instructive
    to use with Lasso to identify the minimal predictive feature set.
    """
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # Clean column names: spaces → underscores
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # Invert target: sklearn 0=malignant, 1=benign → we want malignant=1
    df["malignant"] = (df["target"] == 0).astype(int)
    df.drop(columns=["target"], inplace=True)
    return df


def affairs() -> pd.DataFrame:
    """Fair (1978) marital-affairs survey, binarised, n=6,366. Target: had_affair.

    Features: rate_marriage (self-rated quality 1-5), age, yrs_married,
    children (number), religious (1=anti, 4=very), educ (years),
    occupation (Hollingshead scale 1-7).

    Target: had_affair = 1 if the respondent reported any extramarital affairs
    in the past year (32.3% positive rate).

    The raw `affairs` count is retained for reference but should NOT be used as
    a predictor (it would leak the outcome).

    Interesting modelling aspects: marital satisfaction, religiosity, and age
    have strong negative effects; the logistic interpretation of "one unit
    increase in religiosity reduces the probability of an affair by X pp" is
    intuitively clear.
    """
    df = sm.datasets.fair.load_pandas().data.copy()
    df["had_affair"] = (df["affairs"] > 0).astype(int)
    # Drop the raw count to avoid leaking the outcome
    df.drop(columns=["affairs"], inplace=True)
    return df


# ─── Runner ──────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, object] = {
        "tips.csv": tips,
        "diabetes.csv": diabetes,
        "engel.csv": engel,
        "gapminder_2007.csv": gapminder_2007,
        "statecrime.csv": statecrime,
        "macro.csv": macro,
        "longley.csv": longley,
        "breast_cancer.csv": breast_cancer,
        "affairs.csv": affairs,
    }
    for name, build in datasets.items():
        df = build()
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        print(f"✓ wrote {path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
