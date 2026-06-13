"""Bias/variance assessment via train/test split and k-fold cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split

from regression_pack_core.schemas import BiasVarianceReport

RANDOM_STATE = 42


def run(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_split: float = 0.2,
    cv_folds: int = 5,
    include_learning_curve: bool = False,
) -> BiasVarianceReport:
    """1. Train/test split, refit on train, score on test.
    2. K-fold CV on full data, mean and std of R².
    3. Verdict logic per spec; 4. one-sentence evidence.
    """
    X_arr = X.to_numpy(dtype=float)
    y_arr = np.asarray(y, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_split, random_state=RANDOM_STATE
    )
    lr = LinearRegression().fit(X_train, y_train)
    train_r2 = float(lr.score(X_train, y_train))
    test_r2 = float(lr.score(X_test, y_test))
    gap = train_r2 - test_r2

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(LinearRegression(), X_arr, y_arr, cv=kf, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    if gap > 0.15 and test_r2 < 0.5:
        verdict = "high_variance"
        evidence = (
            f"Train R² ({train_r2:.3f}) far exceeds test R² ({test_r2:.3f}, gap {gap:.3f}) — "
            "the model fits noise it cannot reproduce out of sample."
        )
    elif train_r2 < 0.3:
        verdict = "high_bias"
        evidence = (
            f"Train R² is only {train_r2:.3f} — the model underfits even the data "
            "it was trained on."
        )
    elif gap < 0.05 and cv_std < 0.1:
        # 0.1 rather than 0.05: 5-fold CV R² std runs ~0.05-0.1 even on clean
        # data at n in the hundreds; 0.05 misclassifies healthy fits.
        verdict = "good_fit"
        evidence = (
            f"Train/test gap is small ({gap:.3f}) and CV scores are stable "
            f"({cv_mean:.3f} ± {cv_std:.3f}) — generalisation looks healthy."
        )
    else:
        verdict = "inconsistent"
        evidence = (
            f"Results are mixed (gap {gap:.3f}, CV {cv_mean:.3f} ± {cv_std:.3f}) — "
            "no clear bias/variance pathology, but performance is not fully stable."
        )

    lc = None
    if include_learning_curve:
        sizes, train_scores, test_scores = learning_curve(
            LinearRegression(),
            X_arr,
            y_arr,
            cv=kf,
            train_sizes=np.linspace(0.2, 1.0, 8),
            scoring="r2",
        )
        lc = {
            "sizes": sizes.tolist(),
            "train_scores": train_scores.mean(axis=1).tolist(),
            "test_scores": test_scores.mean(axis=1).tolist(),
        }

    return BiasVarianceReport(
        train_r_squared=train_r2,
        test_r_squared=test_r2,
        cv_r_squared_mean=cv_mean,
        cv_r_squared_std=cv_std,
        gap=gap,
        verdict=verdict,
        evidence=evidence,
        learning_curve=lc,
    )
