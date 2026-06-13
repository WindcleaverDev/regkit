"""CV alpha selection and regularisation path computation.

Returns everything fit.py needs to build the schema objects: the fitted model
at the chosen alpha, the full coefficient path, and the CV score curve.
Coefficients are always returned on the original (un-standardised) scale.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    Ridge,
    enet_path,
    lasso_path,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from regression_pack_core.schemas import CVCurve, RegularizationPath


def _inverse_scale(coef_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Map coefficients from standardised scale back to original feature scale."""
    return coef_scaled / scaler.scale_


def _intercept_orig(
    intercept_scaled: float, coef_orig: np.ndarray, scaler: StandardScaler
) -> float:
    return float(intercept_scaled - np.dot(coef_orig, scaler.mean_))


# ─── Alpha grids ──────────────────────────────────────────────────────────────


def _ridge_alpha_grid(n: int = 100) -> np.ndarray:
    return np.logspace(-4, 4, n)


def _lasso_alpha_grid(X_scaled: np.ndarray, y: np.ndarray, n: int = 100) -> np.ndarray:
    """Data-adaptive grid: from alpha_max (all zero) down 4 log-decades."""
    alpha_max = float(np.max(np.abs(X_scaled.T @ (y - y.mean()))) / len(y))
    return np.geomspace(alpha_max, alpha_max * 1e-4, n)


# ─── CV fitting ───────────────────────────────────────────────────────────────


def _fit_ridge_cv(
    X_scaled: np.ndarray,
    y: np.ndarray,
    alpha_grid: np.ndarray,
    cv_folds: int,
    rule: str,
) -> tuple[float, float | None, np.ndarray, np.ndarray]:
    """Return (selected_alpha, alpha_1se, cv_mean_r2s, cv_std_r2s) for ridge."""
    # Coarse grid for CV curve (30 points) to keep runtime fast
    n_cv = min(30, len(alpha_grid))
    cv_alphas = np.geomspace(alpha_grid.min(), alpha_grid.max(), n_cv)
    means, stds = [], []
    for alpha in cv_alphas:
        scores = cross_val_score(Ridge(alpha=alpha), X_scaled, y, cv=cv_folds, scoring="r2")
        means.append(float(scores.mean()))
        stds.append(float(scores.std()))

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    best_idx = int(np.argmax(means_arr))
    selected_alpha = float(cv_alphas[best_idx])

    alpha_1se = None
    if rule == "1se":
        threshold = means_arr[best_idx] - stds_arr[best_idx]
        # Largest alpha (most regularised) still within 1 SE of best
        candidates = np.where(means_arr >= threshold)[0]
        if len(candidates) > 0:
            alpha_1se = float(cv_alphas[candidates[-1]])
            selected_alpha = alpha_1se

    return selected_alpha, alpha_1se, cv_alphas, means_arr, stds_arr


def _fit_lasso_cv(
    X_scaled: np.ndarray,
    y: np.ndarray,
    alpha_grid: np.ndarray,
    cv_folds: int,
    l1_ratio: float,
    method: str,
    rule: str,
) -> tuple[float, float | None, np.ndarray, np.ndarray, np.ndarray]:
    """Return (selected_alpha, alpha_1se, cv_alphas, mean_r2s, std_r2s)."""
    if method == "lasso":
        cv = LassoCV(alphas=alpha_grid, cv=cv_folds, max_iter=10000)
    else:
        cv = ElasticNetCV(
            alphas=alpha_grid, l1_ratio=l1_ratio, cv=cv_folds, max_iter=10000
        )
    cv.fit(X_scaled, y)

    # mse_path_ shape: (n_alphas, n_folds) — sorted descending in alphas
    alphas = cv.alphas_[::-1]  # ascending order
    mse = cv.mse_path_[::-1]  # (n_alphas, n_folds)
    var_y = float(np.var(y, ddof=1)) or 1.0
    mean_r2 = 1.0 - mse.mean(axis=1) / var_y
    std_r2 = mse.std(axis=1) / var_y

    best_idx = int(np.argmax(mean_r2))
    selected_alpha = float(cv.alpha_)  # the CV-optimal alpha from the estimator

    alpha_1se = None
    if rule == "1se":
        threshold = mean_r2[best_idx] - std_r2[best_idx]
        candidates = np.where(mean_r2 >= threshold)[0]
        if len(candidates) > 0:
            alpha_1se = float(alphas[candidates[-1]])
            selected_alpha = alpha_1se

    return selected_alpha, alpha_1se, alphas, mean_r2, std_r2


# ─── Regularisation path ──────────────────────────────────────────────────────


def _path_ridge(
    X_scaled: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    alpha_grid: np.ndarray,
    feature_names: list[str],
) -> RegularizationPath:
    path_coefs = []
    for alpha in alpha_grid:
        m = Ridge(alpha=alpha).fit(X_scaled, y)
        path_coefs.append(_inverse_scale(m.coef_, scaler).tolist())
    return RegularizationPath(
        alphas=alpha_grid.tolist(),
        feature_names=feature_names,
        coefficients=path_coefs,
    )


def _path_lasso(
    X_scaled: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    alpha_grid: np.ndarray,
    l1_ratio: float,
    method: str,
    feature_names: list[str],
) -> RegularizationPath:
    if method == "lasso":
        alphas_out, coefs, _ = lasso_path(
            X_scaled, y, alphas=alpha_grid[::-1], max_iter=10000
        )
    else:
        alphas_out, coefs, _ = enet_path(
            X_scaled, y, alphas=alpha_grid[::-1], l1_ratio=l1_ratio, max_iter=10000
        )
    # coefs shape: (n_features, n_alphas), alphas_out descending
    coefs_orig = coefs / scaler.scale_[:, np.newaxis]
    return RegularizationPath(
        alphas=alphas_out[::-1].tolist(),
        feature_names=feature_names,
        coefficients=coefs_orig.T[::-1].tolist(),
    )


# ─── Public entry point ───────────────────────────────────────────────────────


def fit_regularized(
    X_scaled: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    feature_names: list[str],
    *,
    method: str,
    cv_folds: int = 5,
    l1_ratio: float = 0.5,
    user_alphas: list[float] | None = None,
    rule: str = "min",
) -> dict:
    """Fit regularised regression with CV alpha selection.

    Returns a dict with keys:
      selected_alpha, alpha_1se, cv_curve (CVCurve), path (RegularizationPath),
      model (fitted sklearn model), coef_orig (ndarray), intercept_orig (float).
    """
    n_alphas = 100

    # Alpha grid
    if user_alphas:
        alpha_grid = np.array(sorted(user_alphas))
    elif method == "ridge":
        alpha_grid = _ridge_alpha_grid(n_alphas)
    else:
        alpha_grid = _lasso_alpha_grid(X_scaled, y, n_alphas)

    # CV
    if method == "ridge":
        selected_alpha, alpha_1se, cv_alphas, mean_r2, std_r2 = _fit_ridge_cv(
            X_scaled, y, alpha_grid, cv_folds, rule
        )
    else:
        selected_alpha, alpha_1se, cv_alphas, mean_r2, std_r2 = _fit_lasso_cv(
            X_scaled, y, alpha_grid, cv_folds, l1_ratio, method, rule
        )

    cv_curve = CVCurve(
        alphas=cv_alphas.tolist(),
        mean_scores=np.clip(mean_r2, -1, 1).tolist(),
        std_scores=np.abs(std_r2).tolist(),
        scoring="r2",
        selected_alpha=selected_alpha,
        alpha_1se=alpha_1se,
    )

    # Refit at selected alpha on full data
    if method == "ridge":
        model = Ridge(alpha=selected_alpha).fit(X_scaled, y)
    elif method == "lasso":
        model = Lasso(alpha=selected_alpha, max_iter=10000).fit(X_scaled, y)
    else:
        model = ElasticNet(alpha=selected_alpha, l1_ratio=l1_ratio, max_iter=10000).fit(
            X_scaled, y
        )

    coef_orig = _inverse_scale(model.coef_, scaler)
    intercept_orig = _intercept_orig(float(model.intercept_), coef_orig, scaler)

    # Full regularisation path
    if method == "ridge":
        path = _path_ridge(X_scaled, y, scaler, alpha_grid, feature_names)
    else:
        path = _path_lasso(X_scaled, y, scaler, alpha_grid, l1_ratio, method, feature_names)

    # Exact zero mask from sklearn (Lasso/ElasticNet set coefs to exactly 0.0)
    coef_zero_mask = model.coef_ == 0.0

    return {
        "selected_alpha": selected_alpha,
        "alpha_1se": alpha_1se,
        "cv_curve": cv_curve,
        "path": path,
        "model": model,
        "coef_orig": coef_orig,
        "intercept_orig": intercept_orig,
        "coef_zero_mask": coef_zero_mask,
    }
