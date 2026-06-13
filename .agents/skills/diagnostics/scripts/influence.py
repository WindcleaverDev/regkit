"""Influence analysis: leverage, Cook's distance, studentized residuals, DFFITS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import OLSInfluence

from regression_pack_core.schemas import InfluencePoint, InfluenceReport

# The 2(k+1)/n and 4/n cutoffs flag a roughly constant *fraction* of rows, so
# on large data the flagged lists grow without bound. Statistics are computed
# on the full data; only the stored per-point details are capped, keeping the
# most influential points. Totals are always reported via n_* and summary.
MAX_STORED_POINTS = 50


def run(model, X: pd.DataFrame) -> InfluenceReport:
    """Flag points where leverage > 2*(k+1)/n, Cook's D > 4/n, or
    |studentized residual| > 3. Per-point details for flagged rows only,
    capped at MAX_STORED_POINTS per list (most influential first).
    """
    infl = OLSInfluence(model)
    leverage = np.asarray(infl.hat_matrix_diag, dtype=float)
    cooks_d = np.asarray(infl.cooks_distance[0], dtype=float)

    n = len(leverage)
    k = X.shape[1]

    # Externally studentized residuals via the closed-form leave-one-out
    # identity t_i = r_i * sqrt((n-p-1)/(n-p-r_i²)) — statsmodels'
    # resid_studentized_external is O(n²) and takes ~minutes beyond n≈20k.
    p = int(model.df_model) + 1  # parameters incl. intercept
    resid = np.asarray(model.resid, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_int = resid / np.sqrt(model.mse_resid * (1 - leverage))
        stud_resid = r_int * np.sqrt((n - p - 1) / (n - p - r_int**2))
        dffits = stud_resid * np.sqrt(leverage / (1 - leverage))
    leverage_cutoff = 2 * (k + 1) / n
    cooks_cutoff = 4 / n

    def point(i: int) -> InfluencePoint:
        return InfluencePoint(
            row_index=int(i),
            leverage=float(leverage[i]),
            cooks_distance=float(cooks_d[i]),
            studentized_residual=float(stud_resid[i]),
            dffits=float(dffits[i]),
        )

    hl_idx = np.where(leverage > leverage_cutoff)[0]
    ck_idx = np.where(cooks_d > cooks_cutoff)[0]
    n_high_leverage = len(hl_idx)
    n_cooks_outliers = len(ck_idx)

    # Keep the worst offenders, not the first by row order
    hl_top = hl_idx[np.argsort(leverage[hl_idx])[::-1][:MAX_STORED_POINTS]]
    ck_top = ck_idx[np.argsort(cooks_d[ck_idx])[::-1][:MAX_STORED_POINTS]]
    truncated = n_high_leverage > MAX_STORED_POINTS or n_cooks_outliers > MAX_STORED_POINTS

    high_leverage = [point(i) for i in hl_top]
    cooks_outliers = [point(i) for i in ck_top]
    extreme_resid = set(np.where(np.abs(stud_resid) > 3)[0].tolist())

    n_flagged = len(set(hl_idx.tolist()) | set(ck_idx.tolist()) | extreme_resid)
    if n_flagged == 0:
        summary = "No influential observations detected."
    else:
        parts = []
        if n_high_leverage:
            parts.append(f"{n_high_leverage} high-leverage point(s) (h > {leverage_cutoff:.3g})")
        if n_cooks_outliers:
            parts.append(f"{n_cooks_outliers} Cook's D outlier(s) (D > {cooks_cutoff:.3g})")
        if extreme_resid:
            parts.append(f"{len(extreme_resid)} extreme studentized residual(s) (|t| > 3)")
        summary = "; ".join(parts) + "."
        if truncated:
            summary += f" Per-point details retained for the top {MAX_STORED_POINTS} by influence."

    return InfluenceReport(
        high_leverage=high_leverage,
        cooks_d_outliers=cooks_outliers,
        n_high_leverage=n_high_leverage,
        n_cooks_d_outliers=n_cooks_outliers,
        truncated=truncated,
        summary=summary,
        max_cooks_d=float(cooks_d.max()),
        max_leverage=float(leverage.max()),
    )
