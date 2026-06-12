"""Influence analysis: leverage, Cook's distance, studentized residuals, DFFITS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import OLSInfluence

from regression_pack_core.schemas import InfluencePoint, InfluenceReport


def run(model, X: pd.DataFrame) -> InfluenceReport:
    """Flag points where leverage > 2*(k+1)/n, Cook's D > 4/n, or
    |studentized residual| > 3. Per-point details for flagged rows only.
    """
    infl = OLSInfluence(model)
    leverage = np.asarray(infl.hat_matrix_diag, dtype=float)
    cooks_d = np.asarray(infl.cooks_distance[0], dtype=float)
    stud_resid = np.asarray(infl.resid_studentized_external, dtype=float)
    dffits = np.asarray(infl.dffits[0], dtype=float)

    n = len(leverage)
    k = X.shape[1]
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

    high_leverage = [point(i) for i in np.where(leverage > leverage_cutoff)[0]]
    cooks_outliers = [point(i) for i in np.where(cooks_d > cooks_cutoff)[0]]
    extreme_resid = set(np.where(np.abs(stud_resid) > 3)[0].tolist())

    n_flagged = len({p.row_index for p in high_leverage}
                    | {p.row_index for p in cooks_outliers}
                    | extreme_resid)
    if n_flagged == 0:
        summary = "No influential observations detected."
    else:
        parts = []
        if high_leverage:
            parts.append(f"{len(high_leverage)} high-leverage point(s) (h > {leverage_cutoff:.3f})")
        if cooks_outliers:
            parts.append(f"{len(cooks_outliers)} Cook's D outlier(s) (D > {cooks_cutoff:.4f})")
        if extreme_resid:
            parts.append(f"{len(extreme_resid)} extreme studentized residual(s) (|t| > 3)")
        summary = "; ".join(parts) + "."

    return InfluenceReport(
        high_leverage=high_leverage,
        cooks_d_outliers=cooks_outliers,
        summary=summary,
        max_cooks_d=float(cooks_d.max()),
        max_leverage=float(leverage.max()),
    )
