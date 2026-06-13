"""Audit the target column: type detection, skewness, outlier count, recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from regression_pack_core.schemas import TargetAudit


def audit_target(y: pd.Series, name: str) -> TargetAudit:
    n = len(y)
    n_missing = int(y.isna().sum())
    y_clean = y.dropna()
    n_unique = int(y_clean.nunique())

    # Type detection — order matters: binary before count before continuous
    if n_unique <= 2:
        target_type = "binary"
    elif (
        pd.api.types.is_integer_dtype(y_clean)
        and (y_clean >= 0).all()
        and int(y_clean.max()) < 30
    ):
        target_type = "count"
    elif pd.api.types.is_numeric_dtype(y_clean):
        target_type = "continuous"
    else:
        target_type = "categorical"

    skewness: float | None = None
    kurtosis: float | None = None
    outlier_count: int | None = None
    recommendations: list[str] = []

    if target_type in ("continuous", "count"):
        arr = y_clean.to_numpy(dtype=float)
        skewness = float(stats.skew(arr))
        kurtosis = float(stats.kurtosis(arr))

        # IQR-based outlier count
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        outlier_count = int(((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)).sum())

        if target_type == "continuous":
            all_positive = bool((y_clean > 0).all())
            if all_positive and skewness > 1.5:
                recommendations.append("log_transform")
            elif all_positive and abs(skewness) > 1.0:
                recommendations.append("box_cox")
        if outlier_count > 0.05 * n:
            recommendations.append("winsorize")

    return TargetAudit(
        name=name,
        type=target_type,
        n_missing=n_missing,
        skewness=skewness,
        kurtosis=kurtosis,
        outlier_count=outlier_count,
        recommendations=recommendations,
    )
