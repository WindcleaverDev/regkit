"""Per-feature audit: type inference, missingness, cardinality, flag detection."""

from __future__ import annotations

import pandas as pd

from regression_pack_core.schemas import FeatureAudit


def _infer_type(col: pd.Series) -> str:
    n_unique = col.nunique()
    if n_unique <= 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(col):
        return "continuous"
    return "categorical"


def audit_feature(col: pd.Series, name: str, n_total: int) -> FeatureAudit:
    n_missing = int(col.isna().sum())
    missing_pct = n_missing / n_total if n_total > 0 else 0.0
    col_clean = col.dropna()
    n_unique = int(col_clean.nunique())
    feat_type = _infer_type(col_clean)

    flags: list[str] = []

    if missing_pct > 0.10:
        flags.append("high_missing")

    if feat_type == "categorical":
        max_cardinality = min(20, n_total // 20) if n_total >= 20 else 20
        if n_unique > max_cardinality:
            flags.append("high_cardinality")
        if n_unique == n_total:
            flags.append("quasi_id")
        elif len(col_clean) > 0:
            top_freq = col_clean.value_counts(normalize=True).iloc[0]
            if top_freq > 0.95:
                flags.append("near_constant")
    elif feat_type == "continuous" and len(col_clean) > 1:
        arr = col_clean.to_numpy(dtype=float)
        std = arr.std(ddof=1)
        mean = arr.mean()
        if std == 0 or (mean != 0 and (std / abs(mean)) < 0.01):
            flags.append("near_constant")

    return FeatureAudit(
        name=name,
        type=feat_type,
        n_missing=n_missing,
        missing_pct=missing_pct,
        n_unique=n_unique,
        flags=flags,
    )
