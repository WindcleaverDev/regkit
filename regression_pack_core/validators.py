"""Input validation and feature coercion shared by all fit skills."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

MIN_ROWS = 30


class ValidationResult(BaseModel):
    ok: bool
    message: str = ""
    issues: list[str] = Field(default_factory=list)


def validate_regression_inputs(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    require_numeric_target: bool = True,
) -> ValidationResult:
    """Verify target & features exist, target is numeric (if required), no
    entirely-null columns, and enough rows remain after dropping NA.
    """
    issues: list[str] = []

    if target not in df.columns:
        issues.append(f"Target column '{target}' not found in data.")
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        issues.append(f"Feature column(s) not found in data: {', '.join(missing_features)}.")
    if not features:
        issues.append("No feature columns specified.")
    if target in features:
        issues.append(f"Target '{target}' cannot also be a feature.")

    if issues:
        return ValidationResult(ok=False, message="Column validation failed.", issues=issues)

    if require_numeric_target and not pd.api.types.is_numeric_dtype(df[target]):
        issues.append(f"Target column '{target}' must be numeric (got dtype {df[target].dtype}).")

    for col in [target, *features]:
        if df[col].isna().all():
            issues.append(f"Column '{col}' is entirely null.")

    if not issues:
        n_complete = len(df[[target, *features]].dropna())
        if n_complete < MIN_ROWS:
            issues.append(
                f"Only {n_complete} complete rows after dropping NA "
                f"(minimum {MIN_ROWS} required)."
            )

    if issues:
        return ValidationResult(ok=False, message="Input validation failed.", issues=issues)
    return ValidationResult(ok=True, message="Inputs valid.")


def coerce_features(
    df: pd.DataFrame,
    features: list[str],
    *,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode categorical / object dtype columns.

    Returns the encoded design matrix and a dict mapping original feature ->
    list of dummy column names (so coefficient interpretation can map back).
    """
    X = df[features].copy()
    dummy_map: dict[str, list[str]] = {}

    categorical = [
        f
        for f in features
        if not pd.api.types.is_numeric_dtype(X[f]) or isinstance(X[f].dtype, pd.CategoricalDtype)
    ]
    # Treat booleans as numeric 0/1 rather than dummies
    for f in features:
        if pd.api.types.is_bool_dtype(X[f]):
            X[f] = X[f].astype(int)
            if f in categorical:
                categorical.remove(f)

    if categorical:
        before = set(X.columns)
        X = pd.get_dummies(X, columns=categorical, drop_first=drop_first, dtype=float)
        after = set(X.columns)
        new_cols = after - before
        for f in categorical:
            dummy_map[f] = sorted(c for c in new_cols if c.startswith(f"{f}_"))

    X = X.astype(float)
    return X, dummy_map
