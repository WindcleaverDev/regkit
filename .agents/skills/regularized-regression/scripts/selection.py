"""Feature selection summary for Lasso / ElasticNet.

Ridge never zeros coefficients, so this returns None for ridge.
"""

from __future__ import annotations

import numpy as np

from regression_pack_core.schemas import FeatureSelection


def build_feature_selection(
    coef_orig: np.ndarray,
    feature_names: list[str],
    method: str,
    coef_zero_mask: np.ndarray | None = None,
    tol: float = 1e-8,
) -> FeatureSelection | None:
    """Return FeatureSelection for lasso/elasticnet; None for ridge.

    coef_zero_mask: boolean array from model.coef_ == 0.0 (preferred — uses
    sklearn's exact sparsity signal). Falls back to abs(coef_orig) < tol.
    """
    if method == "ridge":
        return None

    selected, dropped = [], []
    for i, name in enumerate(feature_names):
        # Prefer the exact zero mask from sklearn; fall back to threshold
        is_zero = (
            bool(coef_zero_mask[i]) if coef_zero_mask is not None else abs(coef_orig[i]) <= tol
        )
        if is_zero:
            dropped.append(name)
        else:
            selected.append(name)

    return FeatureSelection(
        n_selected=len(selected),
        n_dropped=len(dropped),
        selected_features=selected,
        dropped_features=dropped,
    )
