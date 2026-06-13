"""ROC curve data and AUC computation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from regression_pack_core.schemas import ROCData


def compute_roc(y_true: np.ndarray, y_pred_prob: np.ndarray, max_points: int = 100) -> ROCData:
    """Compute ROC curve downsampled to max_points for compact storage."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = float(roc_auc_score(y_true, y_pred_prob))

    # sklearn sets the first threshold to np.inf; clamp to finite range
    thresholds = np.clip(thresholds, 0.0, 1.0)

    # Downsample evenly if there are many points
    n = len(fpr)
    if n > max_points:
        idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
        fpr = fpr[idx]
        tpr = tpr[idx]
        thresholds = thresholds[idx]

    return ROCData(
        fpr=[round(float(v), 6) for v in fpr],
        tpr=[round(float(v), 6) for v in tpr],
        thresholds=[round(float(v), 6) for v in thresholds],
        auc=round(auc, 6),
    )
