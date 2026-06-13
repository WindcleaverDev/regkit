"""Classification statistics at a given decision threshold."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from regression_pack_core.schemas import ClassificationStats


def compute_classification_stats(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationStats:
    """Compute classification metrics at the given threshold."""
    y_pred = (y_pred_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    # Ensure 2×2 even if one class is missing in predictions
    if cm.shape != (2, 2):
        cm = np.array([[int((y_true == 0).sum()), 0], [0, int((y_true == 1).sum())]])

    return ClassificationStats(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 6),
        balanced_accuracy=round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        confusion_matrix=cm.tolist(),
        threshold=threshold,
        n_observations=int(len(y_true)),
        class_balance=round(float(y_true.mean()), 6),
    )
