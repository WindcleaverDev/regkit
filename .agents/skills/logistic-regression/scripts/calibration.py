"""Calibration (reliability diagram) data and Brier score."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss

from regression_pack_core.schemas import CalibrationData


def compute_calibration(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationData:
    """Compute calibration bin statistics using quantile-spaced bins.

    Quantile bins ensure each bin has roughly equal observation count,
    which avoids empty bins at the extremes.
    """
    brier = float(brier_score_loss(y_true, y_pred_prob))

    # Quantile bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(y_pred_prob, quantiles)
    edges = np.unique(edges)  # deduplicate when y_pred_prob has ties

    bin_centers: list[float] = []
    obs_freqs: list[float] = []
    bin_counts: list[int] = []

    bin_idx = np.digitize(y_pred_prob, edges[1:-1])  # which bin each obs falls in
    n_actual_bins = len(edges) - 1

    for b in range(n_actual_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_centers.append(round(float(y_pred_prob[mask].mean()), 6))
        obs_freqs.append(round(float(y_true[mask].mean()), 6))
        bin_counts.append(count)

    return CalibrationData(
        bin_centers=bin_centers,
        observed_frequencies=obs_freqs,
        bin_counts=bin_counts,
        brier_score=round(brier, 6),
    )
