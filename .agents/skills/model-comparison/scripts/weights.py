"""Akaike weights for non-nested model comparison."""

from __future__ import annotations

import math

from regression_pack_core.schemas import AkaikeWeights, ModelEntry


def compute_akaike_weights(entries: list[ModelEntry]) -> AkaikeWeights | None:
    """Compute Akaike weights from model AICs.

    Filters out models whose AIC is None. Returns None if fewer than 2
    models have valid AIC values.
    """
    valid = [(e.name, e.aic) for e in entries if e.aic is not None]
    if len(valid) < 2:
        return None

    names = [v[0] for v in valid]
    aics = [v[1] for v in valid]

    aic_min = min(aics)
    deltas = [a - aic_min for a in aics]

    # exp(-Δ/2), normalised
    raw = [math.exp(-d / 2.0) for d in deltas]
    total = sum(raw)
    weights = [r / total for r in raw]

    return AkaikeWeights(
        model_names=names,
        delta_aic=[round(d, 4) for d in deltas],
        weights=[round(w, 6) for w in weights],
    )
