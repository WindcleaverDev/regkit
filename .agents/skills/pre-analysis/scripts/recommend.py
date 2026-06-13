"""Synthesise modeling_recommendations from all audit findings."""

from __future__ import annotations

from regression_pack_core.schemas import FeatureAudit, TargetAudit


def synthesise_recommendations(
    target: TargetAudit,
    features: list[FeatureAudit],
    multicollinearity: dict,
    suspected_nonlinearity: list[str],
) -> dict:
    transform_target = target.recommendations[0] if target.recommendations else None
    consider_polynomial = suspected_nonlinearity[:]
    drop_or_combine = multicollinearity.get("flagged", [])

    if target.type == "binary":
        preferred_estimator = "logistic_regression"
    elif drop_or_combine:
        preferred_estimator = "regularized_regression"
    else:
        preferred_estimator = "linear_regression"

    return {
        "transform_target": transform_target,
        "consider_polynomial": consider_polynomial,
        "drop_or_combine": drop_or_combine,
        "preferred_estimator": preferred_estimator,
    }
