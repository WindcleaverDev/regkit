"""Generate InterpretationFact objects — the LLM-fact layer.

Each non-intercept coefficient gets exactly one fact: a canonical claim with
a confidence grade and caveats. Claude verbalises these; it never invents
statistics.
"""

from __future__ import annotations

import math
import re

from regression_pack_core.schemas import CoefficientRow, InterpretationFact

BASE_CAVEATS = ["holding other features constant", "association, not causation"]

_POLY_RE = re.compile(r"^(?P<base>.+?)(\^2|\*\*2|_sq(uared)?)$")
_INTERACTION_RE = re.compile(r"^(?P<f1>.+?):(?P<f2>.+)$")


def _direction(beta: float) -> str:
    return "increase" if beta > 0 else "decrease"


def _confidence(row: CoefficientRow) -> str:
    ci_width = row.ci_upper - row.ci_lower
    if row.p_value < 0.01 and ci_width < abs(row.coefficient):
        return "high"
    if row.p_value < 0.05:
        return "medium"
    return "low"


def _classify(
    feature: str,
    target_transform: str | None,
    feature_transforms: dict[str, str],
    dummy_origin: dict[str, str],
) -> str:
    if _INTERACTION_RE.match(feature):
        return "interaction_term"
    if _POLY_RE.match(feature) and _POLY_RE.match(feature).group("base") in feature_transforms.get("__features__", []):
        return "polynomial_term"
    if _POLY_RE.match(feature):
        return "polynomial_term"
    if feature in dummy_origin:
        origin = dummy_origin[feature]
        n_dummies = sum(1 for v in dummy_origin.values() if v == origin)
        return "binary_dummy" if n_dummies == 1 else "categorical_dummy"
    f_log = feature_transforms.get(feature) == "log"
    t_log = target_transform == "log"
    if t_log and f_log:
        return "log_log_elasticity"
    if t_log:
        return "log_linear_semi_elasticity"
    if f_log:
        return "linear_log"
    return "linear_linear"


def _build_fact(
    itype: str,
    feature: str,
    target: str,
    row: CoefficientRow,
    dummy_origin: dict[str, str],
    target_transform: str | None = None,
) -> tuple[str, list[str]]:
    """Return (fact sentence, extra caveats) for an interpretation type."""
    beta = row.coefficient
    direction = _direction(beta)
    extra: list[str] = []

    if itype == "linear_linear":
        fact = (
            f"A one-unit increase in {feature} is associated with a {abs(beta):.3g} "
            f"{direction} in {target}, holding other features constant."
        )
    elif itype == "log_log_elasticity":
        fact = (
            f"A 1% increase in {feature} is associated with a {abs(beta) * 100:.2f}% "
            f"{direction} in {target}, holding other features constant."
        )
        extra.append("elasticity interpretation — both variables on log scale")
    elif itype == "log_linear_semi_elasticity":
        pct = (math.exp(beta) - 1) * 100
        fact = (
            f"A one-unit increase in {feature} is associated with a {abs(pct):.2f}% "
            f"{_direction(pct)} in {target}, holding other features constant."
        )
        extra.append("semi-elasticity — exact percentage uses exp(β)−1")
    elif itype == "linear_log":
        fact = (
            f"A 1% increase in {feature} is associated with a {abs(beta) / 100:.4g} "
            f"{direction} in {target}, holding other features constant."
        )
        extra.append("feature on log scale — effect per percentage change")
    elif itype in ("binary_dummy", "categorical_dummy"):
        origin = dummy_origin.get(feature, feature)
        level = feature.removeprefix(f"{origin}_") if feature.startswith(f"{origin}_") else feature
        if target_transform == "log":
            # β is in log-target units — a raw-unit claim would be wrong
            pct = (math.exp(beta) - 1) * 100
            effect = f"{abs(pct):.2f}% {_direction(pct)}"
            extra.append("semi-elasticity — exact percentage uses exp(β)−1")
        else:
            effect = f"{abs(beta):.3g} {direction}"
        fact = (
            f"{origin}={level} is associated with a {effect} in {target} "
            f"compared to the reference category, holding other features constant."
        )
        extra.append("interpretation is relative to the dropped reference category")
    elif itype == "polynomial_term":
        base = _POLY_RE.match(feature).group("base")
        curvature = "convex (accelerating) curvature" if beta > 0 else "concave (diminishing) curvature"
        fact = (
            f"{base} has a nonlinear relationship with {target}; the {base}² term is "
            f"{beta:.3g} (p={row.p_value:.3g}), indicating {curvature}."
        )
        extra.append("interpret jointly with the linear term, not in isolation")
    elif itype == "interaction_term":
        m = _INTERACTION_RE.match(feature)
        f1, f2 = m.group("f1"), m.group("f2")
        fact = (
            f"The effect of {f1} on {target} varies with {f2}; the interaction "
            f"coefficient is {beta:.3g} (p={row.p_value:.3g})."
        )
        extra.append("main effects must be interpreted at specific values of the moderator")
    else:  # pragma: no cover - exhaustive above
        raise ValueError(f"Unknown interpretation type: {itype}")

    return fact, extra


def generate_interpretations(
    coefs: list[CoefficientRow],
    *,
    target: str = "the target",
    target_transform: str | None = None,
    feature_transforms: dict[str, str] | None = None,
    dummy_origin: dict[str, str] | None = None,  # {"neighborhood_north": "neighborhood"}
) -> list[InterpretationFact]:
    """Generate one fact per non-intercept coefficient."""
    feature_transforms = feature_transforms or {}
    dummy_origin = dummy_origin or {}

    facts: list[InterpretationFact] = []
    for row in coefs:
        if row.feature == "const":
            continue
        itype = _classify(row.feature, target_transform, feature_transforms, dummy_origin)
        fact, extra = _build_fact(itype, row.feature, target, row, dummy_origin, target_transform)
        confidence = _confidence(row)
        caveats = [*BASE_CAVEATS, *extra]
        if confidence == "low":
            caveats.append("not statistically significant at p < 0.05 — treat as inconclusive")
        facts.append(
            InterpretationFact(
                feature=row.feature,
                coefficient=row.coefficient,
                interpretation_type=itype,
                fact=fact,
                confidence=confidence,
                caveats=caveats,
            )
        )
    return facts
