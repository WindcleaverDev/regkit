"""Interpretation facts for logistic regression — AME-based narrative."""

from __future__ import annotations

from regression_pack_core.schemas import InterpretationFact, MarginalEffect, OddsRatioRow


def generate_logistic_interpretations(
    coefs: list[OddsRatioRow],
    marginal_effects: list[MarginalEffect],
    *,
    target: str = "the target",
    positive_class_label: str = "the positive class",
) -> list[InterpretationFact]:
    """Generate one fact per non-intercept coefficient.

    Interpretations are framed in terms of the average marginal effect on
    predicted probability (AME), not raw log-odds or odds ratios. AME is the
    quantity humans find most interpretable.
    """
    ame_by_feature = {me.feature: me for me in marginal_effects}
    facts: list[InterpretationFact] = []

    for row in coefs:
        if row.feature == "const":
            continue
        me = ame_by_feature.get(row.feature)
        if me is None:
            continue

        direction = "increase" if me.ame > 0 else "decrease"
        pct = abs(me.ame) * 100

        # Confidence
        if me.p_value < 0.01:
            confidence = "high"
        elif me.p_value < 0.05:
            confidence = "medium"
        else:
            confidence = "low"

        fact = (
            f"A one-unit increase in {row.feature} is associated with a "
            f"{pct:.1f} percentage-point {direction} in the probability of "
            f"{positive_class_label}, on average (AME = {me.ame:+.4f})."
        )

        caveats = [
            "average marginal effect — impact varies across the predictor space",
            "association, not causation",
            "holding other features constant",
        ]
        or_str = f"{row.odds_ratio:.2f}"
        caveats.append(
            f"odds ratio = {or_str} (AME is on probability scale; OR is on odds scale)"
        )
        if confidence == "low":
            caveats.append("not significant at p < 0.05 — treat as inconclusive")

        facts.append(
            InterpretationFact(
                feature=row.feature,
                coefficient=me.ame,
                interpretation_type="logistic_marginal",
                fact=fact,
                confidence=confidence,
                caveats=caveats,
            )
        )
    return facts
