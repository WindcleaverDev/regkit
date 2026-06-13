"""Build ComparisonVerdict from comparison artefacts."""

from __future__ import annotations

from regression_pack_core.schemas import (
    AkaikeWeights,
    ComparisonVerdict,
    LRTestResult,
    ModelEntry,
)

_WINNER_WEIGHT_THRESHOLD = 0.80
_TIE_DELTA_AIC_THRESHOLD = 2.0
_ADEQUATE_FQ_THRESHOLD = 0.15


def build_verdict(
    entries: list[ModelEntry],
    lr_tests: list[LRTestResult],
    akaike_weights: AkaikeWeights | None,
    alpha: float = 0.05,
) -> ComparisonVerdict:
    """Return a ComparisonVerdict that characterises the comparison outcome."""

    # 1. All inadequate?
    if all(e.fit_quality_primary < _ADEQUATE_FQ_THRESHOLD for e in entries):
        best_entry = max(entries, key=lambda e: e.fit_quality_primary)
        return ComparisonVerdict(
            overall="all_inadequate",
            recommended_model=best_entry.name,
            headline="All models show poor fit quality",
            rationale=(
                f"Every model's primary fit metric is below {_ADEQUATE_FQ_THRESHOLD:.2f}. "
                "Consider adding features, transforming variables, or using a different model family."
            ),
        )

    # 2. Clear winner via Akaike weights
    if akaike_weights is not None:
        max_weight = max(akaike_weights.weights)
        if max_weight >= _WINNER_WEIGHT_THRESHOLD:
            winner_idx = akaike_weights.weights.index(max_weight)
            winner_name = akaike_weights.model_names[winner_idx]
            w_entry = next((e for e in entries if e.name == winner_name), None)
            fq = w_entry.fit_quality_primary if w_entry else 0.0
            return ComparisonVerdict(
                overall="clear_winner",
                recommended_model=winner_name,
                headline=f"{winner_name} is the clear best model (Akaike weight = {max_weight:.2f})",
                rationale=(
                    f"{winner_name} accounts for {max_weight * 100:.0f}% of Akaike weight "
                    f"and has the lowest AIC "
                    f"(Δ AIC = {akaike_weights.delta_aic[winner_idx]:.2f} vs next best). "
                    f"Primary fit quality metric = {fq:.4f}."
                ),
            )

        # Competitive tie: all Δ AIC < threshold
        max_delta = max(akaike_weights.delta_aic)
        if max_delta < _TIE_DELTA_AIC_THRESHOLD:
            best_name = akaike_weights.model_names[0]  # first = lowest AIC
            return ComparisonVerdict(
                overall="competitive_tie",
                recommended_model=best_name,
                headline="Models are statistically indistinguishable by AIC",
                rationale=(
                    f"Max Δ AIC = {max_delta:.2f} < {_TIE_DELTA_AIC_THRESHOLD:.0f}. "
                    "Prefer the simpler model or the one with lower prediction error. "
                    f"{best_name} has marginal AIC advantage."
                ),
            )

    # 3. LR test resolves nesting
    if lr_tests:
        # If any LR test rejects the simpler model, full model wins
        significant_lr = [t for t in lr_tests if t.p_value < alpha]
        if significant_lr:
            # Pick the largest full model that is significantly better
            def _n_feats(name: str) -> int:
                e = next((e for e in entries if e.name == name), None)
                return e.n_features if e else 0

            best_lr = max(significant_lr, key=lambda t: _n_feats(t.full_model))
            winner_name = best_lr.full_model
            return ComparisonVerdict(
                overall="clear_winner",
                recommended_model=winner_name,
                headline=f"{winner_name} significantly outperforms simpler nested models",
                rationale=(
                    f"LR test: {best_lr.full_model} vs {best_lr.nested_model}: "
                    f"LR = {best_lr.likelihood_ratio:.2f}, df = {best_lr.df}, "
                    f"p = {best_lr.p_value:.4f}. {best_lr.conclusion}"
                ),
            )

    # 4. Fallback: rank by fit_quality_primary
    by_fq = sorted(entries, key=lambda e: e.fit_quality_primary, reverse=True)
    best = by_fq[0]
    second = by_fq[1] if len(by_fq) > 1 else None

    if second is None or (best.fit_quality_primary - second.fit_quality_primary) > 0.02:
        return ComparisonVerdict(
            overall="clear_winner",
            recommended_model=best.name,
            headline=f"{best.name} has the best primary fit quality",
            rationale=(
                f"{best.name}: primary metric = {best.fit_quality_primary:.4f}. "
                + (
                    f"{second.name}: {second.fit_quality_primary:.4f}. "
                    if second
                    else ""
                )
                + "AIC comparison not available (models may differ in target or family)."
            ),
        )

    # Families differ or AIC unavailable — complementary strengths
    return ComparisonVerdict(
        overall="complementary_strengths",
        recommended_model=best.name,
        headline="Models show similar fit quality; choice depends on modelling goals",
        rationale=(
            "Primary fit quality metrics are close across models. "
            "Consider interpretability (linear vs regularised), robustness to outliers "
            "(robust SE), or cross-validation scores to differentiate."
        ),
    )
