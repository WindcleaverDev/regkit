"""Triage: combine assumption, influence and bias/variance findings into one verdict."""

from __future__ import annotations

from regression_pack_core.schemas import (
    AssumptionCheck,
    BiasVarianceReport,
    DiagnosticsVerdict,
    InfluenceReport,
    Status,
)

# Flag code per assumption, used for top_issues ordering
ASSUMPTION_CODES = {
    "linearity": "MISSED_NONLINEARITY",
    "homoscedasticity": "HETEROSCEDASTICITY",
    "normality_of_residuals": "NON_NORMAL_RESIDUALS",
    "independence": "AUTOCORRELATION",
    "no_multicollinearity": "HIGH_VIF",
}


# A Cook's D this large is influential regardless of n. Required on top of the
# relative 5*(4/n) rule, which shrinks with n and fires on chance points at scale.
COOKS_ABS_FLOOR = 0.5


def is_severe_cooks(cooks_distance: float, n: int) -> bool:
    return cooks_distance > 5 * (4 / n) and cooks_distance > COOKS_ABS_FLOOR


def build_verdict(
    assumptions: list[AssumptionCheck],
    influence: InfluenceReport,
    bias_variance: BiasVarianceReport,
) -> DiagnosticsVerdict:
    fails = [a for a in assumptions if a.status == Status.FAIL]
    warns = [a for a in assumptions if a.status == Status.WARN]
    n_fail, n_warn = len(fails), len(warns)

    n = _n_from(influence, assumptions)
    severe_influence = any(
        is_severe_cooks(p.cooks_distance, n) for p in influence.cooks_d_outliers
    )

    bv = bias_variance.verdict
    if n_fail >= 3 or bv == "high_bias":
        overall = "unreliable"
    elif n_fail >= 1 or bv == "high_variance":
        overall = "problematic"
    elif n_warn == 0 and bv == "good_fit" and not severe_influence:
        overall = "clean"
    elif n_warn <= 2 and bv in ("good_fit", "inconsistent"):
        overall = "usable_with_caveats"
    else:
        overall = "problematic"

    # top_issues: FAILs first, then WARNs, then influence/bias-variance issues
    top_issues = [ASSUMPTION_CODES[a.name] for a in fails]
    top_issues += [ASSUMPTION_CODES[a.name] for a in warns]
    if severe_influence:
        top_issues.append("HIGH_COOKS_D")
    if bv == "high_variance":
        top_issues.append("HIGH_VARIANCE")
    elif bv == "high_bias":
        top_issues.append("HIGH_BIAS")

    headline = _headline(overall, n_fail, n_warn, bv, severe_influence)
    return DiagnosticsVerdict(overall=overall, top_issues=top_issues, headline=headline)


def _n_from(influence: InfluenceReport, assumptions: list[AssumptionCheck]) -> int:
    """Recover n for the 4/n Cook's threshold from assumption detail if present."""
    for a in assumptions:
        if a.detail and "n" in a.detail:
            return int(a.detail["n"])
    return 100  # conservative fallback


def _headline(
    overall: str, n_fail: int, n_warn: int, bv_verdict: str, severe_influence: bool
) -> str:
    if overall == "clean":
        return "All assumption checks pass and the model generalises well — results can be trusted."
    if overall == "usable_with_caveats":
        issues = f"{n_warn} marginal assumption check(s)" if n_warn else "minor caveats"
        return f"The model is usable, but note {issues} before relying on the estimates."
    if overall == "problematic":
        parts = []
        if n_fail:
            parts.append(f"{n_fail} assumption check(s) fail")
        if bv_verdict == "high_variance":
            parts.append("the model overfits (high variance)")
        if severe_influence:
            parts.append("results are driven by influential observations")
        body = " and ".join(parts) if parts else "multiple checks are marginal"
        return f"Caution: {body} — address these before trusting inference."
    return (
        "The model is unreliable in its current form — multiple assumptions fail "
        "or it badly underfits; remediate before using any estimates."
    )
