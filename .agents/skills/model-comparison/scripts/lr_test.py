"""Chi-squared likelihood-ratio test for nested model pairs."""

from __future__ import annotations

from scipy.stats import chi2

from regression_pack_core.schemas import LRTestResult


def run_lr_test(
    nested_report: dict,
    full_report: dict,
    alpha: float = 0.05,
) -> LRTestResult | None:
    """LR test: 2*(ll_full - ll_nested) ~ chi²(df).

    Returns None if either model lacks a log-likelihood.
    """
    ll_nested = nested_report.get("_log_likelihood")
    ll_full = full_report.get("_log_likelihood")
    if ll_nested is None or ll_full is None:
        return None

    feat_nested = set(nested_report["_features"])
    feat_full = set(full_report["_features"])
    df = len(feat_full) - len(feat_nested)
    if df <= 0:
        return None

    lr_stat = 2.0 * (ll_full - ll_nested)
    if lr_stat < 0:
        # Full model has lower log-likelihood — unusual; still report
        p_value = 1.0
    else:
        p_value = float(1.0 - chi2.cdf(lr_stat, df=df))

    if p_value < 0.001:
        conclusion = f"Full model significantly better (p = {p_value:.2e} < 0.001)"
    elif p_value < alpha:
        conclusion = f"Full model significantly better (p = {p_value:.4f} < {alpha})"
    else:
        conclusion = (
            f"Full model NOT significantly better than nested "
            f"(p = {p_value:.4f} ≥ {alpha}); prefer simpler model"
        )

    return LRTestResult(
        nested_model=nested_report["_name"],
        full_model=full_report["_name"],
        likelihood_ratio=round(lr_stat, 4),
        df=df,
        p_value=round(p_value, 6),
        conclusion=conclusion,
    )
