"""Average marginal effect (AME) computation via statsmodels get_margeff."""

from __future__ import annotations

from regression_pack_core.schemas import MarginalEffect


def compute_marginal_effects(model) -> list[MarginalEffect]:
    """Compute AME for each non-intercept feature.

    Uses statsmodels get_margeff(at='overall', dummy=True):
    - Continuous features: average ∂P/∂x over all observations
    - Binary dummies: average P(x=1) − P(x=0) over all observations
    """
    mfx = model.get_margeff(at="overall", dummy=True)
    df = mfx.summary_frame(alpha=0.05)
    # Columns: 'dy/dx', 'Std. Err.', 'z', 'Pr(>|z|)', 'Conf. Int. Lo', 'Conf. Int. Hi'

    # statsmodels column names vary by version; use positional fallback
    cols = list(df.columns)
    # Expected: ['dy/dx', 'Std. Err.', 'z', 'Pr(>|z|)', <ci_lo>, <ci_hi>]
    ci_lo_col = next((c for c in cols if "low" in c.lower() or "lo" in c.lower()), cols[4] if len(cols) > 4 else None)
    ci_hi_col = next((c for c in cols if "hi" in c.lower()), cols[5] if len(cols) > 5 else None)

    results: list[MarginalEffect] = []
    for name, row in df.iterrows():
        name = str(name)
        if name == "const":
            continue
        results.append(
            MarginalEffect(
                feature=name,
                ame=round(float(row["dy/dx"]), 6),
                std_error=round(float(row["Std. Err."]), 6),
                ci_lower=round(float(row[ci_lo_col]), 6) if ci_lo_col else 0.0,
                ci_upper=round(float(row[ci_hi_col]), 6) if ci_hi_col else 0.0,
                p_value=round(float(row["Pr(>|z|)"]), 6),
            )
        )
    return results
