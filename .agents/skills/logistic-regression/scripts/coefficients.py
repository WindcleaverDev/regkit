"""Build OddsRatioRow list from a fitted statsmodels Logit model."""

from __future__ import annotations

import numpy as np

from regression_pack_core.schemas import OddsRatioRow


def build_odds_ratio_table(model) -> list[OddsRatioRow]:
    """Return one OddsRatioRow per parameter (including const).

    CIs are on the log-odds scale from the model; exponentiated for the OR scale.
    """
    conf_int = model.conf_int(alpha=0.05)
    rows: list[OddsRatioRow] = []

    for name in model.params.index:
        beta = float(model.params[name])
        se = float(model.bse[name])
        z = float(model.tvalues[name])
        p = float(model.pvalues[name])
        ci_lo = float(conf_int.loc[name, 0])
        ci_hi = float(conf_int.loc[name, 1])
        rows.append(
            OddsRatioRow(
                feature=name,
                log_odds_coefficient=round(beta, 6),
                odds_ratio=round(float(np.exp(beta)), 6),
                std_error=round(se, 6),
                z_stat=round(z, 4),
                p_value=round(p, 6),
                ci_lower_log_odds=round(ci_lo, 6),
                ci_upper_log_odds=round(ci_hi, 6),
                ci_lower_odds_ratio=round(float(np.exp(ci_lo)), 6),
                ci_upper_odds_ratio=round(float(np.exp(ci_hi)), 6),
            )
        )
    return rows
