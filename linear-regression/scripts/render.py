"""Compose the linear-regression HTML report."""

from __future__ import annotations

import html

from regression_pack_core import plotting, reports
from regression_pack_core.schemas import LinearRegressionReport


def _interpretation_blocks(report: LinearRegressionReport) -> str:
    blocks = []
    for fact in report.interpretations:
        caveats = "".join(f"<li>{html.escape(c)}</li>" for c in fact.caveats)
        pill_tone = {"high": "ok", "medium": "warn", "low": "fail"}[fact.confidence]
        blocks.append(
            '<div style="background:var(--bg-soft);border-radius:6px;padding:14px 16px;margin-bottom:10px;">'
            f'<div style="font-size:14px;margin-bottom:6px;">{html.escape(fact.fact)} '
            f'<span class="pill {pill_tone}">{fact.confidence} confidence</span></div>'
            f'<ul style="font-size:12px;color:var(--ink-soft);margin:4px 0 0;padding-left:18px;">{caveats}</ul>'
            "</div>"
        )
    return "".join(blocks)


def render_report(
    report: LinearRegressionReport,
    *,
    target: str,
    dataset_name: str,
    timestamp: str,
) -> str:
    fs = report.fit_statistics

    # 1. Verdict card
    tone = "ok" if report.fit_quality.interpretation in ("strong", "very_strong") else "warn"
    verdict = {
        "tone": tone,
        "label": f"Fit quality: {report.fit_quality.interpretation.replace('_', ' ')}",
        "headline": report.headline,
    }

    # 2. Fit summary
    fit_stats = reports.stat_grid(
        [
            {"label": "Observations", "value": f"{fs.n_observations:,}"},
            {"label": "Features", "value": str(fs.n_features)},
            {"label": "R²", "value": f"{fs.r_squared:.4f}"},
            {"label": "Adj. R²", "value": f"{fs.adj_r_squared:.4f}"},
            {"label": "F-statistic", "value": f"{fs.f_statistic:.2f}", "sub": f"p = {fs.f_p_value:.3g}"},
            {"label": "AIC", "value": f"{fs.aic:.1f}"},
            {"label": "BIC", "value": f"{fs.bic:.1f}"},
            {"label": "Residual SE", "value": f"{fs.residual_std_error:.4g}"},
        ]
    )
    extras = []
    if report.target_transform:
        extras.append(f"target transform: <strong>{report.target_transform}</strong>")
    if report.robust_se_used:
        extras.append(f"robust standard errors: <strong>{report.robust_se_used}</strong>")
    extra_html = f"<p>{' · '.join(extras)}</p>" if extras else ""
    fit_section = reports.section(
        "Fit summary", fit_stats + f"<p>{html.escape(report.headline)}</p>" + extra_html
    )

    # 3. Coefficients — intercept first, others sorted by |β| descending
    const_rows = [c for c in report.coefficients if c.feature == "const"]
    others = sorted(
        (c for c in report.coefficients if c.feature != "const"),
        key=lambda c: abs(c.coefficient),
        reverse=True,
    )
    table = reports.coefficient_table_html(const_rows + others)
    forest = plotting.coef_forest(
        [c.feature for c in others],
        [c.coefficient for c in others],
        [c.ci_lower for c in others],
        [c.ci_upper for c in others],
        title="Coefficients with 95% confidence intervals",
    )
    coef_section = reports.section(
        "Coefficients",
        table + f'<div class="plot-container">{plotting.to_inline_html(forest, "coef-forest")}</div>',
    )

    # 4. Interpretation
    interp_section = reports.section("Interpretation", _interpretation_blocks(report))

    # 5. Diagnostics placeholder
    diag_section = reports.section(
        "Diagnostics",
        "<p>For full assumption checks, influence analysis, and bias/variance assessment, "
        "run the <code>diagnostics</code> skill on this fit. See README.</p>",
    )

    body = fit_section + coef_section + interp_section + diag_section
    return reports.render_html_report(
        title=f"Linear regression: {target}",
        skill_name="linear-regression",
        dataset_name=dataset_name,
        n_obs=fs.n_observations,
        timestamp=timestamp,
        verdict=verdict,
        body_html=body,
    )
