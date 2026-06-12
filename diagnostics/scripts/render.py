"""Compose the diagnostics HTML report."""

from __future__ import annotations

import html

from regression_pack_core import plotting, reports
from regression_pack_core.schemas import DiagnosticsReport

TONE_BY_OVERALL = {
    "clean": "ok",
    "usable_with_caveats": "ok",
    "problematic": "warn",
    "unreliable": "fail",
}


def _assumptions_table(report: DiagnosticsReport) -> str:
    rows = []
    for a in report.assumptions:
        stat = f"{a.statistic:.4g}" if a.statistic is not None else "—"
        p = f"{a.p_value:.4g}" if a.p_value is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(a.name.replace('_', ' '))}</td>"
            f"<td>{html.escape(a.test_name)}</td>"
            f'<td class="num">{stat}</td>'
            f'<td class="num">{p}</td>'
            f'<td><span class="pill {a.status.value}">{a.status.value}</span></td>'
            "</tr>"
        )
    return (
        '<table class="data-table"><thead><tr>'
        "<th>Assumption</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Status</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _influence_section(report: DiagnosticsReport) -> str:
    flagged = {p.row_index: p for p in report.influence.high_leverage}
    flagged.update({p.row_index: p for p in report.influence.cooks_d_outliers})
    if not flagged:
        return "<p>No influential observations detected.</p>"
    rows = []
    for idx in sorted(flagged):
        p = flagged[idx]
        dffits = f"{p.dffits:.3f}" if p.dffits is not None else "—"
        rows.append(
            "<tr>"
            f'<td class="num">{p.row_index}</td>'
            f'<td class="num">{p.leverage:.4f}</td>'
            f'<td class="num">{p.cooks_distance:.4f}</td>'
            f'<td class="num">{p.studentized_residual:.3f}</td>'
            f'<td class="num">{dffits}</td>'
            "</tr>"
        )
    table = (
        '<table class="data-table"><thead><tr>'
        "<th>Row</th><th>Leverage</th><th>Cook's D</th><th>Studentized residual</th><th>DFFITS</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    return f"<p>{html.escape(report.influence.summary)}</p>{table}"


def render_report(
    report: DiagnosticsReport,
    resid_data: dict,
    *,
    dataset_name: str,
    n_obs: int,
    timestamp: str,
) -> str:
    # 1. Verdict card
    verdict = {
        "tone": TONE_BY_OVERALL[report.verdict.overall],
        "label": f"Verdict: {report.verdict.overall.replace('_', ' ')}",
        "headline": report.verdict.headline,
    }

    # 2. Assumption checks
    assumptions_section = reports.section("Assumption checks", _assumptions_table(report))

    # 3. Diagnostic plots (grid-2)
    fitted = resid_data["fitted"]
    residuals = resid_data["residuals"]
    plots = [
        plotting.residuals_vs_fitted(fitted, residuals, title="Residuals vs fitted"),
        plotting.qq_plot(residuals, title="Normal QQ plot"),
        plotting.scale_location(fitted, residuals, title="Scale-location"),
        plotting.leverage_plot(
            resid_data["leverage"],
            resid_data["std_residuals"],
            resid_data["cooks_d"],
            title="Residuals vs leverage (size ∝ Cook's D)",
        ),
    ]
    plots_html = "".join(
        f"<div>{plotting.to_inline_html(fig, f'diag-plot-{i}')}</div>"
        for i, fig in enumerate(plots)
    )
    plots_section = reports.section(
        "Diagnostic plots", f'<div class="plot-container grid-2">{plots_html}</div>'
    )

    # 4. Influential observations
    influence_section = reports.section("Influential observations", _influence_section(report))

    # 5. Bias / variance
    bv = report.bias_variance
    bv_stats = reports.stat_grid(
        [
            {"label": "Train R²", "value": f"{bv.train_r_squared:.4f}"},
            {"label": "Test R²", "value": f"{bv.test_r_squared:.4f}"},
            {
                "label": "CV R²",
                "value": f"{bv.cv_r_squared_mean:.4f}",
                "sub": f"± {bv.cv_r_squared_std:.4f}",
            },
            {"label": "Gap", "value": f"{bv.gap:.4f}", "sub": f"verdict: {bv.verdict}"},
        ]
    )
    bv_body = bv_stats + f"<p>{html.escape(bv.evidence)}</p>"
    if bv.learning_curve:
        lc = bv.learning_curve
        fig = plotting.line(
            lc["sizes"], lc["train_scores"], x_label="Training size", y_label="R²",
            title="Learning curve",
        )
        fig.add_scatter(
            x=lc["sizes"], y=lc["test_scores"], mode="lines",
            line=dict(color=plotting.WARN, width=2), name="CV score",
        )
        bv_body += f'<div class="plot-container">{plotting.to_inline_html(fig, "learning-curve")}</div>'
    bv_section = reports.section("Bias / variance", bv_body)

    # 6. Recommendations
    recs = "".join(
        f"<li><strong>{html.escape(r.action)}</strong>"
        + (f" ({html.escape(r.target)})" if r.target else "")
        + f" — {html.escape(r.reason)}</li>"
        for r in report.recommendations
    )
    rec_body = reports.flag_list_html(report.flags)
    if recs:
        rec_body += f'<ol style="font-size:14px;">{recs}</ol>'
    rec_section = reports.section("Recommendations", rec_body)

    body = (
        assumptions_section
        + plots_section
        + influence_section
        + bv_section
        + rec_section
    )
    return reports.render_html_report(
        title="Regression diagnostics",
        skill_name="diagnostics",
        dataset_name=dataset_name,
        n_obs=n_obs,
        timestamp=timestamp,
        verdict=verdict,
        body_html=body,
    )
