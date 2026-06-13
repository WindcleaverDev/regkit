"""Compose the regularised-regression HTML report."""

from __future__ import annotations

import html

from regression_pack_core import plotting, reports
from regression_pack_core.schemas import RegularizedRegressionReport


def _feature_selection_html(report: RegularizedRegressionReport) -> str:
    fs = report.feature_selection
    if fs is None:
        return ""
    selected_li = "".join(f"<li><code>{html.escape(f)}</code></li>" for f in fs.selected_features)
    dropped_li = (
        "".join(f"<li><code>{html.escape(f)}</code></li>" for f in fs.dropped_features)
        if fs.dropped_features
        else "<li><em>none</em></li>"
    )
    return reports.section(
        "Feature selection",
        f"<p>{fs.n_selected} feature(s) retained, {fs.n_dropped} zeroed by regularisation.</p>"
        f"<h3 style='font-size:13px;margin:12px 0 4px;'>Retained</h3><ul>{selected_li}</ul>"
        f"<h3 style='font-size:13px;margin:12px 0 4px;'>Zeroed (dropped)</h3><ul>{dropped_li}</ul>",
    )


def _ols_comparison_html(report: RegularizedRegressionReport) -> str:
    comp = report.comparison_to_ols
    if not comp or "error" in comp:
        if comp and "error" in comp:
            return reports.section(
                "OLS comparison",
                f'<p style="color:var(--ink-soft);">{html.escape(comp["error"])}</p>',
            )
        return ""

    method = report.method.upper()
    rows = []
    for feat in comp["ols_coef"]:
        ols_v = comp["ols_coef"][feat]
        reg_v = comp["reg_coef"].get(feat, 0.0)
        delta = reg_v - ols_v
        pct = f"{delta / abs(ols_v) * 100:+.1f}%" if abs(ols_v) > 1e-10 else "—"
        rows.append(
            f"<tr><td>{html.escape(feat)}</td>"
            f'<td class="num">{ols_v:.4g}</td>'
            f'<td class="num">{reg_v:.4g}</td>'
            f'<td class="num">{pct}</td></tr>'
        )
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in comp.get("notes", []))
    shrinkage = comp.get("coef_shrinkage_mean", "—")
    body = (
        reports.stat_grid([
            {"label": "OLS R²", "value": f"{comp['ols_r2']:.4f}"},
            {"label": f"{method} R²", "value": f"{comp['reg_r2']:.4f}"},
            {"label": "Mean |shrinkage|", "value": f"{shrinkage:.4g}"},
        ])
        + f'<table class="data-table"><thead><tr>'
        f"<th>Feature</th><th>OLS β</th><th>{html.escape(method)} β</th><th>Δ%</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        + (f"<ul style='font-size:13px;margin-top:10px;'>{notes_html}</ul>" if notes_html else "")
    )
    return reports.section("OLS comparison", body)


def _interpretation_blocks(report: RegularizedRegressionReport) -> str:
    if not report.interpretations:
        return '<p style="color:var(--ink-soft);">No non-intercept coefficients to interpret.</p>'
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
    report: RegularizedRegressionReport,
    *,
    target: str,
    dataset_name: str,
    timestamp: str,
) -> str:
    fs = report.fit_statistics
    method_label = report.method.capitalize()

    # Verdict
    r2 = fs.r_squared
    if r2 >= 0.8:
        tone, label = "ok", f"Strong fit (R² = {r2:.3f})"
    elif r2 >= 0.5:
        tone, label = "warn", f"Moderate fit (R² = {r2:.3f})"
    else:
        tone, label = "fail", f"Weak fit (R² = {r2:.3f})"
    verdict = {"tone": tone, "label": label, "headline": report.headline}

    # 1. Fit summary
    fit_stats = reports.stat_grid([
        {"label": "Observations", "value": f"{fs.n_observations:,}"},
        {"label": "Features", "value": str(fs.n_features)},
        {"label": "Method", "value": method_label},
        {"label": "α (selected)", "value": f"{report.selected_alpha:.4g}"},
        {"label": "R²", "value": f"{fs.r_squared:.4f}"},
        {"label": "Adj. R²", "value": f"{fs.adj_r_squared:.4f}"},
        {"label": "Residual SE", "value": f"{fs.residual_std_error:.4g}"},
        *([{"label": "l1_ratio", "value": f"{report.selected_l1_ratio:.2f}"}]
          if report.selected_l1_ratio is not None else []),
    ])
    fit_section = reports.section(
        "Fit summary",
        fit_stats + f"<p>{html.escape(report.headline)}</p>",
    )

    # 2. Coefficients — non-zero, sorted by |β|
    const_rows = [c for c in report.coefficients if c.feature == "const"]
    others = sorted(
        (c for c in report.coefficients if c.feature != "const" and abs(c.coefficient) > 1e-10),
        key=lambda c: abs(c.coefficient),
        reverse=True,
    )
    table = reports.coefficient_table_html(const_rows + others)

    first_plot = True

    forest = plotting.coef_forest(
        [c.feature for c in others],
        [c.coefficient for c in others],
        [c.ci_lower for c in others],
        [c.ci_upper for c in others],
        title=f"{method_label} coefficients (95% CI, approx)",
    )
    forest_html = plotting.to_inline_html(forest, "coef-forest", include_js=first_plot)
    first_plot = False

    coef_section = reports.section(
        "Coefficients",
        table + f'<div class="plot-container">{forest_html}</div>',
    )

    # 3. Regularisation path
    path_fig = plotting.regularization_path(
        report.path,
        selected_alpha=report.selected_alpha,
        title="Regularisation path",
    )
    path_html = plotting.to_inline_html(path_fig, "reg-path", include_js=first_plot)
    first_plot = False

    path_section = reports.section(
        "Regularisation path",
        f'<div class="plot-container">{path_html}</div>'
        "<p style='font-size:12px;color:var(--ink-soft);'>Each line is one feature. "
        "Dashed orange line marks the CV-selected α.</p>",
    )

    # 4. CV curve
    cv_fig = plotting.cv_curve(report.cv_curve, title="Cross-validation score vs α")
    cv_html = plotting.to_inline_html(cv_fig, "cv-curve", include_js=first_plot)

    cv_section = reports.section(
        "Cross-validation",
        f'<div class="plot-container">{cv_html}</div>'
        f"<p style='font-size:12px;color:var(--ink-soft);'>Scoring: {report.cv_curve.scoring}. "
        "Shaded band = ±1 SD. "
        f"Selected α = {report.selected_alpha:.4g}"
        + (f"; 1-SE α = {report.cv_curve.alpha_1se:.4g}" if report.cv_curve.alpha_1se else "")
        + ".</p>",
    )

    # 5. Interpretation
    interp_section = reports.section("Interpretation", _interpretation_blocks(report))

    # 6. Feature selection (lasso/elasticnet only)
    fs_section = _feature_selection_html(report)

    # 7. OLS comparison
    ols_section = _ols_comparison_html(report)

    # 8. Flags & recommendations
    flags_section = reports.section(
        "Flags & recommendations", reports.flag_list_html(report.flags)
    )

    body = (
        fit_section
        + coef_section
        + path_section
        + cv_section
        + interp_section
        + fs_section
        + ols_section
        + flags_section
    )

    return reports.render_html_report(
        title=f"{method_label} regression: {target}",
        skill_name="regularized-regression",
        dataset_name=dataset_name,
        n_obs=fs.n_observations,
        timestamp=timestamp,
        verdict=verdict,
        body_html=body,
    )
