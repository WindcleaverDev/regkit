"""Compose the logistic-regression HTML report."""

from __future__ import annotations

import html

import plotly.graph_objects as go

from regression_pack_core import plotting, reports
from regression_pack_core.schemas import LogisticRegressionReport

_ACCENT = "#0F766E"
_INK_SOFT = "#6B7280"


def _cm_cell(v: int, label: str, color: str) -> str:
    return (
        f'<td style="text-align:center;padding:12px 20px;background:{color};border-radius:4px;">'
        f'<div style="font-size:22px;font-weight:700;">{v:,}</div>'
        f'<div style="font-size:11px;color:#4B5563;">{html.escape(label)}</div>'
        "</td>"
    )


def _confusion_matrix_html(cm: list[list[int]], positive_class: str) -> str:
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp or 1
    return (
        '<table style="border-collapse:separate;border-spacing:8px;margin:0 auto;">'
        "<thead><tr>"
        '<th style="text-align:center;font-size:12px;color:#6B7280;padding-bottom:4px;">'
        f"Predicted: not {html.escape(str(positive_class))}</th>"
        '<th style="text-align:center;font-size:12px;color:#6B7280;padding-bottom:4px;">'
        f"Predicted: {html.escape(str(positive_class))}</th>"
        "</tr></thead><tbody><tr>"
        + _cm_cell(tn, "True negative", "#F0FDF4")
        + _cm_cell(fp, "False positive", "#FEF2F2")
        + "</tr><tr>"
        + _cm_cell(fn, "False negative", "#FEF2F2")
        + _cm_cell(tp, "True positive", "#F0FDF4")
        + "</tr></tbody></table>"
        f'<p style="font-size:12px;color:#6B7280;text-align:center;margin-top:8px;">'
        f"n = {total:,}; positive class = {html.escape(str(positive_class))}</p>"
    )


def _significance(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _marginal_table_html(report: LogisticRegressionReport) -> str:
    rows = sorted(report.marginal_effects, key=lambda m: abs(m.ame), reverse=True)
    trs = "".join(
        f"<tr><td>{html.escape(m.feature)}</td>"
        f'<td class="num">{m.ame:+.4f}</td>'
        f'<td class="num">{m.std_error:.4f}</td>'
        f'<td class="num">[{m.ci_lower:+.4f}, {m.ci_upper:+.4f}]</td>'
        f'<td class="num">{m.p_value:.4g}</td>'
        f'<td class="sig-stars">{_significance(m.p_value)}</td></tr>'
        for m in rows
    )
    return (
        '<table class="data-table"><thead><tr>'
        "<th>Feature</th><th>AME</th><th>Std. Error</th><th>95% CI</th><th>p-value</th><th></th>"
        f"</tr></thead><tbody>{trs}</tbody></table>"
        '<p style="font-size:12px;color:var(--ink-soft);margin-top:6px;">'
        "AME = average marginal effect on P(positive class); *** p&lt;0.001 ** p&lt;0.01 * p&lt;0.05</p>"
    )


def _or_table_html(report: LogisticRegressionReport) -> str:
    non_const = [c for c in report.coefficients if c.feature != "const"]
    sorted_coefs = sorted(non_const, key=lambda c: abs(c.log_odds_coefficient), reverse=True)
    trs = "".join(
        f"<tr><td>{html.escape(c.feature)}</td>"
        f'<td class="num">{c.log_odds_coefficient:+.4f}</td>'
        f'<td class="num">{c.odds_ratio:.4f}</td>'
        f'<td class="num">{c.std_error:.4f}</td>'
        f'<td class="num">{c.z_stat:.3f}</td>'
        f'<td class="num">{c.p_value:.4g}</td>'
        f'<td class="num">[{c.ci_lower_odds_ratio:.4f}, {c.ci_upper_odds_ratio:.4f}]</td>'
        f'<td class="sig-stars">{_significance(c.p_value)}</td></tr>'
        for c in sorted_coefs
    )
    return (
        '<table class="data-table"><thead><tr>'
        "<th>Feature</th><th>log-odds (β)</th><th>OR exp(β)</th>"
        "<th>Std. Error</th><th>z</th><th>p-value</th><th>OR 95% CI</th><th></th>"
        f"</tr></thead><tbody>{trs}</tbody></table>"
    )


def _or_forest_fig(report: LogisticRegressionReport) -> go.Figure:
    non_const = [c for c in report.coefficients if c.feature != "const"]
    sorted_coefs = sorted(non_const, key=lambda c: c.odds_ratio)
    features = [c.feature for c in sorted_coefs]
    ors = [c.odds_ratio for c in sorted_coefs]
    ci_lo = [c.ci_lower_odds_ratio for c in sorted_coefs]
    ci_hi = [c.ci_upper_odds_ratio for c in sorted_coefs]

    fig = plotting.figure("Odds ratios (log scale, 95% CI)")
    fig.add_trace(
        go.Scatter(
            x=ors, y=features,
            mode="markers",
            marker=dict(color=_ACCENT, size=9),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[h - o for o, h in zip(ors, ci_hi, strict=True)],
                arrayminus=[o - lo for o, lo in zip(ors, ci_lo, strict=True)],
                color=_INK_SOFT, thickness=1.5, width=4,
            ),
            hovertemplate="%{y}: OR = %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=1.0, line=dict(color=_INK_SOFT, width=1, dash="dot"))
    fig.update_xaxes(type="log", title_text="Odds ratio (log scale)")
    fig.update_yaxes(autorange="reversed")
    return fig


def _interpretation_blocks(report: LogisticRegressionReport) -> str:
    if not report.interpretations:
        return '<p style="color:var(--ink-soft);">No interpretations available.</p>'
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
    report: LogisticRegressionReport,
    *,
    target: str,
    dataset_name: str,
    timestamp: str,
) -> str:
    fs = report.fit_statistics
    cs = report.classification_stats
    auc = report.roc.auc

    # Verdict
    has_convergence = any(f.code == "CONVERGENCE_ISSUE" for f in report.flags)
    has_imbalance = any(f.code == "CLASS_IMBALANCE" for f in report.flags)
    has_low_auc = any(f.code == "LOW_AUC" for f in report.flags)
    if has_convergence:
        tone = "fail"
    elif has_imbalance or has_low_auc:
        tone = "warn"
    else:
        tone = "ok"
    verdict = {"tone": tone, "label": f"AUC = {auc:.3f}", "headline": report.headline}

    # 1. Fit summary
    pseudo_r2 = fs.get("pseudo_r_squared", 0.0)
    fit_stats_html = reports.stat_grid([
        {"label": "Observations", "value": f"{fs['n_obs']:,}"},
        {"label": "Features", "value": str(fs["n_features"])},
        {"label": "Positive class", "value": f"{html.escape(str(report.positive_class))}"},
        {"label": "Positive rate", "value": f"{cs.class_balance * 100:.1f}%"},
        {"label": "AUC", "value": f"{auc:.4f}"},
        {"label": "Brier score", "value": f"{report.calibration.brier_score:.4f}"},
        {"label": "McFadden R²", "value": f"{pseudo_r2:.4f}"},
        {"label": "AIC", "value": f"{fs.get('aic', 0):.1f}"},
        {"label": "BIC", "value": f"{fs.get('bic', 0):.1f}"},
        {"label": "Converged", "value": "yes" if fs.get("converged", True) else "NO"},
    ])
    fit_section = reports.section(
        "Fit summary", fit_stats_html + f"<p>{html.escape(report.headline)}</p>"
    )

    first_plot = True

    # 2. Marginal effects — primary interpretation surface
    me_fig = plotting.marginal_effects_plot(
        report.marginal_effects, title="Average marginal effects on P(positive class)"
    )
    me_html = plotting.to_inline_html(me_fig, "marginal-effects", include_js=first_plot)
    first_plot = False
    marginal_section = reports.section(
        "Marginal effects",
        f'<div class="plot-container">{me_html}</div>' + _marginal_table_html(report),
    )

    # 3. Coefficients (log-odds + OR)
    or_fig = _or_forest_fig(report)
    or_fig_html = plotting.to_inline_html(or_fig, "or-forest", include_js=first_plot)
    coef_section = reports.section(
        "Coefficients (log-odds & odds ratios)",
        _or_table_html(report)
        + f'<div class="plot-container">{or_fig_html}</div>',
    )

    # 4. Classification performance
    cm_html = _confusion_matrix_html(cs.confusion_matrix, report.positive_class)
    perf_stats = reports.stat_grid([
        {"label": "Accuracy", "value": f"{cs.accuracy:.4f}"},
        {"label": "Balanced acc.", "value": f"{cs.balanced_accuracy:.4f}"},
        {"label": "Precision", "value": f"{cs.precision:.4f}"},
        {"label": "Recall", "value": f"{cs.recall:.4f}"},
        {"label": "F1", "value": f"{cs.f1:.4f}"},
        {"label": "Threshold", "value": f"{cs.threshold}"},
    ])
    classify_section = reports.section(
        "Classification performance", perf_stats + cm_html
    )

    # 5. ROC curve
    roc_fig = plotting.roc_curve(report.roc, title=f"ROC curve (AUC = {auc:.3f})")
    roc_html = plotting.to_inline_html(roc_fig, "roc-curve", include_js=first_plot)
    roc_section = reports.section(
        "ROC curve", f'<div class="plot-container">{roc_html}</div>'
    )

    # 6. Calibration
    cal_fig = plotting.calibration_plot(report.calibration)
    cal_html = plotting.to_inline_html(cal_fig, "calibration", include_js=first_plot)
    cal_section = reports.section(
        "Calibration",
        f'<div class="plot-container">{cal_html}</div>'
        f'<p style="font-size:12px;color:var(--ink-soft);">'
        f"Brier score = {report.calibration.brier_score:.4f}. "
        "Bars show observed positive rate per predicted-probability bin; "
        "dashed diagonal = perfect calibration.</p>",
    )

    # 7. Interpretation
    interp_section = reports.section("Interpretation", _interpretation_blocks(report))

    # 8. Flags
    flags_section = reports.section(
        "Flags & recommendations", reports.flag_list_html(report.flags)
    )

    body = (
        fit_section
        + marginal_section
        + coef_section
        + classify_section
        + roc_section
        + cal_section
        + interp_section
        + flags_section
    )

    return reports.render_html_report(
        title=f"Logistic regression: {target}",
        skill_name="logistic-regression",
        dataset_name=dataset_name,
        n_obs=fs["n_obs"],
        timestamp=timestamp,
        verdict=verdict,
        body_html=body,
    )
