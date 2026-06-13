"""Compose the model-comparison HTML report."""

from __future__ import annotations

import html

import plotly.graph_objects as go

from regression_pack_core import plotting, reports
from regression_pack_core.schemas import ModelComparisonReport

_ACCENT = "#0F766E"
_INK_SOFT = "#6B7280"
_WINNER_BG = "#F0FDF4"


def _models_table_html(report: ModelComparisonReport) -> str:
    recommended = report.verdict.recommended_model

    header_cols = [
        "Model", "Family", "n", "k",
        "Primary metric", "AIC", "BIC", "Notes",
    ]
    header = "".join(f"<th>{c}</th>" for c in header_cols)

    rows = []
    for e in report.models:
        is_winner = e.name == recommended
        bg = f'style="background:{_WINNER_BG};"' if is_winner else ""
        winner_badge = ' <span class="pill ok">recommended</span>' if is_winner else ""
        aic_str = f"{e.aic:.1f}" if e.aic is not None else "—"
        bic_str = f"{e.bic:.1f}" if e.bic is not None else "—"
        notes_str = "; ".join(e.notes) if e.notes else "—"
        rows.append(
            f"<tr {bg}>"
            f"<td><strong>{html.escape(e.name)}</strong>{winner_badge}</td>"
            f"<td>{html.escape(e.family)}</td>"
            f'<td class="num">{e.n_observations:,}</td>'
            f'<td class="num">{e.n_features}</td>'
            f'<td class="num">{e.fit_quality_primary:.4f}</td>'
            f'<td class="num">{aic_str}</td>'
            f'<td class="num">{bic_str}</td>'
            f"<td style=\"font-size:12px;\">{html.escape(notes_str)}</td>"
            "</tr>"
        )

    return (
        '<table class="data-table"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _akaike_bar_fig(report: ModelComparisonReport) -> go.Figure | None:
    aw = report.akaike_weights
    if aw is None:
        return None

    fig = plotting.figure("Akaike weights (higher = more evidence)")
    fig.add_trace(
        go.Bar(
            x=aw.model_names,
            y=aw.weights,
            marker_color=[
                _ACCENT if n == report.verdict.recommended_model else "#94A3B8"
                for n in aw.model_names
            ],
            text=[f"{w:.3f}" for w in aw.weights],
            textposition="outside",
            hovertemplate="%{x}<br>weight = %{y:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1.05], title_text="Akaike weight")
    fig.update_xaxes(title_text="Model")
    return fig


def _delta_aic_fig(report: ModelComparisonReport) -> go.Figure | None:
    aw = report.akaike_weights
    if aw is None:
        return None

    fig = plotting.figure("Δ AIC (lower = better; Δ < 2 = competitive)")
    fig.add_trace(
        go.Bar(
            x=aw.model_names,
            y=aw.delta_aic,
            marker_color=[
                _ACCENT if d == 0.0 else "#94A3B8" for d in aw.delta_aic
            ],
            text=[f"{d:.2f}" for d in aw.delta_aic],
            textposition="outside",
            hovertemplate="%{x}<br>Δ AIC = %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=2, line=dict(color=_INK_SOFT, width=1, dash="dash"))
    fig.update_yaxes(title_text="Δ AIC from best")
    fig.update_xaxes(title_text="Model")
    return fig


def _lr_tests_html(report: ModelComparisonReport) -> str:
    if not report.lr_tests:
        return '<p style="color:var(--ink-soft);">No nested model pairs detected.</p>'

    rows = []
    for t in report.lr_tests:
        sig = "***" if t.p_value < 0.001 else ("**" if t.p_value < 0.01 else ("*" if t.p_value < 0.05 else ""))
        rows.append(
            f"<tr><td>{html.escape(t.nested_model)}</td>"
            f"<td>{html.escape(t.full_model)}</td>"
            f'<td class="num">{t.likelihood_ratio:.4f}</td>'
            f'<td class="num">{t.df}</td>'
            f'<td class="num">{t.p_value:.4g}</td>'
            f'<td class="sig-stars">{sig}</td>'
            f"<td style=\"font-size:12px;\">{html.escape(t.conclusion)}</td></tr>"
        )

    return (
        '<table class="data-table"><thead><tr>'
        "<th>Nested</th><th>Full</th><th>LR stat</th><th>df</th>"
        "<th>p-value</th><th></th><th>Conclusion</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + '<p style="font-size:12px;color:var(--ink-soft);margin-top:6px;">'
        "*** p&lt;0.001 ** p&lt;0.01 * p&lt;0.05</p>"
    )


def _coef_comparison_fig(raw_reports: list[dict]) -> go.Figure | None:
    """Side-by-side coefficient bars for linear-family models."""
    linear_reports = [r for r in raw_reports if r["_family"] in ("linear", "ridge", "lasso", "elasticnet")]
    if not linear_reports:
        return None

    # Union of all features across models
    all_feats: list[str] = []
    seen: set[str] = set()
    for r in linear_reports:
        for f in r["_features"]:
            if f not in seen:
                all_feats.append(f)
                seen.add(f)

    model_names = [r["_name"] for r in linear_reports]
    coefs_by_model: dict[str, list[float | None]] = {}
    for r in linear_reports:
        coef_map = {c["feature"]: c.get("coefficient", 0.0) for c in r.get("coefficients", [])}
        coefs_by_model[r["_name"]] = [coef_map.get(f) for f in all_feats]

    return plotting.coef_comparison(model_names, all_feats, coefs_by_model)


def render_report(
    report: ModelComparisonReport,
    raw_reports: list[dict],
    *,
    dataset_name: str,
    timestamp: str,
) -> str:
    v = report.verdict

    tone_map = {
        "clear_winner": "ok",
        "competitive_tie": "warn",
        "complementary_strengths": "warn",
        "all_inadequate": "fail",
    }
    tone = tone_map.get(v.overall, "warn")
    verdict_card = {"tone": tone, "label": v.overall.replace("_", " ").title(), "headline": v.headline}

    first_plot = True

    # 1. Verdict summary
    rationale_html = (
        f'<div style="background:var(--bg-soft);border-radius:6px;padding:14px 16px;">'
        f'<p style="font-size:14px;margin:0;">{html.escape(v.rationale)}</p>'
        "</div>"
    )
    rec_html = ""
    if v.recommended_model:
        rec_html = (
            f'<p><strong>Recommended model:</strong> '
            f'{html.escape(v.recommended_model)}</p>'
        )
    verdict_section = reports.section("Verdict", rec_html + rationale_html)

    # 2. Models compared table
    table_section = reports.section("Models compared", _models_table_html(report))

    # 3. Akaike weights
    aic_html = ""
    aw_fig = _akaike_bar_fig(report)
    delta_fig = _delta_aic_fig(report)
    if aw_fig is not None:
        aw_html = plotting.to_inline_html(aw_fig, "akaike-weights", include_js=first_plot)
        first_plot = False
        aic_html = f'<div class="plot-container">{aw_html}</div>'
    if delta_fig is not None:
        d_html = plotting.to_inline_html(delta_fig, "delta-aic", include_js=first_plot)
        first_plot = False
        aic_html += f'<div class="plot-container">{d_html}</div>'

    if aic_html:
        aic_note = (
            '<p style="font-size:12px;color:var(--ink-soft);">'
            "Akaike weight = exp(−Δ/2) / Σ exp(−Δ/2). "
            "Weight > 0.80 indicates a clear winner. "
            "Δ AIC &lt; 2 indicates models are statistically indistinguishable.</p>"
        )
        akaike_section = reports.section("AIC / Akaike weights", aic_html + aic_note)
    else:
        akaike_section = reports.section(
            "AIC / Akaike weights",
            '<p style="color:var(--ink-soft);">AIC comparison not available '
            "(models may differ in target variable or family).</p>",
        )

    # 4. LR tests
    lr_section = reports.section("Likelihood-ratio tests (nested pairs)", _lr_tests_html(report))

    # 5. Coefficient comparison
    coef_fig = _coef_comparison_fig(raw_reports)
    if coef_fig is not None:
        coef_fig_html = plotting.to_inline_html(coef_fig, "coef-comparison", include_js=first_plot)
        first_plot = False
        coef_section = reports.section(
            "Coefficient comparison",
            f'<div class="plot-container">{coef_fig_html}</div>'
            '<p style="font-size:12px;color:var(--ink-soft);">'
            "Coefficients on original feature scale. Features absent from a model shown as 0.</p>",
        )
    else:
        coef_section = ""

    # 6. Flags
    flags_section = (
        reports.section("Flags & recommendations", reports.flag_list_html(report.flags))
        if report.flags
        else ""
    )

    body = (
        verdict_section
        + table_section
        + akaike_section
        + lr_section
        + coef_section
        + flags_section
    )

    n_obs = report.models[0].n_observations if report.models else 0
    return reports.render_html_report(
        title="Model comparison",
        skill_name="model-comparison",
        dataset_name=dataset_name,
        n_obs=n_obs,
        timestamp=timestamp,
        verdict=verdict_card,
        body_html=body,
    )
