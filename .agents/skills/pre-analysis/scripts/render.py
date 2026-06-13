"""Compose the pre-analysis HTML report."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from regression_pack_core import plotting, reports
from regression_pack_core.schemas import PreAnalysisReport, Severity


def _tone(report: PreAnalysisReport) -> str:
    severities = {f.severity for f in report.flags}
    if Severity.HIGH in severities:
        return "fail"
    if Severity.WARN in severities:
        return "warn"
    return "ok"


def _headline(report: PreAnalysisReport) -> str:
    tone = _tone(report)
    rec = report.modeling_recommendations
    estimator = rec.get("preferred_estimator", "linear_regression").replace("_", "-")
    high_flags = [f for f in report.flags if f.severity == Severity.HIGH]
    if tone == "fail":
        msg = high_flags[0].message if high_flags else "Multiple high-severity issues found."
        return f"Data needs remediation before modelling — {msg}"
    if tone == "warn":
        n_warn = len(report.flags)
        return f"Data is mostly ready; review {n_warn} warning(s) before fitting."
    transform = rec.get("transform_target")
    suffix = f" Consider {transform.replace('_', ' ')} on the target." if transform else ""
    return f"Data looks clean; proceed with {estimator}.{suffix}"


def _target_section(
    report: PreAnalysisReport, y_raw: pd.Series, first_plot: bool
) -> tuple[str, bool]:
    ta = report.target
    stats_items = [
        {"label": "Type", "value": ta.type},
        {"label": "N", "value": str(report.n_samples)},
        {"label": "Missing", "value": str(ta.n_missing)},
    ]
    if ta.skewness is not None:
        stats_items.append({"label": "Skewness", "value": f"{ta.skewness:.3f}"})
    if ta.kurtosis is not None:
        stats_items.append({"label": "Kurtosis", "value": f"{ta.kurtosis:.3f}"})
    if ta.outlier_count is not None:
        stats_items.append({"label": "Outliers (IQR)", "value": str(ta.outlier_count)})
    grid = reports.stat_grid(stats_items)

    plots_html = ""
    y_clean = y_raw.dropna()
    if pd.api.types.is_numeric_dtype(y_clean):
        fig1 = plotting.histogram(
            y_clean.tolist(), x_label=ta.name, title=f"{ta.name} distribution"
        )
        div1 = plotting.to_inline_html(fig1, "target-hist", include_js=first_plot)
        first_plot = False
        if "log_transform" in ta.recommendations and (y_clean > 0).all():
            log_y = np.log(y_clean.to_numpy(dtype=float)).tolist()
            fig2 = plotting.histogram(
                log_y, x_label=f"log({ta.name})", title=f"log({ta.name}) distribution"
            )
            div2 = plotting.to_inline_html(fig2, "target-log-hist", include_js=False)
            plots_html = (
                f'<div class="plot-container grid-2"><div>{div1}</div><div>{div2}</div></div>'
            )
        else:
            plots_html = f'<div class="plot-container">{div1}</div>'

    recs_html = ""
    if ta.recommendations:
        items = "".join(f"<li>{html.escape(r)}</li>" for r in ta.recommendations)
        recs_html = (
            '<p style="font-size:13px;"><strong>Recommendations:</strong></p>'
            f'<ul style="font-size:13px;">{items}</ul>'
        )

    return reports.section("Target", grid + plots_html + recs_html), first_plot


def _features_section(report: PreAnalysisReport) -> str:
    rows = []
    for fa in report.features:
        flag_pills = " ".join(
            f'<span class="pill warn" style="font-size:10px;">{html.escape(f)}</span>'
            for f in fa.flags
        ) or "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(fa.name)}</td>"
            f"<td>{html.escape(fa.type)}</td>"
            f'<td class="num">{fa.missing_pct * 100:.1f}%</td>'
            f'<td class="num">{fa.n_unique}</td>'
            f"<td>{flag_pills}</td>"
            "</tr>"
        )
    table = (
        '<table class="data-table"><thead><tr>'
        "<th>Feature</th><th>Type</th><th>Missing %</th><th>Unique</th><th>Flags</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    return reports.section("Features", table)


def _relationships_section(
    report: PreAnalysisReport,
    nonlinearity_plots: dict[str, dict],
    first_plot: bool,
) -> tuple[str, bool]:
    if not report.suspected_nonlinearity:
        return (
            reports.section(
                "Relationships", "<p>No suspected nonlinear relationships detected.</p>"
            ),
            first_plot,
        )

    divs = []
    for i, feat in enumerate(report.suspected_nonlinearity):
        if feat not in nonlinearity_plots:
            continue
        data = nonlinearity_plots[feat]
        fig = plotting.scatter(
            data["x"],
            data["y"],
            x_label=feat,
            y_label=report.target.name,
            title=f"{feat} vs {report.target.name}",
        )
        fig.add_trace(
            go.Scatter(
                x=data["lowess_x"],
                y=data["lowess_y"],
                mode="lines",
                line=dict(color=plotting.WARN, width=2),
                name="LOWESS",
                hoverinfo="skip",
            )
        )
        caption = (
            f'<p style="font-size:12px;color:var(--ink-soft);">LOWESS deviates from linear'
            f" — consider a polynomial term for <strong>{html.escape(feat)}</strong>.</p>"
        )
        div = f"<div>{plotting.to_inline_html(fig, f'nl-{i}', include_js=first_plot)}{caption}</div>"
        divs.append(div)
        first_plot = False

    body = (
        f'<div class="plot-container grid-2">{"".join(divs)}</div>'
        if divs
        else "<p>No plot data available.</p>"
    )
    return reports.section("Relationships", body), first_plot


def _multicollinearity_section(
    report: PreAnalysisReport, first_plot: bool
) -> tuple[str, bool]:
    mc = report.multicollinearity
    names = mc.get("correlation_names", [])
    matrix = mc.get("matrix", [])
    heatmap_html = ""

    if names and matrix and len(names) >= 2:
        fig = plotting.figure("Correlation matrix")
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=names,
                y=names,
                colorscale=[
                    [0.0, plotting.FAIL],
                    [0.5, plotting.BG],
                    [1.0, plotting.ACCENT],
                ],
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in matrix],
                texttemplate="%{text}",
                hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(height=max(350, 40 * len(names)))
        heatmap_html = (
            f'<div class="plot-container">'
            f"{plotting.to_inline_html(fig, 'corr-heatmap', include_js=first_plot)}"
            f"</div>"
        )
        first_plot = False

    vif_dict = mc.get("vif_by_feature", {})
    if vif_dict:
        sorted_vif = sorted(vif_dict.items(), key=lambda kv: kv[1], reverse=True)
        rows = []
        for fname, vif_val in sorted_vif:
            if vif_val > 10:
                badge = ' <span class="pill fail" style="font-size:10px;">SEVERE</span>'
            elif vif_val > 5:
                badge = ' <span class="pill warn" style="font-size:10px;">HIGH</span>'
            else:
                badge = ""
            rows.append(
                f"<tr><td>{html.escape(fname)}</td>"
                f'<td class="num">{vif_val:.2f}{badge}</td></tr>'
            )
        vif_table = (
            '<table class="data-table" style="max-width:400px;"><thead><tr>'
            "<th>Feature</th><th>VIF</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    else:
        vif_table = "<p>VIF not computed (fewer than 2 numeric features).</p>"

    flagged = mc.get("flagged", [])
    note = ""
    if flagged:
        note = (
            f'<p style="font-size:13px;">Features with VIF &gt; 5: '
            f"<strong>{html.escape(', '.join(flagged))}</strong>. "
            "Consider dropping one of each collinear pair or switching to "
            "<code>regularized-regression --method ridge</code>.</p>"
        )

    return reports.section("Multicollinearity", heatmap_html + vif_table + note), first_plot


def _recommendations_section(report: PreAnalysisReport) -> str:
    rec = report.modeling_recommendations
    items = []

    if rec.get("transform_target"):
        t = rec["transform_target"]
        cli_flags = {"log_transform": "--log-target", "box_cox": "--log-target"}
        cli = cli_flags.get(t, f"--{t.replace('_', '-')}")
        items.append(
            f"<li><strong>Transform target:</strong> apply <code>{html.escape(t)}</code> "
            f"to <em>{html.escape(report.target.name)}</em> before fitting. "
            f"Pass <code>{html.escape(cli)}</code> to the fit skill.</li>"
        )

    if rec.get("consider_polynomial"):
        feats = ", ".join(rec["consider_polynomial"])
        items.append(
            f"<li><strong>Polynomial terms:</strong> consider adding polynomial features for "
            f"<code>{html.escape(feats)}</code> — LOWESS suggests nonlinearity.</li>"
        )

    if rec.get("drop_or_combine"):
        feats = ", ".join(rec["drop_or_combine"])
        items.append(
            f"<li><strong>Multicollinearity:</strong> drop or combine "
            f"<code>{html.escape(feats)}</code>, or use "
            f"<code>regularized-regression --method ridge</code>.</li>"
        )

    skill_map = {
        "linear_regression": "linear-regression",
        "logistic_regression": "logistic-regression",
        "regularized_regression": "regularized-regression",
    }
    estimator = rec.get("preferred_estimator", "linear_regression")
    skill = skill_map.get(estimator, estimator.replace("_", "-"))
    items.append(
        f"<li><strong>Suggested next skill:</strong> <code>{html.escape(skill)}</code></li>"
    )

    flags_html = reports.flag_list_html(report.flags)
    recs_html = f'<ol style="font-size:14px;">{"".join(items)}</ol>' if items else ""
    return reports.section("Recommendations", flags_html + recs_html)


def render_report(
    report: PreAnalysisReport,
    *,
    y_raw: pd.Series,
    X_raw: pd.DataFrame,
    nonlinearity_plots: dict[str, dict],
    dataset_name: str,
    n_obs: int,
    timestamp: str,
) -> str:
    tone = _tone(report)
    verdict = {
        "tone": tone,
        "label": f"Pre-analysis: {'ready' if tone == 'ok' else tone}",
        "headline": _headline(report),
    }

    first_plot = True
    target_sec, first_plot = _target_section(report, y_raw, first_plot)
    features_sec = _features_section(report)
    rel_sec, first_plot = _relationships_section(report, nonlinearity_plots, first_plot)
    mc_sec, first_plot = _multicollinearity_section(report, first_plot)
    rec_sec = _recommendations_section(report)

    body = target_sec + features_sec + rel_sec + mc_sec + rec_sec
    return reports.render_html_report(
        title="Pre-analysis report",
        skill_name="pre-analysis",
        dataset_name=dataset_name,
        n_obs=n_obs,
        timestamp=timestamp,
        verdict=verdict,
        body_html=body,
    )
