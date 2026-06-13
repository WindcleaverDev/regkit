"""Plotly theme and chart helpers for the regression pack.

Every plot in every skill is built through these helpers so the visual
language stays consistent. Constants must match the design tokens in
style.css exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

if TYPE_CHECKING:
    from regression_pack_core.schemas import (
        CalibrationData,
        CVCurve,
        MarginalEffect,
        RegularizationPath,
        ROCData,
    )

# Constants — must match the design tokens table exactly
ACCENT = "#0F766E"
ACCENT_LIGHT = "#5EEAD4"
INK = "#1F2937"
INK_SOFT = "#6B7280"
GRID = "#E5E7EB"
BG = "#FFFFFF"
BG_SOFT = "#F9FAFB"
OK = "#059669"
WARN = "#D97706"
FAIL = "#DC2626"
FONT_FAMILY = "Inter, IBM Plex Sans, system-ui, -apple-system, sans-serif"


def figure(title: str | None = None) -> go.Figure:
    """Return a Plotly figure with the pack theme applied.

    Used as the starting point for every chart.
    """
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        font=dict(family=FONT_FAMILY, size=12, color=INK),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=60, r=24, t=48 if title else 24, b=48),
        hoverlabel=dict(
            bgcolor=INK,
            font=dict(family=FONT_FAMILY, size=12, color=BG),
            bordercolor=INK,
        ),
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=INK_SOFT, size=11),
        title_font=dict(color=INK_SOFT, size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=INK_SOFT, size=11),
        title_font=dict(color=INK_SOFT, size=12),
    )
    return fig


def to_inline_html(fig: go.Figure, div_id: str | None = None, *, include_js: bool = True) -> str:
    """Render a figure to inline HTML — Plotly JS inlined, no CDN.

    Mode bar hidden, responsive. Plotly JS is ~4.5 MB: in a report with
    several figures, pass include_js=True for the first figure only and
    include_js=False for the rest, or the library is embedded once per plot.
    """
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True if include_js else False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
        default_height=420,
    )


def _lowess_trace(x: Sequence[float], y: Sequence[float]) -> go.Scatter:
    """A LOWESS smoother line through (x, y)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    smoothed = lowess(np.asarray(y, dtype=float), np.asarray(x, dtype=float), frac=0.6)
    return go.Scatter(
        x=smoothed[:, 0],
        y=smoothed[:, 1],
        mode="lines",
        line=dict(color=WARN, width=2),
        name="LOWESS",
        hoverinfo="skip",
    )


def scatter(
    x,
    y,
    *,
    x_label: str = "",
    y_label: str = "",
    hover_text=None,
    title: str | None = None,
) -> go.Figure:
    fig = figure(title)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color=ACCENT, size=7, opacity=0.65),
            text=hover_text,
            hoverinfo="text" if hover_text is not None else None,
        )
    )
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


def line(
    x,
    y,
    *,
    x_label: str = "",
    y_label: str = "",
    title: str | None = None,
    color: str = ACCENT,
) -> go.Figure:
    fig = figure(title)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2)))
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


def histogram(x, *, x_label: str = "", title: str | None = None, bins: int = 30) -> go.Figure:
    fig = figure(title)
    fig.add_trace(go.Histogram(x=x, nbinsx=bins, marker=dict(color=ACCENT, opacity=0.8)))
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text="Count")
    return fig


def coef_forest(
    features: list[str],
    coefs: list[float],
    lower: list[float],
    upper: list[float],
    *,
    title: str | None = None,
) -> go.Figure:
    """Coefficient forest plot.

    Markers at coefs, horizontal error bars from lower to upper, vertical
    dotted line at x=0. Y-axis reversed so first feature appears at top.
    """
    fig = figure(title)
    fig.add_trace(
        go.Scatter(
            x=coefs,
            y=features,
            mode="markers",
            marker=dict(color=ACCENT, size=9),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[u - c for c, u in zip(coefs, upper, strict=True)],
                arrayminus=[c - lo for c, lo in zip(coefs, lower, strict=True)],
                color=INK_SOFT,
                thickness=1.5,
                width=4,
            ),
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=INK_SOFT, width=1, dash="dot"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Coefficient (95% CI)")
    return fig


def qq_plot(residuals, *, title: str | None = None) -> go.Figure:
    """QQ plot of residuals vs theoretical normal quantiles."""
    resid = np.sort(np.asarray(residuals, dtype=float))
    n = len(resid)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    # Reference line through the quartiles (as in R's qqline)
    q25, q75 = np.percentile(resid, [25, 75])
    t25, t75 = stats.norm.ppf([0.25, 0.75])
    slope = (q75 - q25) / (t75 - t25)
    intercept = q25 - slope * t25

    fig = figure(title)
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=resid,
            mode="markers",
            marker=dict(color=ACCENT, size=6, opacity=0.65),
            hovertemplate="theoretical %{x:.2f}, sample %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[theoretical[0], theoretical[-1]],
            y=[intercept + slope * theoretical[0], intercept + slope * theoretical[-1]],
            mode="lines",
            line=dict(color=INK_SOFT, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title_text="Theoretical quantiles")
    fig.update_yaxes(title_text="Sample quantiles")
    return fig


def residuals_vs_fitted(fitted, residuals, *, title: str | None = None) -> go.Figure:
    """Scatter of residuals vs fitted with a LOWESS smoother overlay."""
    fig = scatter(fitted, residuals, x_label="Fitted values", y_label="Residuals", title=title)
    fig.add_hline(y=0, line=dict(color=INK_SOFT, width=1, dash="dot"))
    fig.add_trace(_lowess_trace(fitted, residuals))
    return fig


def scale_location(fitted, residuals, *, title: str | None = None) -> go.Figure:
    """sqrt(|standardized residuals|) vs fitted, with LOWESS overlay."""
    resid = np.asarray(residuals, dtype=float)
    std_resid = resid / resid.std(ddof=1)
    y = np.sqrt(np.abs(std_resid))
    fig = scatter(
        fitted, y, x_label="Fitted values", y_label="√|standardized residuals|", title=title
    )
    fig.add_trace(_lowess_trace(fitted, y))
    return fig


def leverage_plot(
    leverage,
    studentized_residuals,
    cooks_d=None,
    *,
    title: str | None = None,
) -> go.Figure:
    """Leverage vs studentized residuals; if cooks_d supplied, size markers by Cook's D."""
    fig = figure(title)
    if cooks_d is not None:
        cd = np.asarray(cooks_d, dtype=float)
        max_cd = cd.max() if cd.max() > 0 else 1.0
        sizes = 6 + 18 * (cd / max_cd)
        hover = [
            f"row {i}<br>leverage {lev:.3f}<br>stud. resid {sr:.2f}<br>Cook's D {c:.4f}"
            for i, (lev, sr, c) in enumerate(
                zip(leverage, studentized_residuals, cd, strict=True)
            )
        ]
    else:
        sizes = 7
        hover = [
            f"row {i}<br>leverage {lev:.3f}<br>stud. resid {sr:.2f}"
            for i, (lev, sr) in enumerate(zip(leverage, studentized_residuals, strict=True))
        ]
    fig.add_trace(
        go.Scatter(
            x=leverage,
            y=studentized_residuals,
            mode="markers",
            marker=dict(color=ACCENT, size=sizes, opacity=0.65),
            text=hover,
            hoverinfo="text",
        )
    )
    fig.add_hline(y=0, line=dict(color=INK_SOFT, width=1, dash="dot"))
    fig.update_xaxes(title_text="Leverage")
    fig.update_yaxes(title_text="Studentized residuals")
    return fig


# ─── Phase 2 plot helpers ─────────────────────────────────────────────────────


def roc_curve(roc: ROCData, *, title: str | None = None) -> go.Figure:
    """ROC curve with diagonal chance reference and AUC annotation."""
    fig = figure(title or "ROC curve")
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color=INK_SOFT, width=1, dash="dash"),
            hoverinfo="skip",
            name="Chance",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=roc.fpr, y=roc.tpr,
            mode="lines",
            line=dict(color=ACCENT, width=2),
            name=f"AUC = {roc.auc:.3f}",
            hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=0.98, y=0.06,
        xref="paper", yref="paper",
        text=f"AUC = {roc.auc:.3f}",
        showarrow=False,
        font=dict(size=13, color=ACCENT),
        xanchor="right",
    )
    fig.update_xaxes(title_text="False positive rate", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="True positive rate", range=[-0.02, 1.02])
    fig.update_layout(showlegend=False)
    return fig


def calibration_plot(cal: CalibrationData, *, title: str | None = None) -> go.Figure:
    """Reliability diagram — observed frequency vs predicted probability per bin."""
    fig = figure(title or f"Calibration (Brier = {cal.brier_score:.3f})")
    # Perfect calibration reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color=INK_SOFT, width=1, dash="dash"),
            hoverinfo="skip",
            name="Perfect calibration",
        )
    )
    hover = [
        f"bin centre {c:.2f}<br>observed {o:.3f}<br>n = {n}"
        for c, o, n in zip(cal.bin_centers, cal.observed_frequencies, cal.bin_counts, strict=True)
    ]
    fig.add_trace(
        go.Bar(
            x=cal.bin_centers,
            y=cal.observed_frequencies,
            marker=dict(color=ACCENT, opacity=0.75),
            text=hover,
            hoverinfo="text",
            name="Observed frequency",
            width=[0.08] * len(cal.bin_centers),
        )
    )
    fig.update_xaxes(title_text="Mean predicted probability", range=[-0.05, 1.05])
    fig.update_yaxes(title_text="Observed frequency", range=[-0.05, 1.05])
    fig.update_layout(showlegend=False)
    return fig


def regularization_path(
    path: RegularizationPath,
    selected_alpha: float | None = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """Coefficient paths: one line per feature, x = log10(α)."""
    fig = figure(title or "Regularisation path")
    log_alphas = np.log10(np.asarray(path.alphas, dtype=float))
    coefs = np.asarray(path.coefficients, dtype=float)  # shape (n_alphas, n_features)

    n_features = len(path.feature_names)
    # Sequential palette derived from ACCENT (teal family)
    palette = [
        f"hsl({h}, 60%, 45%)"
        for h in np.linspace(160, 200, n_features).tolist()
    ]
    for j, (fname, color) in enumerate(zip(path.feature_names, palette, strict=True)):
        fig.add_trace(
            go.Scatter(
                x=log_alphas.tolist(),
                y=coefs[:, j].tolist(),
                mode="lines",
                line=dict(color=color, width=1.5),
                name=fname,
                hovertemplate=f"{fname}<br>log α=%{{x:.2f}}<br>β=%{{y:.4g}}<extra></extra>",
            )
        )
    if selected_alpha is not None:
        fig.add_vline(
            x=float(np.log10(selected_alpha)),
            line=dict(color=WARN, width=1.5, dash="dash"),
            annotation_text=f"α={selected_alpha:.3g}",
            annotation_position="top right",
        )
    fig.update_xaxes(title_text="log₁₀(α)")
    fig.update_yaxes(title_text="Coefficient")
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=10)))
    return fig


def cv_curve(curve: CVCurve, *, title: str | None = None) -> go.Figure:
    """CV score vs log10(α) with ±1 SD band and selected-α markers."""
    fig = figure(title or f"CV {curve.scoring} vs α")
    log_alphas = np.asarray(curve.alphas, dtype=float)
    means = np.asarray(curve.mean_scores, dtype=float)
    stds = np.asarray(curve.std_scores, dtype=float)
    log_a = np.log10(log_alphas)

    # ±1 SD band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([log_a, log_a[::-1]]).tolist(),
            y=np.concatenate([means + stds, (means - stds)[::-1]]).tolist(),
            fill="toself",
            fillcolor="rgba(15,118,110,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="±1 SD",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=log_a.tolist(),
            y=means.tolist(),
            mode="lines",
            line=dict(color=ACCENT, width=2),
            hovertemplate="log α=%{x:.2f}<br>score=%{y:.4f}<extra></extra>",
            name="CV mean",
        )
    )
    fig.add_vline(
        x=float(np.log10(curve.selected_alpha)),
        line=dict(color=WARN, width=1.5, dash="dash"),
        annotation_text="selected α",
        annotation_position="top left",
    )
    if curve.alpha_1se is not None:
        fig.add_vline(
            x=float(np.log10(curve.alpha_1se)),
            line=dict(color=INK_SOFT, width=1, dash="dot"),
            annotation_text="1-SE α",
            annotation_position="top right",
        )
    fig.update_xaxes(title_text="log₁₀(α)")
    fig.update_yaxes(title_text=curve.scoring.replace("_", " "))
    fig.update_layout(showlegend=False)
    return fig


def marginal_effects_plot(
    effects: list[MarginalEffect], *, title: str | None = None
) -> go.Figure:
    """Forest plot of average marginal effects with 95% CIs."""
    features = [e.feature for e in effects]
    ames = [e.ame for e in effects]
    lowers = [e.ci_lower for e in effects]
    uppers = [e.ci_upper for e in effects]

    fig = figure(title or "Average marginal effects")
    fig.add_trace(
        go.Scatter(
            x=ames,
            y=features,
            mode="markers",
            marker=dict(color=ACCENT, size=9),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[u - a for a, u in zip(ames, uppers, strict=True)],
                arrayminus=[a - lo for a, lo in zip(ames, lowers, strict=True)],
                color=INK_SOFT,
                thickness=1.5,
                width=4,
            ),
            hovertemplate="%{y}: AME = %{x:.4g}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=INK_SOFT, width=1, dash="dot"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Average marginal effect on probability (95% CI)")
    return fig


def coef_comparison(
    model_names: list[str],
    features: list[str],
    coefs_by_model: dict[str, list[float]],
    *,
    title: str | None = None,
) -> go.Figure:
    """Grouped horizontal bars — one group per feature, one bar per model."""
    # Up to 4 models; palette uses ACCENT, ACCENT_LIGHT, and two muted variants
    palette = [ACCENT, ACCENT_LIGHT, "#6366F1", "#F59E0B"]
    fig = figure(title or "Coefficient comparison")
    for i, name in enumerate(model_names):
        coefs = coefs_by_model.get(name, [])
        # Pad/fill None where a feature wasn't in this model
        y_vals = [c if c is not None else 0.0 for c in coefs]
        fig.add_trace(
            go.Bar(
                x=y_vals,
                y=features,
                orientation="h",
                name=name,
                marker=dict(color=palette[i % len(palette)], opacity=0.8),
                hovertemplate=f"{name}<br>%{{y}}: %{{x:.4g}}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line=dict(color=INK_SOFT, width=1, dash="dot"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Coefficient")
    fig.update_layout(
        barmode="group",
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )
    return fig
