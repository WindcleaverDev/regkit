"""Plotly theme and chart helpers for the regression pack.

Every plot in every skill is built through these helpers so the visual
language stays consistent. Constants must match the design tokens in
style.css exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

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


def to_inline_html(fig: go.Figure, div_id: str | None = None) -> str:
    """Render a figure to inline HTML — Plotly JS inlined, no CDN.

    Mode bar hidden, responsive.
    """
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
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
