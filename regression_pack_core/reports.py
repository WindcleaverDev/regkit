"""Jinja loader and HTML assembly helpers shared by all skill reports."""

from __future__ import annotations

import html
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from regression_pack_core.schemas import CoefficientRow, Flag

_PKG_DIR = Path(__file__).parent
_env = Environment(loader=FileSystemLoader(_PKG_DIR / "templates"), autoescape=False)


def _load_css() -> str:
    return (_PKG_DIR / "style.css").read_text()


def render_html_report(
    *,
    title: str,
    skill_name: str,
    dataset_name: str,
    n_obs: int,
    timestamp: str,
    verdict: dict | None,  # {"tone": "ok|warn|fail", "label": str, "headline": str}
    body_html: str,
) -> str:
    """Load templates/base_report.html.j2, inject inlined CSS from style.css,
    render. Returns the complete HTML document as a string.
    """
    template = _env.get_template("base_report.html.j2")
    return template.render(
        title=title,
        skill_name=skill_name,
        dataset_name=dataset_name,
        n_obs=n_obs,
        timestamp=timestamp,
        verdict=verdict,
        body=body_html,
        css=_load_css(),
    )


def section(title: str, body: str) -> str:
    return f'<div class="section"><h2>{html.escape(title)}</h2>{body}</div>'


def stat_grid(stats: list[dict]) -> str:
    """stats is a list of {"label": str, "value": str} (optional "sub")."""
    tiles = []
    for s in stats:
        sub = f'<div class="sub">{html.escape(str(s["sub"]))}</div>' if s.get("sub") else ""
        tiles.append(
            '<div class="stat">'
            f'<div class="label">{html.escape(str(s["label"]))}</div>'
            f'<div class="value">{html.escape(str(s["value"]))}</div>'
            f"{sub}</div>"
        )
    return f'<div class="stat-grid">{"".join(tiles)}</div>'


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def coefficient_table_html(coefs: list[CoefficientRow]) -> str:
    """Coefficient table with significance stars (*** p<0.001, ** p<0.01,
    * p<0.05, . p<0.1). All numbers right-aligned, tabular nums.
    """
    has_std = any(c.standardized_coefficient is not None for c in coefs)
    headers = ["Feature", "Coefficient", "Std. Error", "t", "p-value", "95% CI", ""]
    if has_std:
        headers.insert(2, "Std. β")

    rows = []
    for c in coefs:
        cells = [f"<td>{html.escape(c.feature)}</td>", f'<td class="num">{c.coefficient:.4g}</td>']
        if has_std:
            std = f"{c.standardized_coefficient:.4g}" if c.standardized_coefficient is not None else "—"
            cells.append(f'<td class="num">{std}</td>')
        cells += [
            f'<td class="num">{c.std_error:.4g}</td>',
            f'<td class="num">{c.t_stat:.3f}</td>',
            f'<td class="num">{c.p_value:.4g}</td>',
            f'<td class="num">[{c.ci_lower:.4g}, {c.ci_upper:.4g}]</td>',
            f'<td class="sig-stars">{significance_stars(c.p_value)}</td>',
        ]
        rows.append(f"<tr>{''.join(cells)}</tr>")

    head = "".join(f"<th>{h}</th>" for h in headers)
    note = (
        '<p style="font-size:12px;color:var(--ink-soft);margin-top:8px;">'
        "Significance: *** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05 &nbsp; . p&lt;0.1</p>"
    )
    return (
        f'<table class="data-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>{note}"
    )


def flag_list_html(flags: list[Flag]) -> str:
    """Render flags as styled .flag.info|.warn|.high blocks."""
    if not flags:
        return '<p style="color:var(--ink-soft);">No flags raised.</p>'
    blocks = []
    for f in flags:
        blocks.append(
            f'<div class="flag {f.severity.value}">'
            f'<span class="badge">{html.escape(f.severity.value)}</span>'
            f"<div><strong>{html.escape(f.code)}</strong> — {html.escape(f.message)}</div>"
            "</div>"
        )
    return "".join(blocks)
