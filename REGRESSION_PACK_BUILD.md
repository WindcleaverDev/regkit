# regression-pack — Build Specification

A pack of statistically rigorous regression skills for Claude. Hand this document to Claude Code (in Cursor) and build phase by phase. Each phase has a verifiable checkpoint.

---

## 0. What we're building

A repository of related Claude Skills for regression analysis. Each skill is a self-contained capability that:

1. Computes statistics deterministically (Python — `statsmodels`, `scikit-learn`, `scipy`)
2. Produces a Pydantic-validated structured output (JSON) suitable for chaining into other skills
3. Renders a standalone HTML report (Plotly + Jinja + inlined CSS, no CDN deps)
4. Returns *interpretation facts* (not prose) that Claude verbalises in chat

The LLM's role is constrained to the judgment layer: choosing transforms, narrating coefficients for the audience, deciding which diagnostic issue to surface first. It never does arithmetic, never reports p-values it didn't compute.

### Non-goals

- Not a general AutoML system. We don't pick algorithms by CV score.
- Not a notebook-first tool. Output is a deliverable HTML report, not a notebook.
- Not a deep-learning library. Regression family only (extensions later).

---

## 1. Stack & conventions

| | |
|---|---|
| Package manager | `uv` |
| Python | `>=3.13` |
| Build backend | `hatchling` (so `regression_pack_core` installs as an editable package) |
| Linting | `ruff` (line length 100) |
| Schema validation | `pydantic` v2 |
| Stats | `statsmodels` (primary), `scipy.stats` |
| ML | `scikit-learn` (for regularised models, CV) |
| Plotting | `plotly` only — no matplotlib unless explicitly needed for a diagnostic grid |
| Templating | `jinja2` |

### Naming conventions

- Skill directories use **hyphens**: `linear-regression/`, `diagnostics/`
- Python packages use **underscores**: `regression_pack_core/`
- Scripts inside skills are CLIs invoked by file path: `uv run python linear-regression/scripts/fit.py ...`
- Structured output files are always `report.json` + `report.html` in a user-chosen output directory

### Code style

- Use `from __future__ import annotations` in all Python files (optional on 3.13+, but kept for forward-compat and header consistency)
- Type hints on all public functions
- Pydantic models for all cross-script data; never pass raw dicts between scripts
- Plotly figures are built via helpers in `regression_pack_core.plotting`, never instantiated directly in skill scripts (this keeps the theme consistent)
- Every CLI script: `argparse`, has `--help`, exits non-zero on failure, prints a short success summary to stdout

---

## 2. Repository structure (full tree)

```
regression-pack/
├── pyproject.toml
├── README.md
├── .gitignore
├── BUILD.md                              # this document
├── regression_pack_core/                 # shared internal library
│   ├── __init__.py
│   ├── schemas.py                        # ALL Pydantic models
│   ├── plotting.py                       # Plotly theme + chart helpers
│   ├── validators.py                     # data validation
│   ├── reports.py                        # Jinja loader, HTML assembly
│   ├── style.css                         # shared report CSS
│   └── templates/
│       └── base_report.html.j2
├── linear-regression/
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── fit.py                        # main entry point (CLI)
│   │   ├── coefficients.py               # builds coefficient table
│   │   ├── interpret.py                  # generates InterpretationFact list
│   │   └── render.py                     # builds the HTML report
│   └── references/
│       ├── interpretation.md
│       └── robust_se.md
├── diagnostics/
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── diagnose.py                   # main entry point (CLI)
│   │   ├── residuals.py
│   │   ├── assumptions.py
│   │   ├── influence.py
│   │   ├── bias_variance.py
│   │   ├── nonlinearity.py
│   │   ├── triage.py
│   │   └── render.py
│   └── references/
│       ├── plots_guide.md
│       └── remediation.md
├── examples/
│   ├── synth_data.py                     # generates synthetic test datasets
│   ├── run_linear_regression.sh
│   └── run_diagnostics.sh
└── evals/                                # added with diagnostics phase
    ├── correctness/
    ├── detection/
    └── triggering/
```

Future skills (`pre-analysis/`, `logistic-regression/`, `regularized-regression/`, `model-comparison/`) follow the same shape.

---

## 3. Phase ordering & checkpoints

Build in this order. Do not start phase N+1 until phase N's checkpoint passes.

### Phase 1.0 — Project scaffold

Create:
- `pyproject.toml` (see §4)
- `.gitignore`
- `README.md` (brief — full README comes after phase 1.2)
- Empty directories per the tree above

**Checkpoint:** `uv sync` succeeds. Project has no Python errors.

### Phase 1.1 — `regression_pack_core`

Build the shared library in full (§5). All Pydantic schemas for the entire pack go in here now, even for skills not yet built — they're cheap to define and locking them down early prevents drift.

**Checkpoint:** `uv run python -c "from regression_pack_core import schemas, plotting, validators, reports; print('ok')"` prints `ok`.

### Phase 1.2 — `linear-regression` skill

Build per §6. Generate a synthetic dataset (§9), run the skill end-to-end.

**Checkpoint:** Running the example produces `report.json` validating against `LinearRegressionReport` and a renderable `report.html` opening in a browser without errors. The CLI prints a one-line headline matching the report's headline field.

### Phase 1.3 — `diagnostics` skill

Build per §7. Wire it to accept a `LinearRegressionReport` JSON as input plus the original data, and produce a `DiagnosticsReport`.

**Checkpoint:** Run on the same synthetic dataset; produces `diagnostics.json` validating against `DiagnosticsReport` and a `diagnostics.html` report. At least one planted assumption violation in the test data is detected with the correct status.

### Phase 1.4 — Smoke test + eval scaffolding

Add three synthetic datasets in `examples/synth_data.py`:
- Clean data (all assumptions satisfied)
- Heteroscedastic data (planted heteroscedasticity)
- Influential-point data (one obvious leverage outlier)

Add `evals/correctness/` with at least one textbook problem (Anscombe's quartet works well — same regression line, very different diagnostics).

**Checkpoint:** All three smoke tests produce the expected diagnostic verdicts. README updated with example output screenshots/links.

### Phase 2 (not in this document)

`pre-analysis`, `logistic-regression`, `regularized-regression`, `model-comparison`. To be specified after phase 1 ships.

---

## 4. `pyproject.toml` (exact)

```toml
[project]
name = "regression-pack"
version = "0.1.0"
description = "Statistically rigorous regression skills for Claude — structured outputs, diagnostic-aware reports, plain-English interpretation"
readme = "README.md"
requires-python = ">=3.13"
license = { text = "MIT" }

dependencies = [
    "pandas>=2.2",
    "numpy>=2.0",
    "scikit-learn>=1.5",
    "statsmodels>=0.14",
    "scipy>=1.13",
    "plotly>=5.18",
    "pydantic>=2.5",
    "jinja2>=3.1",
]

[dependency-groups]
dev = [
    "pytest>=7.4",
    "ruff>=0.4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["regression_pack_core"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]
ignore = ["E501"]
```

---

## 5. `regression_pack_core` (shared library)

### 5.1 Design tokens (single source of truth)

Both `plotting.py` and `style.css` consume these. If you change a colour, change it in both.

| Token | Value | Used for |
|---|---|---|
| accent | `#0F766E` | Primary accent (data marks, link colour, OK borders) |
| accent-light | `#5EEAD4` | Hover states, secondary highlights |
| ink | `#1F2937` | Primary text |
| ink-soft | `#6B7280` | Secondary text, axes, captions |
| grid | `#E5E7EB` | Gridlines, hairline borders |
| bg | `#FFFFFF` | Report background |
| bg-soft | `#F9FAFB` | Card / stat tile backgrounds |
| ok | `#059669` | Status: passed assumption |
| ok-soft | `#D1FAE5` | Background tint for OK flags |
| warn | `#D97706` | Status: caution / warning |
| warn-soft | `#FEF3C7` | Background tint for WARN flags |
| fail | `#DC2626` | Status: failed assumption |
| fail-soft | `#FEE2E2` | Background tint for FAIL flags |
| font-family | `'Inter', 'IBM Plex Sans', system-ui, -apple-system, sans-serif` | All text |

### 5.2 `schemas.py` — Pydantic models (full spec)

All cross-script data uses these. Define in this exact shape.

```python
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─── Shared primitives ──────────────────────────────────────────────────────

class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    HIGH = "high"


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class Flag(BaseModel):
    """A flagged issue raised by any skill in the pack."""
    severity: Severity
    code: str                                # e.g. "TARGET_SKEW", "HIGH_VIF", "HETEROSCEDASTICITY"
    message: str                             # short human-readable summary
    detail: Optional[dict] = None            # arbitrary supporting data


class Recommendation(BaseModel):
    """An actionable suggestion produced by a skill."""
    action: str                              # e.g. "log_transform", "use_robust_se", "inspect_row"
    target: Optional[str] = None             # feature name, row index, model element
    reason: str
    priority: Severity = Severity.INFO


# ─── Linear regression schemas ──────────────────────────────────────────────

class CoefficientRow(BaseModel):
    feature: str
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    standardized_coefficient: Optional[float] = None   # None for intercept


class FitStatistics(BaseModel):
    n_observations: int
    n_features: int                          # excludes intercept
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    aic: float
    bic: float
    log_likelihood: float
    residual_std_error: float
    df_residuals: int


class InterpretationFact(BaseModel):
    """One coefficient interpretation, ready for Claude to verbalise.
    
    Distinguishing this from prose is the central design choice: this skill
    produces facts; Claude produces prose. Same facts, different prose for
    different audiences.
    """
    feature: str
    coefficient: float
    interpretation_type: Literal[
        "linear_linear",
        "log_log_elasticity",
        "log_linear_semi_elasticity",
        "linear_log",
        "binary_dummy",
        "categorical_dummy",
        "polynomial_term",
        "interaction_term",
    ]
    fact: str                                # canonical claim, e.g. "A 1% increase in sqft is associated with a 0.43% increase in price, holding other features constant"
    confidence: Literal["high", "medium", "low"]   # based on p-value + CI width
    caveats: list[str] = Field(default_factory=list)   # e.g. ["ceteris paribus", "not causal", "linear-on-log scale"]


class FitQuality(BaseModel):
    r_squared: float
    adj_r_squared: float
    interpretation: Literal["weak", "moderate", "strong", "very_strong"]


class LinearRegressionReport(BaseModel):
    """The complete structured output of the linear-regression skill."""
    fit_statistics: FitStatistics
    coefficients: list[CoefficientRow]
    interpretations: list[InterpretationFact]
    fit_quality: FitQuality
    headline: str                            # one-sentence summary Claude leads with
    target_transform: Optional[str] = None   # e.g. "log"
    feature_transforms: dict[str, str] = Field(default_factory=dict)
    robust_se_used: Optional[str] = None     # None | "HC0" | "HC1" | "HC2" | "HC3"
    flags: list[Flag] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    report_html_path: Optional[str] = None


# ─── Diagnostics schemas ────────────────────────────────────────────────────

class AssumptionCheck(BaseModel):
    """One assumption test result."""
    name: Literal[
        "linearity",
        "homoscedasticity",
        "normality_of_residuals",
        "independence",
        "no_multicollinearity",
    ]
    status: Status
    test_name: str                           # e.g. "Breusch-Pagan", "Durbin-Watson"
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    evidence: str                            # human-readable summary of the test result
    detail: Optional[dict] = None


class InfluencePoint(BaseModel):
    row_index: int
    leverage: float
    cooks_distance: float
    studentized_residual: float
    dffits: Optional[float] = None


class InfluenceReport(BaseModel):
    high_leverage: list[InfluencePoint]      # leverage > 2*(k+1)/n
    cooks_d_outliers: list[InfluencePoint]   # Cook's D > 4/n
    summary: str                             # human-readable summary
    max_cooks_d: float
    max_leverage: float


class BiasVarianceReport(BaseModel):
    train_r_squared: float
    test_r_squared: float
    cv_r_squared_mean: float
    cv_r_squared_std: float
    gap: float                               # train_r2 - test_r2
    verdict: Literal["high_bias", "good_fit", "high_variance", "inconsistent"]
    evidence: str
    learning_curve: Optional[dict] = None    # {"sizes": [...], "train_scores": [...], "test_scores": [...]}


class DiagnosticsVerdict(BaseModel):
    overall: Literal["clean", "usable_with_caveats", "problematic", "unreliable"]
    top_issues: list[str]                    # ordered codes from flags, most actionable first
    headline: str                            # one-sentence verdict for Claude to lead with


class DiagnosticsReport(BaseModel):
    """The complete structured output of the diagnostics skill."""
    assumptions: list[AssumptionCheck]
    influence: InfluenceReport
    bias_variance: BiasVarianceReport
    verdict: DiagnosticsVerdict
    flags: list[Flag] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    report_html_path: Optional[str] = None


# ─── Pre-analysis schemas (for phase 2; lock down now) ──────────────────────

class TargetAudit(BaseModel):
    name: str
    type: Literal["continuous", "binary", "count", "categorical"]
    n_missing: int
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outlier_count: Optional[int] = None
    recommendations: list[str] = Field(default_factory=list)


class FeatureAudit(BaseModel):
    name: str
    type: Literal["continuous", "binary", "categorical", "ordinal"]
    n_missing: int
    missing_pct: float
    n_unique: int
    flags: list[str] = Field(default_factory=list)


class PreAnalysisReport(BaseModel):
    n_samples: int
    target: TargetAudit
    features: list[FeatureAudit]
    multicollinearity: dict                  # {"max_vif": float, "flagged": [str], "matrix": [[...]]}
    suspected_nonlinearity: list[str]
    flags: list[Flag] = Field(default_factory=list)
    modeling_recommendations: dict           # {"transform_target": str|None, "consider_polynomial": [str], ...}
    report_html_path: Optional[str] = None
```

### 5.3 `plotting.py` — theme + chart helpers

Provide at minimum the following functions. Every plot in every skill uses these.

```python
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
    """Returns a Plotly figure with the pack theme applied (font, colours,
    gridlines, margins, hover label styling). Used as the starting point
    for every chart."""


def to_inline_html(fig: go.Figure, div_id: str | None = None) -> str:
    """Render a figure to inline HTML — Plotly JS inlined, no CDN. Mode bar
    hidden, responsive."""


def scatter(x, y, *, x_label: str = "", y_label: str = "",
            hover_text=None, title: str | None = None) -> go.Figure: ...


def line(x, y, *, x_label: str = "", y_label: str = "",
         title: str | None = None, color: str = ACCENT) -> go.Figure: ...


def histogram(x, *, x_label: str = "", title: str | None = None,
              bins: int = 30) -> go.Figure: ...


def coef_forest(features: list[str], coefs: list[float],
                lower: list[float], upper: list[float],
                *, title: str | None = None) -> go.Figure:
    """Coefficient forest plot. Markers at coefs, horizontal error bars
    from lower to upper, vertical dotted line at x=0. Y-axis reversed
    so first feature appears at top."""


def qq_plot(residuals, *, title: str | None = None) -> go.Figure:
    """QQ plot of residuals vs theoretical normal quantiles."""


def residuals_vs_fitted(fitted, residuals, *, title: str | None = None) -> go.Figure:
    """Scatter of residuals vs fitted with a LOWESS smoother overlay."""


def scale_location(fitted, residuals, *, title: str | None = None) -> go.Figure:
    """sqrt(|standardized residuals|) vs fitted, with LOWESS overlay."""


def leverage_plot(leverage, studentized_residuals, cooks_d=None,
                  *, title: str | None = None) -> go.Figure:
    """Leverage vs studentized residuals; if cooks_d supplied, size
    markers by Cook's D."""
```

All chart helpers return a `go.Figure`. Skills call `to_inline_html(fig)` to embed in reports.

### 5.4 `validators.py`

```python
class ValidationResult(BaseModel):
    ok: bool
    message: str = ""
    issues: list[str] = Field(default_factory=list)


def validate_regression_inputs(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    require_numeric_target: bool = True,
) -> ValidationResult:
    """Verify: target & features in df, target numeric (if required),
    no entirely-null columns, enough rows after dropping NA. Returns
    ValidationResult with .ok set, message and issues populated."""


def coerce_features(
    df: pd.DataFrame,
    features: list[str],
    *,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode categorical / object dtype columns. Return the
    encoded design matrix and a dict mapping original feature -> list
    of dummy column names (so coefficient interpretation can map back)."""
```

### 5.5 `reports.py`

```python
def render_html_report(
    *,
    title: str,
    skill_name: str,
    dataset_name: str,
    n_obs: int,
    timestamp: str,
    verdict: dict | None,                # {"tone": "ok|warn|fail", "label": str, "headline": str}
    body_html: str,
) -> str:
    """Loads templates/base_report.html.j2, injects inlined CSS from
    style.css, renders. Returns the complete HTML document as a string."""


def section(title: str, body: str) -> str:
    """Returns: <div class="section"><h2>...</h2>{body}</div>"""


def stat_grid(stats: list[dict]) -> str:
    """stats is a list of {"label": str, "value": str}. Returns the
    .stat-grid HTML block."""


def coefficient_table_html(coefs: list[CoefficientRow]) -> str:
    """Renders coefficient table with significance stars (*** p<0.001,
    ** p<0.01, * p<0.05, . p<0.1). All numbers right-aligned, tabular
    nums."""


def flag_list_html(flags: list[Flag]) -> str:
    """Renders flags as styled .flag.info|.warn|.high blocks."""
```

### 5.6 `style.css` (full)

Use these exact rules — they're consumed by every report.

```css
:root {
    --accent: #0F766E;
    --accent-light: #5EEAD4;
    --ink: #1F2937;
    --ink-soft: #6B7280;
    --grid: #E5E7EB;
    --bg: #FFFFFF;
    --bg-soft: #F9FAFB;
    --ok: #059669;
    --ok-soft: #D1FAE5;
    --warn: #D97706;
    --warn-soft: #FEF3C7;
    --fail: #DC2626;
    --fail-soft: #FEE2E2;
    --font: 'Inter', 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
    --max-width: 1100px;
}

* { box-sizing: border-box; }

body {
    font-family: var(--font);
    color: var(--ink);
    background: var(--bg-soft);
    margin: 0;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
}

.report {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 48px 32px;
    background: var(--bg);
    min-height: 100vh;
}

/* Header */
.report-header { border-bottom: 1px solid var(--grid); padding-bottom: 24px; margin-bottom: 32px; }
.report-header .skill-name { font-size: 11px; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 500; }
.report-header h1 { font-size: 28px; font-weight: 600; margin: 6px 0 8px; color: var(--ink); letter-spacing: -0.01em; }
.report-header .meta { font-size: 13px; color: var(--ink-soft); }

/* Verdict card */
.verdict { background: var(--bg-soft); border-left: 3px solid var(--accent); padding: 20px 24px; margin-bottom: 36px; border-radius: 0 6px 6px 0; }
.verdict.warn { border-left-color: var(--warn); }
.verdict.fail { border-left-color: var(--fail); }
.verdict .label { font-size: 10px; font-weight: 600; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.verdict .headline { font-size: 17px; font-weight: 500; color: var(--ink); line-height: 1.45; }

/* Section */
.section { margin-bottom: 44px; }
.section h2 { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-soft); border-bottom: 1px solid var(--grid); padding-bottom: 8px; margin: 0 0 18px; }
.section p { font-size: 14px; color: var(--ink); }

/* Tables */
table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.data-table th { text-align: left; font-weight: 500; color: var(--ink-soft); border-bottom: 1px solid var(--grid); padding: 9px 12px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
table.data-table td { padding: 9px 12px; border-bottom: 1px solid var(--grid); color: var(--ink); }
table.data-table tr:last-child td { border-bottom: none; }
table.data-table .num { text-align: right; }
table.data-table .sig-stars { color: var(--accent); font-weight: 600; }

/* Stat grid */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 28px; }
.stat { background: var(--bg-soft); border-radius: 6px; padding: 14px 16px; }
.stat .label { font-size: 10px; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; font-weight: 600; }
.stat .value { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--ink); letter-spacing: -0.01em; }
.stat .sub { font-size: 12px; color: var(--ink-soft); margin-top: 2px; }

/* Flags */
.flag { display: flex; align-items: flex-start; gap: 12px; padding: 12px 16px; border-radius: 6px; margin-bottom: 8px; font-size: 13px; line-height: 1.45; }
.flag.info { background: var(--bg-soft); }
.flag.warn { background: var(--warn-soft); }
.flag.high { background: var(--fail-soft); }
.flag .badge { flex-shrink: 0; font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 3px; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 1px; }
.flag.info .badge { background: var(--ink-soft); color: white; }
.flag.warn .badge { background: var(--warn); color: white; }
.flag.high .badge { background: var(--fail); color: white; }

/* Status pills (used in assumption tables) */
.pill { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
.pill.ok { background: var(--ok-soft); color: var(--ok); }
.pill.warn { background: var(--warn-soft); color: var(--warn); }
.pill.fail { background: var(--fail-soft); color: var(--fail); }

/* Plot container */
.plot-container { margin: 16px 0 24px; }
.plot-container.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

/* Footer */
.report-footer { border-top: 1px solid var(--grid); padding-top: 18px; margin-top: 56px; font-size: 12px; color: var(--ink-soft); text-align: center; }
.report-footer a { color: var(--accent); text-decoration: none; }
```

### 5.7 `templates/base_report.html.j2`

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>{{ css }}</style>
</head>
<body>
<main class="report">
    <header class="report-header">
        <div class="skill-name">{{ skill_name }}</div>
        <h1>{{ title }}</h1>
        <div class="meta">
            {%- if dataset_name %}{{ dataset_name }} · {% endif -%}
            {{ n_obs }} observations · {{ timestamp }}
        </div>
    </header>

    {%- if verdict %}
    <div class="verdict {{ verdict.tone }}">
        <div class="label">{{ verdict.label }}</div>
        <div class="headline">{{ verdict.headline }}</div>
    </div>
    {%- endif %}

    {{ body | safe }}

    <footer class="report-footer">
        Generated by <a href="#">regression-pack</a> · {{ skill_name }}
    </footer>
</main>
</body>
</html>
```

---

## 6. Skill: `linear-regression`

### 6.1 `SKILL.md` (exact frontmatter and body)

```markdown
---
name: linear-regression
description: Fit a linear regression (OLS) on tabular data and produce a rigorous report — coefficient table with confidence intervals, standardized betas, plain-English interpretation of each coefficient (transform-aware), fit statistics, and a standalone HTML report. Use this skill whenever the user wants to model a continuous outcome from one or more predictors, asks to "fit a regression" or "run OLS" or "model Y from X", or hands over tabular data with a continuous target. Outputs are both a structured JSON (LinearRegressionReport) and a self-contained HTML deliverable. Pair this skill with the diagnostics skill to check assumptions and identify influential observations.
---

# linear-regression

Fits an OLS linear regression with `statsmodels`, produces a `LinearRegressionReport` JSON validating against the pack schema, and renders a standalone HTML report.

## When this skill fires

- User wants to fit a linear regression on tabular data
- User has identified a continuous target and one or more predictors
- User asks for OLS, multiple regression, or "model Y from X"
- A previous skill (e.g. pre-analysis) recommended linear regression

## Inputs

- A CSV or Parquet file with the data
- The target column name (must be numeric)
- A list of predictor column names (numeric or categorical — categoricals are one-hot encoded with the first level dropped)

Optional:
- `--log-target` to fit on log(target) — useful for skewed positive targets
- `--robust-se {HC0|HC1|HC2|HC3}` for heteroscedasticity-robust standard errors
- `--standardize` to also report standardized β coefficients

## How to invoke

```bash
uv run python linear-regression/scripts/fit.py \
    --data path/to/data.csv \
    --target price \
    --features sqft,bedrooms,bathrooms,neighborhood \
    --output results/
```

Outputs `results/report.json` (LinearRegressionReport) and `results/report.html`.

## Verbalising the output

The `interpretations` field contains a list of InterpretationFact objects. Each has:
- `fact` — the canonical claim
- `confidence` — high/medium/low based on p-value and CI width
- `caveats` — list of qualifiers ("ceteris paribus", "not causal", scale notes)

Read the `headline` field aloud first, then walk through the top 2-3 interpretation facts ordered by absolute coefficient size, attaching caveats appropriate to the user's apparent sophistication. Do not invent statistics; only verbalise what's in the report.

## Diagnostics

This skill performs the *fit* — it does not run the full assumption battery. After fitting, recommend the user run the `diagnostics` skill on the fitted model:

```bash
uv run python diagnostics/scripts/diagnose.py --fit-report results/report.json --data path/to/data.csv --output results/
```

## Reference files

- `references/interpretation.md` — when each interpretation_type applies and how to phrase it
- `references/robust_se.md` — when to use each HC variant
```

### 6.2 `scripts/fit.py` (CLI + behaviour)

```
Usage:
    uv run python linear-regression/scripts/fit.py \
        --data <path>                          # CSV or .parquet
        --target <column>                      # numeric target
        --features <a,b,c>                     # comma-separated predictor columns
        --output <dir>                         # output directory (created if missing)
        [--log-target]                         # apply np.log to target
        [--robust-se HC0|HC1|HC2|HC3]
        [--standardize]                        # populate standardized_coefficient
        [--dataset-name <str>]                 # for report header
```

Behaviour:

1. Load data via `pd.read_csv` or `pd.read_parquet` (based on suffix).
2. Validate via `validators.validate_regression_inputs`. On failure: print error to stderr, exit 1.
3. One-hot encode categoricals via `validators.coerce_features` (drop_first=True).
4. Drop rows with any NA in target or features. If <30 rows remain, exit 1 with error.
5. If `--log-target`: assert all target values > 0; apply `np.log`; set `target_transform = "log"`.
6. Fit `sm.OLS(y, sm.add_constant(X)).fit(cov_type=...)` where cov_type is the robust-se choice or `"nonrobust"`.
7. Build `CoefficientRow` list via `coefficients.build_coefficient_table`. If `--standardize`, also compute standardized betas (β * σ_x / σ_y).
8. Compute `FitStatistics` from the fitted model.
9. Build `FitQuality` via the adjusted R² thresholds: ≥0.7 very_strong, ≥0.5 strong, ≥0.3 moderate, else weak.
10. Generate `InterpretationFact` list via `interpret.generate_interpretations`, passing the target_transform.
11. Construct one-sentence headline: `"The model explains {adj_r2*100:.1f}% of variance in {target} ({n_sig} of {n_features} predictors significant at p < 0.05)."`
12. Assemble `LinearRegressionReport`, validate, write to `output/report.json` (indent=2).
13. Render HTML via `render.render_report(...)`, write to `output/report.html`.
14. Print to stdout:
    ```
    ✓ Linear regression fit complete
      n = {n_obs}, k = {n_features}
      R² = {r2:.4f}  (adj: {adj_r2:.4f})
      {headline}
      JSON: {output}/report.json
      HTML: {output}/report.html
    ```
15. Exit 0.

### 6.3 `scripts/coefficients.py`

```python
def build_coefficient_table(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    *,
    standardize: bool = False,
    y: Optional[pd.Series] = None,            # required if standardize=True
) -> list[CoefficientRow]:
    """Build a CoefficientRow per parameter. The intercept ('const') appears
    first. p-values from model.pvalues, CIs from model.conf_int() (95% default).
    If standardize, set standardized_coefficient = coef * X[f].std() / y.std()."""
```

### 6.4 `scripts/interpret.py` — the LLM-fact layer

The most important file in this skill. Generates one `InterpretationFact` per non-intercept coefficient. Behaviour by `interpretation_type`:

| Target transform | Feature transform | interpretation_type | Fact template |
|---|---|---|---|
| none | none | `linear_linear` | "A one-unit increase in {f} is associated with a {β:.3g} {direction} in {target}, holding other features constant." |
| log | log | `log_log_elasticity` | "A 1% increase in {f} is associated with a {β*100:.2f}% {direction} in {target}, holding other features constant." |
| log | none | `log_linear_semi_elasticity` | "A one-unit increase in {f} is associated with a {(exp(β)-1)*100:.2f}% {direction} in {target}, holding other features constant." |
| none | log | `linear_log` | "A 1% increase in {f} is associated with a {β/100:.4g} {direction} in {target}, holding other features constant." |
| any | dummy from binary | `binary_dummy` | "{f}={level} is associated with a {β:.3g} {direction} in {target} compared to the reference category, holding other features constant." |
| any | dummy from categorical | `categorical_dummy` | (same shape as binary_dummy, references the dropped baseline level) |
| any | polynomial (e.g. f^2) | `polynomial_term` | "{f} has a nonlinear relationship with {target}; the {f}² term is {β:.3g} (p={p:.3g}), indicating {curvature}." |
| any | interaction (f1:f2) | `interaction_term` | "The effect of {f1} on {target} varies with {f2}; the interaction coefficient is {β:.3g} (p={p:.3g})." |

`direction` is "increase" if β > 0, "decrease" if β < 0 (use absolute value in the number).

`confidence` levels:
- `high` — p < 0.01 AND CI width < |β|
- `medium` — p < 0.05
- `low` — otherwise

`caveats` always include `"holding other features constant"` and `"association, not causation"`. Additional caveats by interpretation_type — see `references/interpretation.md`.

```python
def generate_interpretations(
    coefs: list[CoefficientRow],
    *,
    target_transform: Optional[str] = None,
    feature_transforms: dict[str, str] = None,
    dummy_origin: dict[str, str] = None,       # {"neighborhood_north": "neighborhood"}
) -> list[InterpretationFact]:
    """Generate one fact per non-intercept coefficient."""
```

### 6.5 `scripts/render.py`

Composes the HTML report. Structure:

1. Verdict card: headline, tone = `"ok"` if fit_quality is strong/very_strong else `"warn"`.
2. Section "Fit summary" — `stat_grid` with: n, k, R², adj R², F-stat, AIC, BIC, RSE. Below that, a sentence with the headline.
3. Section "Coefficients" — `coefficient_table_html` (intercept first, others sorted by |β| descending), followed by a `coef_forest` plot (intercept excluded).
4. Section "Interpretation" — for each InterpretationFact: the `fact` sentence in its own block, with confidence pill and caveats listed below in small grey text.
5. Section "Diagnostics" — placeholder note: "For full assumption checks, influence analysis, and bias/variance assessment, run the `diagnostics` skill on this fit. See README."

Embed Plotly via `plotting.to_inline_html`. Use `reports.render_html_report` for the outer scaffold.

### 6.6 References

#### `references/interpretation.md`

Content outline (Claude Code: write the full markdown):

- One section per `interpretation_type`
- For each: when it applies, the canonical sentence form, what "confidence" means in that context, what caveats to attach
- Worked examples: log-log elasticity on price~sqft, semi-elasticity on log-wage~education
- Section on dummy / categorical coefficients: reference category interpretation
- Section on standardized coefficients: when to report them, what they mean

#### `references/robust_se.md`

Content outline:

- What heteroscedasticity is and why robust SE help
- HC0 — original White; biased small samples
- HC1 — Stata default; small-sample correction
- HC2 — leverages-adjusted; preferred when high-leverage points exist
- HC3 — Davidson-MacKinnon; recommended for n < 250
- Decision tree: when to pick each
- Note: HC variants change SEs only; coefficients are unchanged

---

## 7. Skill: `diagnostics`

### 7.1 `SKILL.md`

```markdown
---
name: diagnostics
description: Run the full diagnostic battery on a fitted regression model — assumption tests (linearity, homoscedasticity, normality of residuals, independence, multicollinearity), influence analysis (leverage, Cook's distance, DFFITS), and bias/variance assessment via cross-validation. Produces a structured DiagnosticsReport with status flags and ranked remediation recommendations, plus a standalone HTML report. Use this skill whenever a regression has been fitted and the user wants to know if they can trust it, asks "are the assumptions met?", "is this model overfit?", "is this overfitting or underfitting?", or wants to identify influential outliers. Pair with the linear-regression or regularized-regression skills.
---

# diagnostics

Runs the full diagnostic battery on a fitted regression model. Input is the JSON output of a fit skill (e.g. LinearRegressionReport) plus the original data.

## When this skill fires

- User has fitted a regression (via linear-regression or similar) and asks about model quality
- User asks "is this overfitting?", "are the assumptions met?", "is there multicollinearity?"
- User asks to identify influential observations, outliers, or high-leverage points
- The fit skill recommends running diagnostics

## Inputs

- `--fit-report <path>` — JSON of a LinearRegressionReport (or compatible)
- `--data <path>` — the original data file (must contain the same target & features used in the fit)
- `--output <dir>` — output directory

Optional:
- `--cv-folds <int>` — folds for cross-validation (default 5)
- `--learning-curve` — include learning curve data in the report (slower)
- `--test-split <float>` — held-out fraction for train/test gap (default 0.2)

## How to invoke

```bash
uv run python diagnostics/scripts/diagnose.py \
    --fit-report results/report.json \
    --data data/houses.csv \
    --output results/
```

Outputs `results/diagnostics.json` (DiagnosticsReport) and `results/diagnostics.html`.

## Verbalising the output

Read the `verdict.headline` first. Then surface `verdict.top_issues` in order — each maps to one or more flags. For each flag, the report's recommendations field has the actionable next step. Do not list every assumption check unless asked; lead with what's broken or marginal.

## Reference files

- `references/plots_guide.md` — what each diagnostic plot should look like
- `references/remediation.md` — what to do when each assumption fails
```

### 7.2 `scripts/diagnose.py` (CLI + behaviour)

Orchestrates the other modules. Behaviour:

1. Load `--fit-report` JSON, validate against `LinearRegressionReport`.
2. Load `--data`, reconstruct X and y using the same target/features/transforms from the fit report. Apply same one-hot encoding.
3. Refit the model locally using `statsmodels` (we need the fitted object to compute residuals, leverage, etc.). The refit must match the original — fail loudly if R² differs by > 1e-6.
4. Run assumption checks (`assumptions.run_all`), influence analysis (`influence.run`), bias/variance via CV (`bias_variance.run`).
5. Build flags from each module's findings. Convert assumption FAIL to severity=HIGH, WARN to severity=WARN.
6. Run `triage.build_verdict` to produce `DiagnosticsVerdict`.
7. Generate recommendations from `references/remediation.md` patterns: for each FAIL flag, emit one `Recommendation` with the appropriate action code.
8. Assemble `DiagnosticsReport`, write JSON + HTML.

### 7.3 Module specs

#### `scripts/assumptions.py`

```python
def linearity(model, X, y) -> AssumptionCheck:
    """Ramsey RESET test. p < 0.05 → status=WARN ('missed nonlinearity'),
    p < 0.01 → status=FAIL."""

def homoscedasticity(model) -> AssumptionCheck:
    """Breusch-Pagan test (het_breuschpagan). p < 0.05 → WARN, p < 0.01 → FAIL."""

def normality_of_residuals(model) -> AssumptionCheck:
    """Shapiro-Wilk if n ≤ 5000, else Anderson-Darling. p < 0.05 → WARN.
    Always FAIL only if extreme deviation visible in QQ — defer to plot.
    Include skewness and kurtosis in detail."""

def independence(model) -> AssumptionCheck:
    """Durbin-Watson. < 1.5 or > 2.5 → WARN. < 1.0 or > 3.0 → FAIL.
    Note: only meaningful for time-ordered data — include caveat."""

def no_multicollinearity(X) -> AssumptionCheck:
    """VIF per feature (excluding intercept). max VIF > 5 → WARN, > 10 → FAIL.
    detail includes per-feature VIFs."""

def run_all(model, X, y) -> list[AssumptionCheck]: ...
```

#### `scripts/influence.py`

```python
def run(model, X) -> InfluenceReport:
    """Compute leverage (hat diagonal), Cook's distance, studentized
    residuals, DFFITS. Flag points where:
    - leverage > 2*(k+1)/n
    - Cook's D > 4/n
    - |studentized residual| > 3
    Return InfluenceReport with per-point details for flagged rows only."""
```

#### `scripts/bias_variance.py`

```python
def run(
    X, y,
    *,
    test_split: float = 0.2,
    cv_folds: int = 5,
    include_learning_curve: bool = False,
) -> BiasVarianceReport:
    """1. Train/test split, refit on train, score on test.
       2. K-fold CV on full data, capture mean and std of R².
       3. Verdict logic:
            gap > 0.15 AND test_r2 < 0.5  → 'high_variance'
            train_r2 < 0.3                → 'high_bias'
            gap < 0.05 AND cv_std < 0.05  → 'good_fit'
            else                          → 'inconsistent'
       4. Evidence: a one-sentence explanation of the verdict."""
```

#### `scripts/residuals.py`

```python
def residual_data(model) -> dict:
    """Return: {'fitted': [...], 'residuals': [...], 'std_residuals': [...],
    'leverage': [...], 'cooks_d': [...]}. Used by render.py for plots."""
```

#### `scripts/nonlinearity.py`

```python
def partial_residual_plots(model, X) -> dict[str, dict]:
    """One per continuous feature. Returns {feature: {'x': [...], 'partial_resid': [...]}}.
    Used by render.py — feature has potential nonlinearity if LOWESS deviates
    visibly from linear; flagged in detail."""
```

#### `scripts/triage.py`

```python
def build_verdict(
    assumptions: list[AssumptionCheck],
    influence: InfluenceReport,
    bias_variance: BiasVarianceReport,
) -> DiagnosticsVerdict:
    """Logic:
    - count of FAIL assumptions, count of WARN
    - bias_variance verdict
    - whether any Cook's D outlier exceeds 4/n by >5x
    
    overall mapping:
        0 FAIL, 0 WARN, good_fit                → 'clean'
        ≤2 WARN, no FAIL, good_fit              → 'usable_with_caveats'
        ≥1 FAIL, OR high_variance               → 'problematic'
        ≥3 FAIL, OR high_bias                   → 'unreliable'
    
    top_issues: ordered list of flag codes, FAILs first then WARNs.
    headline: one sentence."""
```

### 7.4 `scripts/render.py`

Report structure:

1. **Verdict card** — verdict.headline; tone by overall (clean→ok, usable_with_caveats→ok, problematic→warn, unreliable→fail)
2. **Section "Assumption checks"** — table with columns: Assumption | Test | Statistic | p-value | Status (pill). One row per `AssumptionCheck`.
3. **Section "Diagnostic plots"** — grid-2 layout: residuals vs fitted, QQ, scale-location, residuals vs leverage (Cook's D as marker size).
4. **Section "Influential observations"** — if any flagged, table with row index, leverage, Cook's D, studentized residual. If none flagged, single line: "No influential observations detected."
5. **Section "Bias / variance"** — stat_grid: Train R², Test R², CV R² mean ± std, Gap. Below: verdict sentence. If learning curve enabled, plot.
6. **Section "Recommendations"** — `flag_list_html` of all flags, then numbered list of `Recommendation` items.

### 7.5 References

#### `references/plots_guide.md`

Content outline:

- Residuals vs Fitted: what flat-LOWESS-around-zero means; funnel = heteroscedasticity; curve = missed nonlinearity
- QQ plot: light/heavy tails interpretation
- Scale-location: detecting heteroscedasticity visually
- Residuals vs Leverage: Cook's D contour interpretation, influential points

#### `references/remediation.md`

Content outline — keyed by flag code:

- `HETEROSCEDASTICITY` → use robust SE (point to robust_se.md), or log/sqrt transform target
- `MISSED_NONLINEARITY` → add polynomial term, or use partial residual plot to identify feature
- `HIGH_VIF` → drop one collinear feature, or use ridge regression
- `NON_NORMAL_RESIDUALS` → transform target (log/Box-Cox), or rely on CLT if n large
- `AUTOCORRELATION` → if time series, use OLS-with-AR-errors or switch to time-series model
- `HIGH_COOKS_D` → inspect the point; if data error, fix; if legitimate, refit with and without and compare
- `HIGH_VARIANCE` → regularise (ridge/lasso), or get more data, or simplify model
- `HIGH_BIAS` → add features, polynomial terms, or interactions

---

## 8. Examples

### 8.1 `examples/synth_data.py`

Generate three CSVs in `examples/data/`:

```python
# clean.csv — n=400
#   x1, x2 ~ N(0,1) independent
#   y = 2 + 1.5*x1 - 0.8*x2 + eps,  eps ~ N(0, 1)
#   All assumptions satisfied; should diagnose clean.

# heteroscedastic.csv — n=400
#   x1, x2 as above
#   y = 2 + 1.5*x1 - 0.8*x2 + eps,  eps ~ N(0, |x1| + 0.5)
#   Should diagnose: homoscedasticity FAIL.

# influential.csv — n=200
#   x1, x2 as above
#   y = 2 + 1.5*x1 - 0.8*x2 + eps,  eps ~ N(0, 1)
#   Then row 0: set x1=10, y=-50 (clear high-leverage + Cook's D outlier)
#   Should diagnose: 1+ influential observations flagged.
```

Set `np.random.seed(42)` so outputs are reproducible.

### 8.2 `examples/run_linear_regression.sh`

```bash
#!/usr/bin/env bash
set -e
uv run python examples/synth_data.py
uv run python linear-regression/scripts/fit.py \
    --data examples/data/clean.csv \
    --target y \
    --features x1,x2 \
    --output examples/output/clean \
    --dataset-name "Clean synthetic data"
echo "Done. Open examples/output/clean/report.html"
```

### 8.3 `examples/run_diagnostics.sh`

```bash
#!/usr/bin/env bash
set -e
# Assumes run_linear_regression.sh already ran
uv run python diagnostics/scripts/diagnose.py \
    --fit-report examples/output/clean/report.json \
    --data examples/data/clean.csv \
    --output examples/output/clean
echo "Done. Open examples/output/clean/diagnostics.html"
```

---

## 9. Quality criteria (what "done" means)

For each skill, before considering it shipped:

1. **Schemas validate.** Every JSON output round-trips through its Pydantic model without error.
2. **HTML reports render.** Open in browser, no console errors, all plots interactive.
3. **CLI is robust.** Wrong column name → graceful error to stderr, exit 1. Empty data after NA drop → graceful error. Wrong file format → graceful error.
4. **Smoke tests pass.** All three synthetic datasets produce expected outputs:
   - `clean.csv`: assumptions all OK, bias_variance verdict `good_fit`, no influence flags
   - `heteroscedastic.csv`: homoscedasticity FAIL, recommendation to use robust SE
   - `influential.csv`: at least one influence flag, Cook's D for row 0 > 4/n by >5x
5. **Visual coherence.** Both reports look like they're from the same product. Same font, same colours, same section structure.
6. **No prose from scripts.** Scripts return facts and statuses. Any human-readable string in the output is either (a) a deterministic template fill (interpretation facts), or (b) a short evidence sentence from a test result (e.g. "Residual spread increases with fitted values"). Long-form narration happens in Claude's chat layer, not in script outputs.

---

## 10. Build order summary (for Cursor / Claude Code)

```
[ ] Phase 1.0  Scaffold       (§§ 1-4)         → uv sync passes
[ ] Phase 1.1  Core library   (§ 5)            → import smoke test passes
[ ] Phase 1.2  linear-regression (§ 6)         → run_linear_regression.sh produces valid report
[ ] Phase 1.3  diagnostics    (§ 7)            → run_diagnostics.sh produces valid report
[ ] Phase 1.4  Smoke tests + evals (§§ 8-9)    → all three synthetic datasets diagnose correctly
[ ] Phase 1.5  README polish + screenshots     → shippable repo
```

After phase 1.5, the repo is presentable as a portfolio artifact. Phase 2 (pre-analysis, logistic-regression, regularized-regression, model-comparison) gets its own spec.

---

## 11. Things to verify with the user before building

If anything below is unclear when you start, ask before assuming:

- Target Python version (locked to 3.13+; bump if they're on a newer release and want to use 3.13-only or later features)
- Whether they want the `uv.lock` committed (assumed yes — reproducibility)
- Whether to include `pre-commit` hooks (not in this spec; can add later)
- Any house style for docstrings (assumed Google-style)
- Whether they want CLI scripts converted to a Typer/Click app at any point (not in this spec — argparse is fine)

---

End of build spec.
