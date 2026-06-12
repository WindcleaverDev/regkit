# regression-pack

Statistically rigorous regression skills for Claude — deterministic statistics,
structured JSON outputs, and standalone HTML reports.

Each skill computes its numbers in Python (`statsmodels`, `scikit-learn`,
`scipy`), validates its output against a shared Pydantic schema, and renders a
self-contained HTML deliverable (Plotly inlined, no CDN). The LLM's role is
constrained to the judgment layer: choosing transforms, narrating coefficients
for the audience, deciding which diagnostic issue to surface first. It never
does arithmetic and never reports a p-value it didn't compute.

## Skills

| Skill | What it does | Output |
|---|---|---|
| [`linear-regression`](linear-regression/SKILL.md) | OLS fit: coefficient table with CIs, transform-aware plain-English interpretation facts, fit statistics | `report.json` ([`LinearRegressionReport`](regression_pack_core/schemas.py)) + `report.html` |
| [`diagnostics`](diagnostics/SKILL.md) | Assumption battery (linearity, homoscedasticity, normality, independence, multicollinearity), influence analysis (leverage, Cook's D, DFFITS), bias/variance via CV, triaged verdict | `diagnostics.json` ([`DiagnosticsReport`](regression_pack_core/schemas.py)) + `diagnostics.html` |

Planned (phase 2): `pre-analysis`, `logistic-regression`, `regularized-regression`, `model-comparison`.

## Quick start

```bash
uv sync

# Generate synthetic data and fit a regression
bash examples/run_linear_regression.sh

# Run the full diagnostic battery on that fit
bash examples/run_diagnostics.sh

open examples/output/clean/report.html
open examples/output/clean/diagnostics.html
```

Typical fit output:

```
✓ Linear regression fit complete
  n = 400, k = 2
  R² = 0.7288  (adj: 0.7275)
  The model explains 72.7% of variance in y (2 of 2 predictors significant at p < 0.05).
  JSON: examples/output/clean/report.json
  HTML: examples/output/clean/report.html
```

And the diagnostics verdict on the same data:

```
✓ Diagnostics complete
  verdict: clean
  assumptions: 5 ok, 0 warn, 0 fail
  All assumption checks pass and the model generalises well — results can be trusted.
```

## Using on your own data

```bash
uv run python linear-regression/scripts/fit.py \
    --data path/to/data.csv \
    --target price \
    --features sqft,bedrooms,bathrooms,neighborhood \
    --output results/ \
    [--log-target] [--robust-se HC3] [--standardize]

uv run python diagnostics/scripts/diagnose.py \
    --fit-report results/report.json \
    --data path/to/data.csv \
    --output results/
```

Categorical features are one-hot encoded automatically (first level dropped);
each dummy coefficient gets a reference-category interpretation fact.

## Repository layout

```
regression_pack_core/   shared library: Pydantic schemas, Plotly theme,
                        validators, Jinja report assembly, CSS
linear-regression/      fit skill (SKILL.md + CLI scripts + references)
diagnostics/            diagnostics skill (SKILL.md + CLI scripts + references)
examples/               synthetic data generator + runnable examples
evals/
  correctness/          textbook problems (Anscombe's quartet)
  detection/            planted-violation smoke tests (run via pytest)
  triggering/           prompt-level skill-selection eval plan
```

## Design principles

1. **Facts, not prose.** Scripts emit `InterpretationFact` objects — canonical
   claims with confidence grades and caveats. Claude turns the same facts into
   different prose for different audiences.
2. **Schemas are the contract.** Every cross-script payload round-trips
   through a model in `regression_pack_core/schemas.py`; skills chain by JSON.
3. **Deterministic and reproducible.** Same data, same numbers. Synthetic
   datasets are seeded; the diagnostics refit verifies it matches the original
   fit to 1e-6 before trusting any residual.
4. **One visual language.** All plots go through `regression_pack_core.plotting`,
   all reports through the shared template and design tokens.

## Development

```bash
uv run pytest evals/   # correctness + detection evals
uv run ruff check .    # lint
```

The detection evals assert the pack's headline guarantees: clean data
diagnoses `clean` with zero flags, planted heteroscedasticity fails
Breusch-Pagan with a `use_robust_se` recommendation, and a planted
high-leverage outlier is flagged with Cook's D more than 5× the 4/n threshold.
The correctness evals check Anscombe's quartet: four datasets with the same
regression line whose diagnostics must come back different.
