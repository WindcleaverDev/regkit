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
