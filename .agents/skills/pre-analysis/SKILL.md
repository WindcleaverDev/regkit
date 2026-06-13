---
name: pre-analysis
description: Audit tabular data before fitting any regression — target distribution and skewness, feature types and missingness, multicollinearity (pairwise correlation + VIF), suspected nonlinearity, univariate outliers, and concrete modeling recommendations (which transforms to apply, which features to drop or combine, which estimator family to use). Produces a structured PreAnalysisReport and a standalone HTML report. Use this skill whenever the user is about to fit a model and hasn't done EDA yet, asks "what should I do with this data?", "should I transform anything?", "is this data ready to model?", or hands over a CSV with a target in mind but no clear modeling plan. Run this BEFORE linear-regression or logistic-regression to avoid refit cycles.
---

# pre-analysis

Audits tabular data and recommends modeling choices before any fit. Produces a `PreAnalysisReport` JSON validating against the pack schema and a standalone HTML report.

## When this skill fires

- User has tabular data + identified target, but hasn't fitted yet
- User asks for EDA, data audit, "what should I do with this data?"
- User asks whether to transform variables before modeling
- Before any fit skill — especially before linear-regression if heteroscedasticity or skew is plausible

## Inputs

- `--data <path>` — CSV or Parquet
- `--target <column>` — target variable (continuous, binary, or count)
- `--features <a,b,c>` — comma-separated predictor columns, or `all` to use every non-target column
- `--output <dir>` — output directory

## How to invoke

```bash
uv run python .agents/skills/pre-analysis/scripts/audit.py \
    --data path/to/data.csv \
    --target price \
    --features all \
    --output results/
```

Outputs `results/pre_analysis.json` (PreAnalysisReport) and `results/pre_analysis.html`.

## Verbalising the output

Read the warning block first (`flags` field, severity HIGH first). Then walk through `modeling_recommendations` — these are the concrete actions the user should take before fitting. Do not skip the recommendations to dive into raw distributions; the recommendations are the deliverable.

If `target.recommendations` includes `log_transform`, suggest passing `--log-target` to the subsequent fit skill. If `multicollinearity.flagged` is non-empty, suggest either dropping features or using `regularized-regression` with ridge.

## Chaining

The typical next step is one of:
- `linear-regression` (continuous target, OK assumptions) — pass any recommended transforms
- `logistic-regression` (binary target)
- `regularized-regression` (multicollinearity flagged)

## Reference files

- `references/transformations.md` — when to log/sqrt/Box-Cox/Yeo-Johnson
- `references/encoding.md` — categorical encoding choices
- `references/vif_remediation.md` — what to do about multicollinearity
