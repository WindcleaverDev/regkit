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
