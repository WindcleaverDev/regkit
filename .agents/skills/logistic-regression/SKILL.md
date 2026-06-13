# logistic-regression

Fits a binary logistic regression with `statsmodels`, produces a `LogisticRegressionReport` validating against the pack schema, renders a standalone HTML report.

## When to use

- Target has exactly two distinct values (0/1, yes/no, true/false, two category names)
- User asks for logistic regression, "model a binary outcome", "predict yes/no"
- pre-analysis identified the target as `binary`
- User wants odds ratios or predicted probabilities, not just coefficient signs

## Quick start

```bash
python .agents/skills/logistic-regression/scripts/fit.py \
    --data path/to/data.csv \
    --target churned \
    --positive-class yes \
    --features tenure,monthly_charges,contract_type \
    --output results/
```

Outputs `results/report.json` (LogisticRegressionReport) and `results/report.html`.

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | CSV or Parquet path |
| `--target` | required | Binary target column (exactly 2 unique values) |
| `--positive-class` | auto | Which value to treat as 1 (default: `1` if 0/1 numeric, else alphabetically last) |
| `--features` | required | Comma-separated columns, or `all` |
| `--output` | required | Output directory |
| `--robust-se` | off | `HC0`–`HC3` robust standard errors |
| `--threshold` | `0.5` | Classification decision threshold |
| `--dataset-name` | `""` | Label shown in report header |

## Outputs

```
out/
├── report.json   # LogisticRegressionReport (Pydantic schema)
└── report.html   # Self-contained HTML with Plotly charts
```

### Report sections

1. **Fit summary** — AUC, Brier score, McFadden pseudo-R², AIC/BIC, positive rate
2. **Marginal effects** — AME forest plot + table (the primary interpretation surface)
3. **Coefficients** — log-odds table + odds ratio forest plot (log scale)
4. **Classification performance** — confusion matrix, accuracy, balanced accuracy, F1
5. **ROC curve** — AUC annotated
6. **Calibration** — reliability diagram with Brier score
7. **Interpretation** — AME-based plain-English facts per feature
8. **Flags & recommendations**

## Interpreting the output

Lead with **marginal effects** (AME), not raw log-odds. AME tells you the average change in predicted probability per unit increase in each feature — the quantity humans find most interpretable.

- AUC 0.5 = random, 0.7 = OK, 0.8 = good, 0.9+ = strong
- Brier score 0.25 = baseline for balanced classes (lower is better)
- `CLASS_IMBALANCE` flag: accuracy is misleading; prefer balanced accuracy and AUC

## Chaining

After fit, run `diagnostics` for assumption checks adapted to logistic models (log-odds linearity, leverage analysis). Or run `model-comparison` alongside an OLS fit on the same binary target.

## References

- `references/odds_ratio_interpretation.md`
- `references/marginal_effects.md`
- `references/class_imbalance.md`
- `references/threshold_choice.md`
