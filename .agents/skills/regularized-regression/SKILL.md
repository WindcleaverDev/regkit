# regularized-regression

Fits Ridge, Lasso, or ElasticNet regression with cross-validated alpha selection. Produces a structured `RegularizedRegressionReport` (JSON + HTML) with a regularisation path, CV curve, feature selection summary, and OLS comparison.

## When to use

- Your pre-analysis flags **HIGH_VIF** (multicollinearity) → try Ridge or ElasticNet.
- You have many features and expect a **sparse signal** → try Lasso.
- You want **automatic feature selection** with grouping of correlated predictors → use ElasticNet.
- Your OLS standard errors are unreliable due to collinearity → regularisation stabilises estimates.

## Quick start

```bash
# Lasso (default) — auto-selects alpha via 5-fold CV
python .agents/skills/regularized-regression/scripts/fit.py \
  --data data.csv \
  --target price \
  --features sqft,bedrooms,bathrooms,location \
  --output out/lasso \
  --dataset-name "Housing data"

# Ridge — good when all features plausibly contribute
python .agents/skills/regularized-regression/scripts/fit.py \
  --data data.csv --target y --features all \
  --method ridge --output out/ridge

# ElasticNet — multicollinear groups + sparse signal
python .agents/skills/regularized-regression/scripts/fit.py \
  --data data.csv --target y --features all \
  --method elasticnet --l1-ratio 0.7 --output out/enet
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | CSV or Parquet path |
| `--target` | required | Numeric target column |
| `--features` | required | Comma-separated columns, or `all` |
| `--method` | `lasso` | `ridge`, `lasso`, or `elasticnet` |
| `--output` | required | Output directory |
| `--log-target` | off | Apply `np.log` to target before fitting |
| `--l1-ratio` | `0.5` | ElasticNet mix (0 = Ridge, 1 = Lasso) |
| `--cv-folds` | `5` | k-fold CV for alpha selection |
| `--alpha-rule` | `min` | `min` (best CV) or `1se` (parsimony) |
| `--dataset-name` | `""` | Label shown in HTML report header |

## Outputs

```
out/
├── report.json   # RegularizedRegressionReport (Pydantic schema)
└── report.html   # Self-contained HTML with Plotly charts
```

### Report sections

- **Fit summary** — R², adj. R², residual SE, selected α
- **Coefficients** — table + forest plot (non-zero features, sorted by |β|)
- **Regularisation path** — coefficient trajectories over the full α grid
- **Cross-validation** — CV R² curve with ±1 SD band and selected-α markers
- **Interpretation** — one plain-language fact per non-zero coefficient
- **Feature selection** *(Lasso/ElasticNet only)* — retained vs. zeroed features
- **OLS comparison** — side-by-side coefficient table and R² comparison
- **Flags & recommendations**

## Notes

- Features are standardised before fitting; coefficients in the report are on the **original scale**.
- SE, CI, and p-values are **approximate** (derived from OLS). They are presented for orientation only — regularised estimators are biased, so classical frequentist inference does not strictly apply.
- Ridge never zeros coefficients; `feature_selection` is `null` for ridge.

## References

- `references/regularization_choice.md` — When to pick Ridge vs. Lasso vs. ElasticNet
- `references/cv_strategies.md` — How cross-validation and alpha selection rules work
