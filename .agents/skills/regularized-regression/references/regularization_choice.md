# Choosing a Regularisation Method

## Quick decision tree

```
Does multicollinearity flag the data?
├── Yes → Ridge or ElasticNet
│   └── Want feature selection too? → ElasticNet (mix of L1 + L2)
└── No
    ├── Want sparse output (automatic feature selection)? → Lasso
    └── Just want variance reduction? → Ridge
```

## Ridge (L2)

- **Penalty:** λ Σ βⱼ²
- **Effect:** Shrinks all coefficients toward zero; never zeros them.
- **When to use:** Many features, each contributing a small amount; severe multicollinearity.
- **Key property:** Closed-form solution. Coefficients of collinear features are averaged, not dropped.
- **Limitation:** Does not select features — you keep all predictors.

## Lasso (L1)

- **Penalty:** λ Σ |βⱼ|
- **Effect:** Zeros out some coefficients exactly → automatic feature selection.
- **When to use:** Sparse signal expected; interpretability is a priority; you want to know *which* features matter.
- **Key property:** Selects at most min(n, p) features. Among a group of correlated features, it tends to pick one and drop the rest.
- **Limitation:** Arbitrary between correlated features; unstable which one survives.

## ElasticNet (L1 + L2)

- **Penalty:** λ [l1\_ratio · Σ|βⱼ| + (1 − l1\_ratio) · Σβⱼ²]
- **Effect:** Feature selection (like Lasso) + grouping effect (correlated features tend to stay or go together).
- **When to use:** Both multicollinearity AND sparse signal; groups of correlated predictors where you want the group to survive or be dropped together.
- **l1\_ratio = 1** → Lasso; **l1\_ratio = 0** → Ridge.

## Alpha (λ) interpretation

- **α = 0:** Equivalent to OLS (no regularisation).
- **Large α:** Strong regularisation; coefficients collapse toward zero.
- The CV curve shows the trade-off. The 1-SE rule selects the most parsimonious α within one standard error of the best score.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Using Lasso when features are highly correlated | Switch to ElasticNet |
| Comparing raw regularised R² to OLS R² as a failure | OLS maximises in-sample R² by design; compare test-set scores |
| Interpreting zeroed Lasso coefficients as "causally irrelevant" | Zero means the model chose not to use that feature; correlated features may carry the same signal |
| Forgetting to standardise features | Regularisation penalises raw coefficient magnitude — on un-standardised data, large-scale features are penalised less |
