# Cross-Validation Strategies for Alpha Selection

## Why CV?

The regularisation strength α controls the bias–variance trade-off. A too-small α overfits; a too-large α underfits. Cross-validation estimates generalisation performance at each α without using test data.

## Default: k-Fold CV (k = 5)

Five-fold CV is the default in this skill. It balances:

- **Bias:** each fold trains on 80% of data — close to the full-data estimate.
- **Variance:** five independent test sets give a stable mean score.
- **Runtime:** five fits per alpha point.

For n < 100, consider increasing folds to 10 or using leave-one-out (LOO-CV). Pass `--cv-folds 10`.

## Scoring metric: R²

R² is the default scoring metric displayed in the CV curve. It is directly comparable across datasets (bounded, scaled to variance). Note:

- CV R² can be negative when the model is worse than predicting the mean.
- MSE is mathematically equivalent for alpha selection but less interpretable.

## Alpha selection rules

### min (default)

Selects the α that maximises mean CV score. Minimises expected prediction error on new data.

```
selected_α = arg max_{α} E[R²(α)]
```

### 1-SE rule (`--alpha-rule 1se`)

Selects the **largest** α whose mean score is within one standard error of the best. This is a *parsimony preference*: it accepts slightly lower performance in exchange for more regularisation (fewer non-zero coefficients in Lasso/ElasticNet; more shrinkage in Ridge).

```
selected_α = max α s.t. E[R²(α)] ≥ E[R²(α*)] − SE(R²(α*))
```

Use 1-SE when:
- You expect future data to differ from training data (higher uncertainty).
- Interpretability and sparsity matter more than peak CV score.

## Alpha grid

| Method | Default grid |
|--------|-------------|
| Ridge | 10⁻⁴ to 10⁴ (100 log-spaced) |
| Lasso | α_max to α_max × 10⁻⁴ (100 log-spaced) |
| ElasticNet | Same as Lasso, at fixed l1_ratio |

α_max for Lasso is the smallest α that zeros all coefficients: `α_max = max_j |⟨X_j, y⟩| / n`.

## CV curve interpretation

The report shows mean CV R² ± 1 SD against log₁₀(α):

- **Rising left, falling right:** classic U-shape — clear optimum in the middle.
- **Flat plateau:** insensitive to α in this region — any value here is fine.
- **Monotone rising to the right:** model benefits from more regularisation; the grid may not extend far enough.
- **Always near zero or negative:** data may not support regression; check for target leakage or misspecified features.
