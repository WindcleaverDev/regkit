# Model selection criteria

## AIC and BIC

**AIC** (Akaike Information Criterion):

    AIC = -2 * log-likelihood + 2 * k

where `k` is the number of free parameters (coefficients + intercept). Lower is better.
AIC penalises complexity less severely than BIC and tends to select richer models.

**BIC** (Bayesian Information Criterion):

    BIC = -2 * log-likelihood + k * log(n)

The `log(n)` penalty grows with sample size, favouring parsimony more than AIC.
BIC is consistent (selects the true model as n → ∞ given it is in the candidate set);
AIC is efficient (minimises expected prediction error).

### When to prefer each

| Goal | Criterion |
|------|-----------|
| Prediction accuracy | AIC |
| True model identification | BIC |
| Small samples (n < 40) | BIC (less overfitting) |
| Large samples, complex truth | AIC |

## Likelihood-ratio test (LR test)

For **nested** models (model A ⊆ model B in terms of predictors):

    LR = 2 * (ll_full - ll_nested) ~ chi²(df)
    df = k_full - k_nested

Null hypothesis: the additional parameters in the full model are jointly zero.
A significant p-value (< α) means the full model explains significantly more variance.

**Validity conditions:**
- Both models must be estimated by maximum likelihood on the same observations.
- The models must be strictly nested (same family, same outcome, same data).
- Robust-SE variants (e.g. HC3) change the SE but not the log-likelihood — LR test
  is still valid for the underlying MLE estimate.

## Adjusted R² vs AIC/BIC

Adjusted R² penalises for extra predictors but does so only relative to the number of
observations and parameters, not via likelihood. For linear models it relates to AIC
as follows (approximate):

    AIC ≈ n * log(RSS/n) + 2k + const

so lower AIC correlates with higher adjusted R². However adj-R² is scale-free while
AIC can be used across models fit on the same data.

## Cross-validation

When models differ in family (e.g. linear vs regularised), AIC comparison is less
interpretable. Held-out CV mean squared error is the universal alternative.
`cv_score_mean` from the regularised skill provides this directly.
