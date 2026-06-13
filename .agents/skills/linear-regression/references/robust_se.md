# Heteroscedasticity-robust standard errors (HC variants)

## What heteroscedasticity is and why robust SEs help

OLS assumes the error variance is constant across observations
(homoscedasticity). When it isn't — residual spread grows with fitted values,
or differs across groups — the coefficient estimates remain unbiased, but the
classical standard errors are wrong, usually too small. That inflates t-stats
and produces falsely confident p-values and confidence intervals.

Heteroscedasticity-consistent (HC) standard errors fix the *inference* without
changing the model: they re-estimate the coefficient covariance matrix using
the observed squared residuals instead of assuming a single σ².

**Key fact: HC variants change standard errors, t-stats, p-values, and CIs
only. The coefficients themselves are identical to plain OLS.**

## The variants

| Variant | Description | When to use |
|---|---|---|
| **HC0** | White's original (1980) estimator. Uses raw squared residuals. Downward-biased in small samples. | Large n, mostly of historical interest |
| **HC1** | HC0 with a degrees-of-freedom correction n/(n−k−1). The Stata default (`robust`). | Good general default for moderate-to-large n |
| **HC2** | Scales each residual by 1/(1−hᵢ) where hᵢ is leverage. Unbiased under homoscedasticity. | When high-leverage points exist |
| **HC3** | Davidson–MacKinnon: scales by 1/(1−hᵢ)². Approximates the jackknife; most conservative. | Recommended for n < 250, or when in doubt |

## Decision tree

1. **Diagnostics show no heteroscedasticity?** → plain OLS SEs are fine.
2. **n < 250?** → use **HC3**.
3. **High-leverage observations flagged by the diagnostics skill?** → use
   **HC2** (or HC3 if also small n).
4. **Otherwise** → **HC1** matches what most applied economists report.

## Practical notes

- Robust SEs are a patch for *inference*, not a fix for the model. If the
  heteroscedasticity reflects a misspecified mean (e.g. multiplicative
  errors), a log transform of the target often fixes both the variance and
  the interpretation — prefer that when the target is positive and skewed.
- Comparing plain vs robust SEs is itself a diagnostic: if they differ a lot,
  take the heteroscedasticity seriously.
- Robust SEs do not address autocorrelation. For time-ordered data with
  serial correlation, HAC (Newey–West) errors are the analogue — out of
  scope for this skill.
