# Marginal Effects in Logistic Regression

## Why marginal effects?

Logistic regression is a nonlinear model. The change in predicted probability from a one-unit increase in X depends on where in the predictor space you are (near the extremes, probabilities barely move; near P=0.5, they move the most). Marginal effects aggregate this heterogeneity into a single interpretable number.

## Types of marginal effects

### AME — Average Marginal Effect (what this pack reports)

Compute the partial effect ∂P/∂x at every observation, then average across the sample.

```
AME(x_j) = (1/n) Σᵢ ∂P(Yᵢ=1|Xᵢ) / ∂x_j
```

**This is the default.** It answers: "On average, how much does P(Y=1) shift when X increases by 1?"

### MEM — Marginal Effect at the Mean

Compute the partial effect at X = X̄ (the mean of each predictor).

```
MEM(x_j) = ∂P(Y=1|X=X̄) / ∂x_j
```

**Pitfall:** The "average person" may not be a realistic or even possible combination of predictor values (e.g., "average sex" doesn't exist).

### MER — Marginal Effect at a Representative value

Same as MEM but using a chosen representative observation. Useful for profiling specific subgroups.

## Dummy variables

For binary features, `get_margeff(dummy=True)` computes the **discrete** change:

```
Discrete effect = E[P(Y=1|x_j=1, X₋ⱼ)] − E[P(Y=1|x_j=0, X₋ⱼ)]
```

This is more meaningful than the derivative (which ignores the discrete nature of the variable).

## Standard errors

AME standard errors are computed by the **delta method**: propagate the uncertainty in the coefficient estimates through the nonlinear transformation. The delta-method SE is approximate; bootstrap is more accurate but expensive.

## Interpreting AMEs

| AME | Interpretation |
|-----|----------------|
| 0.05 | A one-unit increase in X is associated with a +5 pp increase in P(Y=1) on average |
| −0.12 | A one-unit increase in X is associated with a −12 pp decrease in P(Y=1) on average |
| 0.001 | Effect is practically negligible regardless of p-value |

**Unit sensitivity.** A feature measured in thousands will have an AME 1000× smaller than the same feature measured in ones. Always consider the feature's scale when comparing AMEs.
