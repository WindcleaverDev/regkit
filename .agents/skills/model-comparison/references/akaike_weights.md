# Akaike weights

Akaike weights convert raw AIC values into a probability-like summary of relative
model evidence (Burnham & Anderson, 2002).

## Computation

    Δ_i  = AIC_i − min(AIC)
    w_i  = exp(−Δ_i / 2) / Σ_j exp(−Δ_j / 2)

The weights sum to 1 across the candidate set.

## Interpretation

| Δ AIC (best model vs candidate) | Evidence against candidate |
|----------------------------------|---------------------------|
| 0–2 | Substantial support — models nearly equivalent |
| 4–7 | Considerably less support |
| > 10 | Essentially no support |

A weight w_i ≈ 0.95 means model i captures ~95% of the Akaike weight — a clear winner.
When two models split weight 0.6 / 0.4, the comparison is underdetermined.

## Caveats

1. **Same data required**: AIC weights are meaningless across models fit on different
   observations or different outcome variables.
2. **Candidate set dependence**: weights depend on which models you include. Adding a
   terrible model shifts weights but does not change the relative ranking of good ones.
3. **Log-transformed outcomes**: AIC for a model predicting log(y) is on a different
   scale than for a model predicting y directly. Do not mix.
4. **Regularised models**: AIC for Ridge/Lasso uses the effective degrees of freedom
   (non-zero coefficients) as an approximation. This understates the true penalty.
   Treat regularised model AICs as rough indicators only.

## Practical guidance

- **weight > 0.80**: clear winner; proceed with that model.
- **max Δ AIC < 2**: competitive tie; select based on parsimony or theory.
- **No single weight > 0.50**: model uncertainty is substantial; consider model
  averaging or additional data collection.
