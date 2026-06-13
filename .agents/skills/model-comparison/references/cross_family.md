# Cross-family model comparison

## What can and cannot be compared directly

| Comparison | AIC valid | LR test valid | adj-R² / pseudo-R² |
|------------|-----------|---------------|---------------------|
| OLS vs OLS (more features) | ✓ | ✓ (nested) | ✓ |
| OLS vs Ridge/Lasso | ✓* | ✗ | ✓ |
| OLS vs OLS (log-target) | ✗ | ✗ | ✗ (different scale) |
| Linear vs Logistic | ✗ | ✗ | ✗ (different scale) |

*AIC for regularised models uses effective df approximation; treat as rough indicator.

## Comparing OLS and robust-SE OLS

Robust SE (HC1/HC2/HC3) corrects standard errors but does **not** change the
log-likelihood or coefficient estimates. Therefore:
- AIC and BIC are identical to plain OLS.
- LR test results are the same as OLS.
- Akaike weights between OLS and OLS+robust are always 0.5 / 0.5 — uninformative.
- The correct comparison is qualitative: do significance conclusions change? If yes,
  the data are heteroscedastic and robust SE is preferred on reliability grounds.

## Comparing log-transformed vs untransformed outcome

AIC measures fit on the outcome scale the model was trained on. A model for log(y) and
a model for y cannot be compared on raw AIC. To compare on the same scale:
1. Back-transform predictions: ŷ = exp(log_ŷ).
2. Compute out-of-sample RMSE on the original y scale.
3. Use RMSE (or RMSLE) rather than AIC for the comparison.

## Comparing within the regularisation family

Ridge vs Lasso vs ElasticNet can all be compared with Akaike weights if trained on
identical data, since the AIC approximation is computed consistently across them.
The model-comparison skill computes these weights automatically.

## When to use each comparison method

| Situation | Use |
|-----------|-----|
| All models same family + outcome | AIC / Akaike weights |
| Nested models (same data, same outcome) | LR test |
| Different outcome transforms | Out-of-sample RMSE |
| Cross-family (linear vs logistic) | Domain-appropriate metric (R² vs AUC) |
| Regularised vs OLS | CV score (already in regularised report) |
