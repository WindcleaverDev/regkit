# Multicollinearity (VIF) Remediation Reference

## What VIF measures

Variance Inflation Factor for feature j is:

```
VIF_j = 1 / (1 - R²_j)
```

where R²_j is the R² from regressing feature j on all other features. It measures how much the variance of β_j is inflated by collinearity.

- VIF = 1: no collinearity
- VIF = 5: moderate — CIs are √5 ≈ 2.24× wider than they would be without collinearity
- VIF = 10: severe — CIs are √10 ≈ 3.16× wider; coefficients may have wrong signs

## The `HIGH_VIF` flag

- Severity WARN: max VIF > 5 (at least one feature moderately collinear)
- Severity HIGH: max VIF > 10 (severe multicollinearity; coefficients unreliable)

Collinearity inflates standard errors and widens CIs. Point estimates (coefficients) remain unbiased but are imprecise — small data perturbations can swing coefficients dramatically.

## Remediation options (ranked by preference)

### 1. Drop one of the collinear pair

The simplest fix. If two features are near-perfectly correlated (|r| > 0.9), keeping both adds no information and inflates VIF for both.

**How to choose which to drop:**
- Prefer the feature with stronger theoretical justification
- Prefer the one with lower standalone p-value in a univariate regression
- Prefer the one that's easier to measure / less noisy

### 2. Combine into a composite index

If the collinear features measure the same latent construct (e.g. `s1`, `s2`, `s3` in the diabetes dataset are all serum measures), sum or average them into a single index.

**Example:** `serum_composite = (s1 + s2 + s3) / 3`

This preserves information while eliminating collinearity among the components.

### 3. Use regularised regression (Ridge)

Ridge regression adds a penalty `λ||β||²` that shrinks all coefficients toward zero. This stabilises estimation even under severe collinearity — ridge does not require dropping features.

```bash
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data data.csv --target price \
    --features f1,f2,f3,f4 \
    --method ridge \
    --output results/
```

Ridge is preferred when:
- All features are theoretically important and shouldn't be dropped
- The collinearity is structural (e.g. polynomial terms, interaction terms always correlate with their components)

### 4. Use Lasso for feature selection

Lasso (λ||β||₁) drives some coefficients to exactly zero, performing automatic feature selection. For collinear groups, Lasso will typically keep one and zero out the others.

```bash
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --method lasso --output results/
```

**Caveat:** for collinear groups, Lasso's choice of which feature to keep can be arbitrary (whichever is slightly more correlated with the target). Prefer Ridge if you want all features retained with shrinkage.

## When VIF is misleading

- **Interaction terms and polynomial terms** always have high VIF with their parent features (e.g. `x²` is correlated with `x`). This is expected and not a problem — the model is correctly specified; the VIF is an artefact of the functional form. Interpret interaction VIF with caution.
- **Dummy variables from the same categorical feature** are correlated by construction. High VIF among dummies from the same feature is normal.
- **Near-zero-variance features** inflate VIF. Remove `near_constant` features first.

## VIF thresholds in context

The commonly cited thresholds (5, 10) are rules of thumb. In practice:
- If your goal is prediction, collinearity matters less — predictions are still accurate even if individual coefficients are noisy.
- If your goal is inference (understanding which features matter and by how much), collinearity is a serious problem — address it before trusting coefficient signs or magnitudes.
