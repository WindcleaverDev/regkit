# Target Transformations Reference

## When to transform the target

Transform the target when its distribution violates OLS assumptions or when the relationship with predictors is multiplicative rather than additive.

## Log transform (`np.log`)

**When to use:**
- Target is continuous, strictly positive (all values > 0)
- Skewness > 1.5 (right-skewed distribution)
- Residuals fan out with fitted values (heteroscedasticity)
- You expect a multiplicative data-generating process (e.g. prices, incomes, biological counts)

**How:**
```bash
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data data.csv --target price --features sqft,rooms \
    --log-target --output results/
```

**Interpretation after log transform:**
- Linear feature: "A one-unit increase in X is associated with a `(exp(β) - 1) × 100`% change in the target."
- Log feature (log-log): "A 1% increase in X is associated with a `β`% change in the target" (elasticity).
- The `linear-regression` interpret.py handles these automatically based on `target_transform`.

**When log fails:**
- Target has zero or negative values → use Box-Cox or Yeo-Johnson instead.
- After logging, residuals are still skewed → the relationship may need polynomial terms, not a transform.

## Box-Cox transform

**When to use:**
- All values > 0, moderate skewness (|skewness| > 1.0 but ≤ 1.5)
- Log overshoots: log-transformed target is now left-skewed

**How:** use `scipy.stats.boxcox(y)`. The optimal λ is estimated by maximum likelihood.

**Interpretability cost:** Box-Cox coefficients have no simple verbal interpretation. Prefer log when the skewness threshold permits it.

## Yeo-Johnson transform

**When to use:**
- Target has zero or negative values (Box-Cox requires strict positivity)
- Otherwise similar to Box-Cox

**How:** `sklearn.preprocessing.PowerTransformer(method='yeo-johnson')`.

## Square-root transform

**When to use:**
- Count data (non-negative integers) with moderate skewness
- Quicker to explain than log but less powerful

## Winsorisation

**When to use:**
- Extreme outliers drive skewness but the bulk of the distribution is normal
- Outlier count > 5% of n from the IQR rule

**How:** `scipy.stats.mstats.winsorize(y, limits=[0.01, 0.01])`. Clips the top and bottom 1%.

**Caution:** winsorisation destroys genuine extreme values. Prefer a transform if the outliers are real observations.

## No transform

**When to use:**
- Skewness ≤ 1.0 with no heavy tails
- n is large enough for CLT to apply (n > 200 for mildly non-normal residuals)
- Interpretability is critical and transforms would confuse the audience

## Decision tree

```
target all positive?
├── yes → skewness > 1.5?
│         ├── yes → log_transform
│         └── no  → |skewness| > 1.0?
│                   ├── yes → box_cox
│                   └── no  → no transform needed
└── no  → has zeros or negatives?
          ├── yes → yeo_johnson
          └── no  → (shouldn't reach here)

count data (non-negative integer)?
└── sqrt_transform (lighter than log)

outlier_count > 5% of n?
└── winsorize (in addition to any of the above)
```
