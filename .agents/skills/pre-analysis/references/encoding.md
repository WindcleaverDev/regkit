# Categorical Encoding Reference

## Default: dummy (one-hot) encoding with drop-first

The pack uses `pd.get_dummies(drop_first=True)` via `validators.coerce_features`. This creates k-1 binary columns for a feature with k levels, dropping the first level as the reference category.

**When to use:** any categorical feature with low-to-moderate cardinality (n_unique ≤ 20).

**Interpretation:** each dummy coefficient is the difference in the outcome between that level and the reference category, holding other features constant.

**Reference category:** the dropped level (alphabetically first by default). Always mention it when narrating results: "compared to {reference_level}…"

## High-cardinality categoricals (n_unique > 20)

One-hot encoding a feature with 50 levels adds 49 columns. This causes:
- Increased VIF among the dummies
- Many low-frequency cells with unreliable estimates
- Model overfitting if n is not large relative to the number of dummies added

**Options:**
1. **Group rare levels** — combine levels with < 5% frequency into an "other" bucket before encoding.
2. **Target encoding** (mean encoding) — replace each level with the mean target value for that level. Powerful but leaks target information; requires out-of-fold encoding to avoid.
3. **Drop the feature** — if n_unique ≈ n (quasi-ID), the feature has no signal.
4. **Use a tree-based model** — trees handle high cardinality natively (out of scope for this pack).

The `high_cardinality` flag fires when n_unique > min(20, n/20). The `quasi_id` flag fires when n_unique == n.

## Ordinal encoding

For features with a meaningful order (e.g. education level: "high school" < "bachelor's" < "master's" < "PhD"), use integer codes rather than dummies. This imposes equal spacing between levels — acceptable when the levels are roughly equidistant.

The pack does not auto-detect ordinal features; you must encode them manually before passing to the fit skill.

## Binary features

Features with exactly two levels are encoded as a single 0/1 column. The pack handles this automatically — no drop_first is needed (the two dummies are perfectly collinear).

## Interaction terms

To model an interaction between two features (e.g. `sqft * has_pool`), create the product column before passing to the fit skill. The `interpretation_type = "interaction_term"` will apply automatically.

## Target encoding pitfalls

Target encoding can cause severe leakage if the encoding is computed on the full dataset before splitting:
- Each row's own target contributes to its encoded value
- This inflates training R² and deflates test R²

Always compute target encoding within a cross-validation fold or use out-of-fold means. The pack does not currently implement target encoding — use scikit-learn's `TargetEncoder` with `cv=5`.
