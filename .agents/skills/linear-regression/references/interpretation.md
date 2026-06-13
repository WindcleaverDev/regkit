# Coefficient interpretation guide

How to phrase each `interpretation_type`, what confidence means in that
context, and which caveats to attach. The `fact` field in each
`InterpretationFact` is the canonical claim — verbalise it; do not
recompute or embellish the numbers.

All interpretations carry two standing caveats:

1. **Holding other features constant** (ceteris paribus) — the effect is
   conditional on the other predictors in the model staying fixed.
2. **Association, not causation** — OLS on observational data identifies
   conditional associations, not causal effects.

---

## linear_linear

**When it applies:** neither target nor feature is transformed.

**Canonical form:** "A one-unit increase in {feature} is associated with a
{|β|} {increase/decrease} in {target}, holding other features constant."

**Confidence:** high when p < 0.01 and the CI is narrower than |β|; the unit
effect is precisely estimated. Low confidence means the CI likely spans zero —
present the direction as inconclusive.

**Caveats:** check the unit scale is meaningful to the audience (a "one-unit
increase" in sqft is one square foot — often better verbalised per 100 units
by multiplying β accordingly — but the canonical fact stays per-unit).

## log_log_elasticity

**When it applies:** target and feature are both log-transformed.

**Canonical form:** "A 1% increase in {feature} is associated with a
{β×100/100 = β}% change in {target}…" — i.e. β is directly an elasticity.

**Confidence:** as elsewhere; note elasticities are unitless, so they're the
easiest coefficients to compare across features.

**Caveats:** elasticity is local (evaluated at small percentage changes);
both variables must be strictly positive for logs to exist.

**Worked example:** modelling log(price) ~ log(sqft), β = 0.43 →
"A 1% increase in sqft is associated with a 0.43% increase in price, holding
other features constant."

## log_linear_semi_elasticity

**When it applies:** target is log-transformed, feature is not.

**Canonical form:** "A one-unit increase in {feature} is associated with a
{(exp(β)−1)×100}% {increase/decrease} in {target}…"

**Confidence:** as elsewhere. For small |β| (< 0.1), β×100 ≈ the percentage
effect; the exact form exp(β)−1 is what the fact uses.

**Caveats:** the percentage effect compounds — a 10-unit increase is
(exp(10β)−1), not 10× the one-unit effect.

**Worked example:** modelling log(wage) ~ education, β = 0.08 →
exp(0.08)−1 = 8.33% → "Each additional year of education is associated with
an 8.33% increase in wage, holding other features constant."

## linear_log

**When it applies:** feature is log-transformed, target is not.

**Canonical form:** "A 1% increase in {feature} is associated with a
{β/100} {increase/decrease} in {target}…"

**Caveats:** effects are in target units per *percentage* change of the
feature; doubling the feature corresponds to β×log(2).

## binary_dummy

**When it applies:** the feature is a 0/1 dummy from a two-level variable.

**Canonical form:** "{feature}={level} is associated with a {|β|}
{increase/decrease} in {target} compared to the reference category…"

**Confidence:** as elsewhere.

**Caveats:** the comparison is against the dropped reference level — name it
when verbalising. A dummy has no "one-unit increase" reading; it's a group
contrast.

## categorical_dummy

**When it applies:** one dummy from a variable with 3+ levels
(one-hot encoded, first level dropped).

**Canonical form:** same as binary_dummy, but always name the baseline:
"…compared to {baseline level}".

**Caveats:** each dummy is a pairwise contrast with the *baseline*, not with
the other levels. Joint significance of the whole categorical variable needs
an F-test, not the individual p-values. Changing the dropped level changes
every dummy coefficient — the model is the same; the parameterisation isn't.

## polynomial_term

**When it applies:** the feature is a squared (or higher-order) term, e.g.
`age^2` or `age_sq`.

**Canonical form:** "{base} has a nonlinear relationship with {target}; the
{base}² term is {β} (p={p}), indicating {convex/concave} curvature."

**Caveats:** never interpret the squared term alone — the marginal effect of
the base feature is β₁ + 2β₂·x and depends on where you evaluate it. Report
the turning point (−β₁ / 2β₂) when it lies inside the observed data range.

## interaction_term

**When it applies:** product features such as `x1:x2`.

**Canonical form:** "The effect of {f1} on {target} varies with {f2}; the
interaction coefficient is {β} (p={p})."

**Caveats:** main-effect coefficients now mean "the effect when the other
interacting variable equals zero" — often outside the data range. Centre the
variables or verbalise at meaningful moderator values.

---

## Standardized coefficients

When `--standardize` is passed, each non-intercept row also carries
`standardized_coefficient` = β · σₓ / σᵧ — the effect in standard-deviation
units.

**When to report them:** when features are on incommensurable scales and the
user asks "which predictor matters most?" Standardized βs make magnitudes
comparable.

**What they mean:** "A one-standard-deviation increase in {feature} is
associated with a {std β} standard-deviation change in {target}."

**Caveats:** they inherit the sample's standard deviations, so they aren't
portable across datasets; dummies have no natural SD interpretation — avoid
standardized βs for categorical contrasts.
