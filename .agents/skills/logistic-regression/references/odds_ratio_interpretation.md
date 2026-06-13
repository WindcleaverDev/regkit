# Odds Ratio Interpretation

## What an odds ratio is

The odds ratio (OR) measures how the *odds* of the outcome change when a predictor increases by one unit, holding other features constant.

- OR > 1: predictor increases the odds (positive association)
- OR < 1: predictor decreases the odds (negative association)
- OR = 1: no association

**Odds vs probability.** Odds = P / (1 − P). For rare outcomes (P < 0.1), odds ≈ probability. For common outcomes, they diverge significantly.

## The "twice as likely" mistake

"OR = 2" does NOT mean "twice as likely." It means "twice the odds." When the baseline probability is high, the probability ratio can be much smaller than the OR.

| Baseline P | OR = 2 | Actual ΔP |
|------------|--------|-----------|
| 0.05 | 2 | +0.095 → 0.095 |
| 0.30 | 2 | +0.286 → 0.586 |
| 0.50 | 2 | +0.333 → 0.833 |

At baseline P = 50%, OR = 2 means going from 50% to 83% — not doubling to 100%.

## When to use ORs vs AMEs

**Odds ratios** are appropriate when:
- You need a multiplicative effect that stays constant across subgroups
- You're doing epidemiological case-control research (case-control studies can only estimate ORs, not probabilities)
- The outcome is rare (ORs ≈ RRs)

**Average marginal effects (AMEs)** are appropriate when:
- You want to communicate probability changes to a non-technical audience
- The outcome is common (P > 10%)
- You need a direct answer to "how much does X shift P(Y=1)?"

This pack reports **both** — ORs for statistical completeness, AMEs for primary interpretation.

## Reading the CI

- CI includes 1.0 → not statistically significant (log-odds CI includes 0)
- Narrow CI → precise estimate
- Wide CI → noisy estimate, common with small n or rare events

## Perfect separation

If OR is very large (> exp(10)) with a wide CI, suspect **quasi-complete separation**: the predictor perfectly or near-perfectly separates the two classes. The MLE is undefined. Symptoms: non-convergence, very large coefficient with huge SE. Remedy: Firth logistic regression, penalised likelihood, or remove the separator.
