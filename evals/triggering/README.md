# Triggering evals

These evals test whether Claude *invokes the right skill at the right time*
from the SKILL.md descriptions alone — they are prompt-level tests, not
Python tests, and are run against a live model rather than in pytest.

## Cases to cover

| Prompt | Expected skill |
|---|---|
| "Fit a regression of price on sqft and bedrooms" | linear-regression |
| "Run OLS on this CSV, target is wage" | linear-regression |
| "Can I trust this model? Are the assumptions met?" | diagnostics |
| "Is this overfitting or underfitting?" | diagnostics |
| "Which rows are dragging my fit around?" | diagnostics |
| "What's the relationship between X and Y here?" (continuous target) | linear-regression |
| "Summarise this dataset" | *neither* (no fit requested) |

## Method

For each case, present the prompt plus the available SKILL.md frontmatter
descriptions and check the model selects the expected skill (or declines).
Record hits/misses; a description change that drops the hit rate is a
regression in the pack even if all Python tests pass.
