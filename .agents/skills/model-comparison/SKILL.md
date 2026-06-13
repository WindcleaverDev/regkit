# model-comparison skill

Compare two or more regression model reports from any skill in the regression pack.
Produces a `ModelComparisonReport` with Akaike weights, likelihood-ratio tests for
nested pairs, a coefficient comparison chart, and a structured verdict.

## Usage

```bash
python .agents/skills/model-comparison/scripts/compare.py \
    --reports out/ols/report.json out/ridge/report.json out/lasso/report.json \
    --names "OLS" "Ridge" "Lasso" \
    --output out/comparison/ \
    [--alpha 0.05] \
    [--dataset-name "tips"]
```

### Required arguments

| Argument | Description |
|----------|-------------|
| `--reports` | Space-separated paths to `report.json` files (≥ 2) |
| `--names` | Human-readable model names (same order as `--reports`) |
| `--output` | Output directory for `report.json` and `report.html` |

### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--alpha` | `0.05` | Significance level for LR tests |
| `--dataset-name` | `""` | Label shown in the report header |

## What the skill does

1. **Ingest** — loads each report.json, sniffs the model family (linear, logistic,
   ridge, lasso, elasticnet), and derives a `ModelEntry` with AIC, BIC, n, k, and
   primary fit quality (adj-R² for linear family, pseudo-R² for logistic).

2. **Nesting detection** — for each pair of same-family, same-outcome models, checks
   whether one's feature set is a strict subset of the other's.

3. **LR test** — for each nested pair: `LR = 2*(ll_full − ll_nested) ~ chi²(df)`.

4. **Akaike weights** — `Δ_i = AIC_i − min(AIC)`, `w_i = exp(−Δ_i/2) / Σ exp(−Δ_j/2)`.
   Only computed for models with valid AIC values.

5. **Verdict** — one of:
   - `clear_winner` — one model has Akaike weight ≥ 0.80 or LR test rejects simpler models
   - `competitive_tie` — max Δ AIC < 2
   - `complementary_strengths` — different families or targets; primary metrics close
   - `all_inadequate` — all primary metrics below 0.15

6. **Report** — HTML with verdict card, models table, Akaike weight bars, Δ AIC bars,
   LR test table, coefficient comparison chart, and flags.

## Output schema

```
ModelComparisonReport
├── models: list[ModelEntry]
├── lr_tests: list[LRTestResult]
├── akaike_weights: AkaikeWeights | None
├── verdict: ComparisonVerdict
│   ├── overall: "clear_winner" | "competitive_tie" | "complementary_strengths" | "all_inadequate"
│   ├── recommended_model: str | None
│   ├── headline: str
│   └── rationale: str
├── flags: list[Flag]
│   ├── SAMPLE_SIZE_MISMATCH — models trained on different n
│   ├── CROSS_FAMILY_COMPARISON — families differ; AIC not valid
│   └── NO_FORMAL_COMPARISON — neither Akaike weights nor LR tests available
└── recommendations: list[Recommendation]
```

## Comparison validity rules

| Scenario | Akaike weights | LR test |
|----------|----------------|---------|
| Same family, same outcome, different features | ✓ | ✓ (nested pairs) |
| OLS vs OLS+HC3 (same features) | ✓ (equal weight) | — |
| OLS vs OLS(log-target) | ✗ | ✗ |
| Linear vs logistic | ✗ | ✗ |
| Any vs Ridge/Lasso (same outcome) | ✓ (approx) | ✗ |

## References

- [Model selection criteria](references/criteria.md)
- [Akaike weights](references/akaike_weights.md)
- [Cross-family comparison](references/cross_family.md)
