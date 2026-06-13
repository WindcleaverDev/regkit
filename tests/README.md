# Real-dataset test suite

Nine datasets covering all six skills. Every CSV is generated offline from
packages already in the dependency graph — no downloads, no API keys:

```bash
uv run python tests/make_datasets.py   # writes tests/data/*.csv
```

---

## Dataset catalogue

| File | n | Cols | Target | Type | Primary skill(s) |
|------|---|------|--------|------|-----------------|
| `tips.csv` | 244 | 7 | `tip` | continuous | linear-regression, diagnostics, pre-analysis |
| `diabetes.csv` | 442 | 11 | `progression` | continuous | regularized-regression, linear-regression, model-comparison, pre-analysis |
| `engel.csv` | 235 | 2 | `foodexp` | continuous | linear-regression, diagnostics |
| `gapminder_2007.csv` | 142 | 5 | `lifeExp` | continuous | linear-regression, diagnostics |
| `statecrime.csv` | 51 | 8 | `violent` | continuous | diagnostics |
| `macro.csv` | 203 | 14 | `realcons` | continuous | diagnostics |
| `longley.csv` | 16 | 7 | `TOTEMP` | continuous | theory only (n < 30; too small for CLI) |
| `breast_cancer.csv` | 569 | 31 | `malignant` | binary (1=cancer) | logistic-regression |
| `affairs.csv` | 6,366 | 9 | `had_affair` | binary (1=yes) | logistic-regression |

---

## Skill-by-skill guide

### 1. `linear-regression`

#### `tips.csv` — Categorical features + mild heteroscedasticity

```bash
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/tips.csv \
    --target tip \
    --features total_bill,size,sex,smoker,day,time \
    --output /tmp/regpack/tips
```

**Expected output:**
```
✓ Linear regression fit complete
  n = 244, k = 8
  R² = 0.4701  (adj: 0.4520)
  The model explains 45.2% of variance in tip (1 of 8 predictors significant at p < 0.05).
```

What to look for: `sex`, `smoker`, `day`, `time` are one-hot encoded automatically;
the report shows dummy interpretation facts with reference-category caveats.
Only `total_bill` is significant — all categorical dummies are non-significant.

#### `engel.csv` — Textbook simple regression

```bash
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/engel.csv \
    --target foodexp \
    --features income \
    --output /tmp/regpack/engel
```

**Expected output:**
```
✓ Linear regression fit complete
  n = 235, k = 1
  R² = 0.8436  (adj: 0.8429)
```

#### `gapminder_2007.csv` — Nonlinearity (log-linear relationship)

```bash
# Step 1: naive linear fit (Ramsey RESET will fail)
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/gapminder_2007.csv \
    --target lifeExp --features gdpPercap,pop \
    --output /tmp/regpack/gap_linear

# Step 2: log-transform GDP (correct specification)
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/gapminder_2007.csv \
    --target lifeExp --features gdpPercap,pop \
    --log-features gdpPercap \
    --output /tmp/regpack/gap_log
```

---

### 2. `diagnostics`

#### `tips.csv` — Heteroscedasticity (Breusch-Pagan FAIL)

```bash
uv run python .agents/skills/diagnostics/scripts/diagnose.py \
    --fit-report /tmp/regpack/tips/report.json \
    --data tests/data/tips.csv \
    --output /tmp/regpack/tips
```

**Expected outcome:** `HETEROSCEDASTICITY fail` (tip variance grows with bill size) + `use_robust_se` recommendation.

#### `statecrime.csv` — Influence analysis (named outliers)

```bash
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/statecrime.csv \
    --target violent \
    --features hs_grad,poverty,single,white,urban \
    --output /tmp/regpack/crime

uv run python .agents/skills/diagnostics/scripts/diagnose.py \
    --fit-report /tmp/regpack/crime/report.json \
    --data tests/data/statecrime.csv \
    --output /tmp/regpack/crime
```

**Expected outcome:** `HIGH_COOKS_D` flag for DC, Hawaii, Mississippi.
Bias/variance verdict: `high_variance` (n=51 with 5 predictors).

#### `macro.csv` — Autocorrelation (Durbin-Watson ≪ 1.5 despite R² ≈ 0.999)

```bash
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/macro.csv \
    --target realcons \
    --features realgdp,unemp \
    --output /tmp/regpack/macro

uv run python .agents/skills/diagnostics/scripts/diagnose.py \
    --fit-report /tmp/regpack/macro/report.json \
    --data tests/data/macro.csv \
    --output /tmp/regpack/macro
```

**Expected outcome:** `AUTOCORRELATION fail`. The agent should lead with the
Durbin-Watson failure, not the misleadingly high R².

---

### 3. `pre-analysis`

#### `diabetes.csv` — HIGH_VIF → recommends regularized

```bash
uv run python .agents/skills/pre-analysis/scripts/audit.py \
    --data tests/data/diabetes.csv \
    --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --output /tmp/regpack/diabetes_pre
```

**Expected output:**
```
✓ Pre-analysis complete
  n = 442, 10 feature(s)
  flags: 1 high, 0 warn, 0 info
  preferred estimator: regularized_regression
```

The `s1`–`s6` serum measures are strongly collinear (s1/s2 ≈ total/LDL
cholesterol); VIF > 10 fires a HIGH severity flag and tips the recommendation
to regularised regression.

#### `tips.csv` — Categoricals + skewed target

```bash
uv run python .agents/skills/pre-analysis/scripts/audit.py \
    --data tests/data/tips.csv \
    --target tip \
    --features total_bill,size,sex,smoker,day,time \
    --output /tmp/regpack/tips_pre
```

**Expected output:**
```
✓ Pre-analysis complete
  n = 244, 6 feature(s)
  flags: 1 high, 1 warn, 0 info
  preferred estimator: regularized_regression
  → apply box_cox to target before fitting
```

---

### 4. `regularized-regression`

#### `diabetes.csv` — Canonical Lasso benchmark

The Efron et al. (2004) diabetes dataset is the standard Lasso benchmark.
At the 1-SE alpha, Lasso typically retains `bmi`, `bp`, `s3`, `s5` and one
or two others while zeroing the redundant serum measures.

```bash
# Lasso — 1-SE rule (parsimonious)
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data tests/data/diabetes.csv \
    --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --method lasso \
    --alpha-rule 1se \
    --output /tmp/regpack/diabetes_lasso
```

**Expected output:**
```
✓ Lasso regression fit complete
  n = 442, k = 10
  R² = 0.4822  (adj: 0.4702)
  selected α = 5.833
  features retained: 5 / 10
  zeroed: age, s1, s2, s4, s6
```

```bash
# Ridge — all features retained, collinear serum measures shrunk together
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data tests/data/diabetes.csv \
    --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --method ridge \
    --output /tmp/regpack/diabetes_ridge
```

**Expected output:**
```
✓ Ridge regression fit complete
  n = 442, k = 10
  R² = 0.5177  (adj: 0.5065)
  selected α = 0.2043
```

The regularisation path plot shows `s1` and `s2` tracking each other closely
(they are nearly collinear); both approach zero together at large α.

---

### 5. `logistic-regression`

#### `affairs.csv` — Interpretable AMEs, clear sign expectations

Fair (1978) marital-affairs survey. Expected AME signs: `rate_marriage` −,
`religious` −, `yrs_married` +, `age` −.

```bash
uv run python .agents/skills/logistic-regression/scripts/fit.py \
    --data tests/data/affairs.csv \
    --target had_affair \
    --positive-class 1 \
    --features rate_marriage,age,yrs_married,children,religious,educ,occupation \
    --output /tmp/regpack/affairs \
    --dataset-name "Fair (1978) affairs survey"
```

**Expected output:**
```
✓ Logistic regression fit complete
  n = 6,366  k = 7  pos_rate = 32.2%
  AUC = 0.7438  Brier = 0.1832  McFadden R² = 0.1326
  converged = True  6/7 significant at p < 0.05
```

Key interpretation: "A one-unit increase in marital satisfaction rating is
associated with a −X pp decrease in the probability of an affair."

```bash
# Robust SE variant (at n=6,366 the SEs converge but check for robustness)
uv run python .agents/skills/logistic-regression/scripts/fit.py \
    --data tests/data/affairs.csv \
    --target had_affair --positive-class 1 \
    --features rate_marriage,age,yrs_married,children,religious,educ,occupation \
    --robust-se HC3 \
    --output /tmp/regpack/affairs_hc3
```

#### `breast_cancer.csv` — Near-perfect separation; calibrated high-AUC model

```bash
uv run python .agents/skills/logistic-regression/scripts/fit.py \
    --data tests/data/breast_cancer.csv \
    --target malignant \
    --positive-class 1 \
    --features mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension \
    --output /tmp/regpack/bc_logistic \
    --dataset-name "Breast cancer (10 mean features)"
```

**Expected output:**
```
✓ Logistic regression fit complete
  n = 569  k = 10  pos_rate = 37.3%
  AUC = 0.9879  Brier = 0.0390  McFadden R² = 0.8055
  [high] CONVERGENCE_ISSUE: near-perfect separation (max |β| = 76.4)
```

The CONVERGENCE_ISSUE flag is a teaching moment: AUC = 0.99 is genuine, but
near-perfect separation inflates log-odds coefficients and makes their standard
errors unreliable. In practice, use penalised logistic (Firth regression) or
reduce the feature set.

---

### 6. `model-comparison`

#### `diabetes.csv` — OLS vs Lasso vs Ridge (AIC-based comparison)

```bash
# Generate three fits first (or reuse /tmp/regpack/* from above)
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/diabetes.csv --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --output /tmp/regpack/diabetes_ols

uv run python .agents/skills/model-comparison/scripts/compare.py \
    --reports \
        /tmp/regpack/diabetes_ols/report.json \
        /tmp/regpack/diabetes_lasso/report.json \
        /tmp/regpack/diabetes_ridge/report.json \
    --names "OLS" "Lasso (1-SE)" "Ridge" \
    --output /tmp/regpack/diabetes_comparison \
    --dataset-name "Diabetes"
```

**Expected output:**
```
✓ Model comparison complete
  Verdict: complementary_strengths  |  Recommended: OLS
  Akaike weights: OLS = 0.501; Lasso (1-SE) = 0.000; Ridge = 0.499
```

OLS and Ridge are nearly tied on AIC (Ridge shrinks coefficients but barely
changes residuals at the CV-optimal α). Lasso gets Δ AIC > 10 because it
zeroes 5 features, paying a meaningful fit cost for the simpler model.

#### `tips.csv` — Nested linear models (LR test: does `size` add signal?)

```bash
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/tips.csv --target tip \
    --features total_bill \
    --output /tmp/regpack/tips_simple

uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/tips.csv --target tip \
    --features total_bill,size \
    --output /tmp/regpack/tips_size

uv run python .agents/skills/model-comparison/scripts/compare.py \
    --reports \
        /tmp/regpack/tips_simple/report.json \
        /tmp/regpack/tips_size/report.json \
    --names "OLS (bill)" "OLS (bill+size)" \
    --output /tmp/regpack/tips_comparison
```

**Expected output:**
```
✓ Model comparison complete
  Verdict: clear_winner  |  Recommended: OLS (bill+size)
  Akaike weights: OLS (bill) = 0.175; OLS (bill+size) = 0.825
  LR test OLS (bill) vs OLS (bill+size): p = 0.0238
```

Note: adding the full set of categorical dummies (sex, smoker, day, time) to the bill-only
model actually *hurts* AIC — those 7 extra parameters have near-zero coefficients and the
penalisation outweighs the tiny R² gain. This is a useful illustration of how AIC
prevents overfitting from non-significant predictors.

---

## One-shot smoke test

```bash
bash tests/run_real_datasets.sh
```

Runs every skill on its primary dataset end-to-end and fails on any non-zero
exit code.

---

## Recommended testing order in Cursor chat

Test the *skill-selection layer* — does the agent pick the right skill, invoke
the CLI correctly, and verbalise only computed facts?

1. **Baseline clean** — "Run a regression of `y` on `x1` and `x2` in
   `examples/data/clean.csv`, then check assumptions." Expect: both skills
   fire, verdict `clean`, no fabricated numbers.

2. **Categoricals — tips** — "Fit a regression of tip on the other variables
   in `tests/data/tips.csv`." Expect: `linear-regression` fires without being
   named; dummies cite reference category.

3. **Diagnostics trigger** — "Can I trust this model?" (same conversation).
   Expect: `diagnostics` fires using the existing `report.json`.

4. **Remediation loop — engel** — "Model food expenditure from income in
   `tests/data/engel.csv`, check assumptions, fix any problems." Expect:
   fit → diagnostics → HETERO fail → refit with `--robust-se HC3`.

5. **Pre-analysis → regularized — diabetes** — "Explore the diabetes dataset
   before fitting: target `progression`, features `age,sex,bmi,bp,s1–s6`."
   Expect: `pre-analysis` fires first (HIGH_VIF), agent follows with Lasso.

6. **Logistic + AMEs — affairs** — "Model whether someone had an affair using
   `tests/data/affairs.csv`, target `had_affair`." Expect: `logistic-regression`
   fires; agent frames results as probability-scale AMEs, not raw log-odds.

7. **Influence analysis — statecrime** — "Which states are outliers?" after
   fitting violent crime. Expect: `diagnostics` names DC, Hawaii, Mississippi.

8. **Nonlinearity — gapminder** — "Is a linear fit of life expectancy on GDP
   OK?" Expect: RESET fail surfaced; log-transform of `gdpPercap` suggested.

9. **Autocorrelation — macro** — "Regress real consumption on real GDP and
   unemployment. Is the model sound?" Expect: DW failure leads despite R² ≈ 0.999.

10. **Model comparison — diabetes** — "Compare OLS, Lasso, and Ridge on the
    diabetes dataset." Expect: `model-comparison` fires; Akaike weights
    explained; verdict in plain English.

11. **Negative trigger** — "Summarise `tests/data/tips.csv`." Expect: no
    skill fires.
