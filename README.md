<p align="center">
  <img src="docs/banner.svg" alt="regkit" width="280">
</p>
<p align="center"><strong>Statistically valid regression for language models.</strong></p>
<p align="center">Python computes. Claude narrates. Every number is traceable to a statsmodels, scikit-learn, or scipy call.</p>

<p align="center">
  <a href="evals/"><img src="https://img.shields.io/badge/evals-33%20passing-0F766E?style=flat-square" alt="evals"></a>
  <a href=".agents/skills/"><img src="https://img.shields.io/badge/skills-6-0F766E?style=flat-square" alt="skills"></a>
  <img src="https://img.shields.io/badge/schemas-validated-0F766E?style=flat-square" alt="schemas">
  <img src="https://img.shields.io/badge/python-3.13-0F766E?style=flat-square" alt="python">
  <img src="https://img.shields.io/badge/lint-ruff-6B7280?style=flat-square" alt="lint">
  <img src="https://img.shields.io/badge/datasets-9%20real--world-6B7280?style=flat-square" alt="datasets">
</p>

<p align="center">
  <img src="demo/regkit.gif" alt="regkit running Lasso, two linear fits, and a model comparison in one terminal session" width="900">
</p>
<p align="center"><em>Lasso feature selection, two linear fits, and a model comparison — 25 seconds, deterministic Python under the hood, ready to narrate in chat.</em></p>

<!-- TODO: add docs/screenshots/hero_macro_diagnostics.png — screenshot of examples/output/macro/diagnostics.html at 1100px width -->

## Why regkit

**Without regkit** — ask an LLM a regression:

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
# coefficients only. no p-values. no diagnostics. no transform awareness.
# the LLM narrates from these and confidently misinterprets.
```

**With regkit:**
The model explains 45.2% of variance in tip (adj-R² = 0.452, n=244).
Only total_bill is significant (β = 0.105, p < 0.001).
Heteroscedasticity is likely — run diagnostics before trusting inference.

> Most LLM statistics tools let the model do the math. This one doesn't.
> `regkit` constrains Claude to the judgment layer — choosing transforms,
> framing coefficients for the audience, surfacing the one diagnostic issue that matters most.
> It never does arithmetic and never reports a p-value it didn't compute.

## Design principles

1. **Facts, not prose.** Scripts emit `InterpretationFact` objects — canonical claims with confidence grades and caveats. Claude turns the same structured facts into different prose for different audiences without changing the underlying numbers.

2. **Schemas are the contract.** Every cross-skill payload round-trips through a Pydantic model in `regression_pack_core/schemas.py`. Skills chain by reading each other's JSON — the model never passes numbers in prose between turns.

3. **Deterministic and reproducible.** Same data → same numbers, always. Synthetic datasets are seeded; the diagnostics refit verifies it matches the original fit to 1e-6 before trusting any residual.

4. **One visual language.** All plots go through `regression_pack_core.plotting`, all reports through the shared Jinja template and CSS design tokens. Every HTML deliverable is self-contained — Plotly JS inlined, no CDN.

5. **Scales to large data.** Diagnostics runs in O(n) — studentized residuals use the closed-form leave-one-out identity rather than O(n²) refitting. Influence lists are capped at the top 50 by severity; HTML tables at 20 rows; scatter plots at 2,000 points (extremes always kept). True counts are always in the JSON so nothing is silently truncated.

## Architecture

```mermaid
flowchart LR
    A[User in chat] -->|natural language| B[Claude]
    B -->|reads SKILL.md| C[regkit skill]
    C -->|deterministic compute| D[statsmodels / sklearn / scipy]
    D -->|validated JSON| E[report.json]
    E -->|interpretation facts| B
    E -->|render| F[report.html]
    B -->|narration| A

    classDef accent fill:#0F766E,color:#fff,stroke:#0F766E,stroke-width:1px
    class B,C,D,E accent
```

## Skills at a glance

| Skill | What Python computes | Output schema |
|-------|----------------------|---------------|
| [`pre-analysis`](.agents/skills/pre-analysis/SKILL.md) | VIF, outlier scan, target distribution, skill recommendation | `PreAnalysisReport` |
| [`linear-regression`](.agents/skills/linear-regression/SKILL.md) | OLS coefficients, CIs, robust SE, R², AIC/BIC, interpretation facts | `LinearRegressionReport` |
| [`diagnostics`](.agents/skills/diagnostics/SKILL.md) | Breusch-Pagan, RESET, Durbin-Watson, Cook's D, leverage, CV bias/variance | `DiagnosticsReport` |
| [`regularized-regression`](.agents/skills/regularized-regression/SKILL.md) | Ridge / Lasso / ElasticNet, CV alpha, regularisation path, 1-SE rule | `RegularizedRegressionReport` |
| [`logistic-regression`](.agents/skills/logistic-regression/SKILL.md) | MLE, odds ratios, average marginal effects, ROC/AUC, calibration | `LogisticRegressionReport` |
| [`model-comparison`](.agents/skills/model-comparison/SKILL.md) | Akaike weights, LR tests for nested pairs, coefficient comparison, verdict | `ModelComparisonReport` |

Skills chain: every `report.json` is the next skill's input.

## Quick start

```bash
git clone <repo>  &&  cd regkit
uv sync
uv run python tests/make_datasets.py    # writes 9 real-world CSVs to tests/data/
uv run python examples/synth_data.py   # writes synthetic CSVs to examples/data/
bash tests/run_real_datasets.sh        # runs all 6 skills end-to-end, 0 failures
```

**Open the repo in Cursor and try:**
```
Explore tests/data/diabetes.csv before fitting — target is progression, features are all.
```

## Using in Cursor chat

Add the skills to your Cursor project. Claude discovers them from `SKILL.md` files in `.agents/skills/`. You never need to name a skill — just describe what you want.

### Pre-analysis — *"Is my data ready to model?"*

Ask this **before** fitting anything. The skill audits target shape, VIF, outliers, and tells you which modelling approach to use.

**Prompts that trigger it:**
```
Explore the diabetes dataset before fitting — target is progression, features are age,sex,bmi,bp,s1,s2,s3,s4,s5,s6.
```
```
Is there anything I should know about this data before I run a regression?
```
```
Check for multicollinearity and outliers in my CSV before I fit anything.
```

**What Claude does:**
Calls `audit.py` → reads `pre_analysis.json` → surfaces the most important flag first. If VIF > 10, it tells you which features are collinear and recommends Lasso. If the target is skewed, it suggests a log or Box-Cox transform before you waste a fit.

<details>
<summary>Show example output</summary>

```
Pre-analysis flagged HIGH_VIF on s1–s6 (collinear serum measures; VIF up to 18.4).
Recommended estimator: regularized-regression.
Target (progression) is approximately normal — no transform needed.
```

</details>

### Linear regression — *"Fit a regression"*

**Prompts that trigger it:**
```
Fit a regression of tip on total_bill, size, sex, smoker, day, and time using tests/data/tips.csv.
```
```
Run OLS on this CSV — target is price, features are sqft, bedrooms, bathrooms, neighborhood.
```
```
What's the relationship between income and food expenditure in tests/data/engel.csv?
```
```
Model life expectancy from GDP per capita in the gapminder data. Is a linear fit appropriate?
```

**What Claude does:**
Calls `fit.py` → reads `report.json` → leads with the headline (adj-R², n significant predictors). For categorical features, it names the reference category. For log-transformed fits, it back-translates coefficients to percent changes. It does not narrate every coefficient — it surfaces the ones that matter for the question asked.

**Key options you can request in plain English:**
- *"Use robust standard errors"* → `--robust-se HC3`
- *"Log-transform the target"* → `--log-target`
- *"Log-transform GDP but not population"* → `--log-features gdpPercap`
- *"Standardise the features"* → `--standardize`

<details>
<summary>Show example output</summary>

```
The model explains 45.2% of variance in tip (adj-R² = 0.452, n=244).
Only total_bill is significant (β = 0.105, p < 0.001); party size, sex, smoker,
day, and time add no reliable signal beyond bill size.
Heteroscedasticity is likely — run diagnostics before trusting inference.
```

</details>

### Diagnostics — *"Can I trust this model?"*

Best used as a **follow-up** after a linear fit. Claude will reuse the `report.json` from the previous turn automatically.

**Prompts that trigger it:**
```
Can I trust this model?
```
```
Check the assumptions on that fit.
```
```
Are there any influential observations I should know about?
```
```
Is this model overfitting or underfitting?
```

**What Claude does:**
Calls `diagnose.py` with the existing `report.json` → reads `diagnostics.json` → leads with the **verdict** (`clean` / `problematic` / `unreliable`), then explains only the failing or warning checks. It does not list all five assumptions if they all pass — it says *"all assumptions satisfied"* and moves on.

**Verdict meanings:**
| Verdict | Meaning |
|---------|---------|
| `clean` | All checks pass; results can be trusted |
| `problematic` | ≥1 check fails; inference may be misleading |
| `unreliable` | Multiple critical failures; remediate before using any estimates |

**Common follow-ups Claude handles:**
- *Heteroscedasticity detected* → suggests `--robust-se HC3` refit
- *Nonlinearity (RESET fail)* → suggests `--log-features` or `--log-target`
- *High Cook's D* → names the influential rows by index (or by label if a name column exists)
- *Autocorrelation* → warns about spurious regression, suggests GLS or differencing
- *High VIF* → recommends `regularized-regression` skill

<details>
<summary>Show example output</summary>

```
Verdict: unreliable.
Despite R² = 0.999, three assumptions fail:
   Autocorrelation (DW = 0.18 ≪ 1.5) — regression errors are serially correlated.
   Homoscedasticity (BP p < 0.001) — residual variance is not constant.
   Normality (Jarque-Bera p < 0.001) — errors are non-normal.
The near-perfect R² is a symptom of spurious regression on integrated time series,
not a sign of a good model. Consider first-differencing or a cointegration approach.
```

</details>

### Regularized regression — *"Handle collinear features"* or *"Do feature selection"*

**Prompts that trigger it:**
```
Run Lasso on the diabetes dataset — target progression, all 10 features.
Use the 1-SE rule to get a parsimonious model.
```
```
My features are highly collinear. Use Ridge regression on this CSV.
```
```
Which features actually matter? Run Lasso and tell me what gets dropped.
```
```
Compare Ridge vs Lasso vs ElasticNet on this dataset.
```

**What Claude does:**
Calls `fit.py` with `--method lasso/ridge/elasticnet` → reads the report → leads with which features survived (for Lasso/ElasticNet) or how much shrinkage occurred (Ridge). It explains *why* certain features were dropped using the CV curve and regularisation path. For the OLS comparison section, it notes which coefficients changed sign or magnitude due to shrinkage.

**Key options you can request:**
- *"Use the 1-SE rule"* → `--alpha-rule 1se` (more parsimonious than CV minimum)
- *"Use ElasticNet with mostly L1"* → `--method elasticnet --l1-ratio 0.8`

<details>
<summary>Show example output</summary>

```
Lasso (1-SE rule, α = 5.83) retained 5 of 10 features:
   bmi, bp, s3, s5  — kept (strong signal)
   age, s1, s2, s4, s6 — zeroed (redundant or weak)

adj-R² = 0.470, down from OLS adj-R² = 0.507.
The 5-feature model sacrifices 3.7 pp of explained variance to drop 5 correlated
serum measures — a reasonable trade for interpretability and stability.
```

</details>

### Logistic regression — *"Binary outcome"*

**Prompts that trigger it:**
```
Model whether someone had an affair using tests/data/affairs.csv, target had_affair.
```
```
Predict malignancy from the breast cancer features. Target is malignant, positive class is 1.
```
```
Run a logistic regression of loan_default on credit_score, income, and debt_ratio.
```
```
What's the probability that someone churns based on these features?
```

**What Claude does:**
Calls `fit.py` → reads `report.json` → leads with AUC, then narrates **average marginal effects** (not raw log-odds) so the interpretation is on the probability scale. It always reports: *"A one-unit increase in X is associated with a Y pp change in P(outcome)"*. Odds ratios appear as a secondary table.

**Key options:**
- *"Which class is positive?"* → `--positive-class 1` (or the label name)
- *"Use robust standard errors"* → `--robust-se HC3`
- *"Use 0.3 as the decision threshold"* → `--threshold 0.3`

**What the flags mean:**
| Flag | Trigger | What Claude says |
|------|---------|-----------------|
| `CLASS_IMBALANCE` | Positive rate < 20% or > 80% | Accuracy misleads; focus on AUC and precision-recall |
| `CONVERGENCE_ISSUE` | `max\|β\| > 10` or not converged | Near-perfect separation; coefficients inflated; consider Firth regression |
| `LOW_AUC` | AUC < 0.65 | Model barely discriminates; check feature relevance |

<details>
<summary>Show example output</summary>

```
AUC = 0.744  (n=6,366, 32.2% positive)
6 of 7 predictors significant at p < 0.05.

Average marginal effects (probability scale):
  rate_marriage  −0.054  ***   Happier marriages → less likely to have affair
  yrs_married    +0.016  ***   Longer marriages → more likely
  religious      −0.030  ***   More religious → less likely
  age            −0.006  *
  educ, occupation  not significant
```

</details>

### Model comparison — *"Which model is best?"*

**Prompts that trigger it:**
```
Compare the OLS, Lasso, and Ridge fits on the diabetes dataset.
```
```
Which model should I use — the one with all features or just total_bill and size?
```
```
I have three report.json files. Which model wins on AIC?
```

**What Claude does:**
Calls `compare.py` with the report paths → reads `report.json` → narrates the **verdict** and the key evidence (Akaike weights, LR test p-value if nested). It explains *why* the winner was chosen, not just *which* model won.

**Verdict types:**
| Verdict | Meaning |
|---------|---------|
| `clear_winner` | One model has Akaike weight ≥ 0.80 or LR test significant |
| `competitive_tie` | Max Δ AIC < 2; models are statistically indistinguishable |
| `complementary_strengths` | Different families or close metrics; choose by goal |
| `all_inadequate` | All primary fit metrics below 0.15; reconsider the approach |

**Cross-family comparisons:**
AIC is only valid within the same family and outcome. If you mix OLS and logistic, Claude will note this and rank by primary fit metric (adj-R² or pseudo-R²) instead.

<details>
<summary>Show example output</summary>

```
Verdict: complementary_strengths.  Recommended: OLS (marginal AIC advantage).

Akaike weights:  OLS = 0.501  ·  Ridge = 0.499  ·  Lasso (1-SE) = 0.000

OLS and Ridge are statistically tied on AIC — Ridge shrinks coefficients but barely
changes residuals at the CV-optimal α on n=442. Lasso pays a meaningful fit cost
(Δ AIC = 12.4) to drop 5 features.

Choose Ridge or Lasso if interpretability or stability under collinearity matters
more than log-likelihood. Choose OLS if you want the maximum-likelihood estimate
and are comfortable with all 10 features in the model.
```

</details>

## Multi-turn workflows

### Workflow A: full regression pipeline

```
Turn 1: "Explore tests/data/diabetes.csv before fitting — target progression, features all."
→ pre-analysis fires → HIGH_VIF flagged → Claude recommends Lasso

Turn 2: "OK, run Lasso with the 1-SE rule."
→ regularized-regression fires → 5 features retained → report written

Turn 3: "Can I trust this fit?"
→ diagnostics fires on the Lasso report → assumptions checked

Turn 4: "Compare it to plain OLS."
→ linear-regression fires for OLS → model-comparison fires → verdict explained
```

### Workflow B: binary classification

```
Turn 1: "Model affair probability from tests/data/affairs.csv, target had_affair."
→ logistic-regression fires → AUC = 0.744, AMEs reported

Turn 2: "Use robust standard errors instead."
→ logistic-regression refits with HC3 → SEs change, coefficients identical

Turn 3: "Which predictors actually matter? Run without educ and occupation."
→ logistic-regression refits → model-comparison fires automatically → LR test
```

### Workflow C: remediation loop

```
Turn 1: "Fit food expenditure on income in tests/data/engel.csv."
→ linear-regression fires

Turn 2: "Check the assumptions."
→ diagnostics fires → HETEROSCEDASTICITY fail → use_robust_se recommendation

Turn 3: "Fix it."
→ Claude refits with --robust-se HC3 → shows what changed (SEs widen, p-values shift)
```

## Test datasets

Nine real-world datasets in `tests/data/`, generated offline from packages in the dependency graph:

```bash
uv run python tests/make_datasets.py
```

| Dataset | n | Target | Best for |
|---------|---|--------|----------|
| `tips.csv` | 244 | `tip` (continuous) | Categoricals, heteroscedasticity |
| `engel.csv` | 235 | `foodexp` (continuous) | Textbook heteroscedasticity |
| `gapminder_2007.csv` | 142 | `lifeExp` (continuous) | Nonlinearity, log transform |
| `statecrime.csv` | 51 | `violent` (continuous) | Influence analysis, named outliers |
| `macro.csv` | 203 | `realcons` (continuous) | Autocorrelation, spurious R² |
| `diabetes.csv` | 442 | `progression` (continuous) | Collinearity, Lasso, pre-analysis |
| `breast_cancer.csv` | 569 | `malignant` (binary, 1=cancer) | Logistic, near-separation |
| `affairs.csv` | 6,366 | `had_affair` (binary) | Logistic, interpretable AMEs |
| `longley.csv` | 16 | `TOTEMP` (continuous) | Severe collinearity (theory only; n too small for CLI) |

See [`tests/README.md`](tests/README.md) for exact CLI commands, expected outputs, and the 11-step Cursor chat testing order.

## Repository layout

```
regression_pack_core/       shared library
  schemas.py                all Pydantic output schemas
  plotting.py               Plotly theme + chart helpers
  reports.py                HTML template + section builders
  validators.py             input validation + feature coercion

.agents/skills/
  pre-analysis/             SKILL.md · scripts/ · references/
  linear-regression/
  diagnostics/
  regularized-regression/
  logistic-regression/
  model-comparison/

examples/
  synth_data.py             seeded synthetic datasets
  run_*.sh                  one checkpoint script per skill

tests/
  make_datasets.py          9 real-world CSVs (offline, reproducible)
  run_real_datasets.sh      full smoke test across all skills
  data/                     generated CSVs
  README.md                 per-skill guide + Cursor chat testing order

evals/
  correctness/              Anscombe quartet, DGP recovery, OR consistency
  detection/                planted-violation end-to-end tests (pytest)
  triggering/               prompt-level skill-selection eval plan
```

## Development

```bash
uv run pytest evals/ -q      # 33 correctness + detection evals
uv run ruff check .           # lint (zero errors expected)
bash tests/run_real_datasets.sh  # all 6 skills on 9 real datasets
```

The **detection evals** assert the pack's headline guarantees end-to-end:
clean data → `clean` verdict with zero flags; planted heteroscedasticity → Breusch-Pagan fail + `use_robust_se` recommendation; planted high-leverage outlier → Cook's D > 5× the 4/n threshold; logistic AUC > 0.70 on the planted DGP; model-comparison selects the fuller model when the LR test is significant.

The **correctness evals** verify: Anscombe's quartet diagnostics come back different despite identical regression lines; logistic AME signs match the planted DGP; `exp(log-odds) == OR` to float precision; Ridge inverse-standardisation RMSE < 1.0.
