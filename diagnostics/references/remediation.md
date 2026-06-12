# Remediation playbook

What to do when each flag code is raised. Each section maps to one
`Recommendation.action` emitted by the diagnostics skill.

## HETEROSCEDASTICITY → `use_robust_se`

The coefficients are fine; the standard errors are not.

1. **Robust standard errors** — refit with `--robust-se HC3` (n < 250) or
   `HC1` (larger samples). See `linear-regression/references/robust_se.md`
   for the full decision tree.
2. **Transform the target** — if the target is positive and the variance
   grows with the mean, `--log-target` often removes the heteroscedasticity
   *and* gives more natural percentage interpretations. sqrt is the gentler
   alternative for count-like targets.
3. If the funnel reflects a genuinely different data-generating process for
   part of the sample, model the groups separately or use weighted least
   squares.

## MISSED_NONLINEARITY → `add_polynomial_term`

The mean function is misspecified — coefficients are biased, not just noisy.

1. Look at the partial residual plots to identify *which* feature bends.
2. Add a squared term for that feature and refit; the polynomial_term
   interpretation in the fit report handles the phrasing.
3. For monotone-but-curved relationships, a log transform of the feature is
   more parsimonious than a polynomial.
4. Re-run diagnostics: RESET should clear.

## HIGH_VIF → `drop_collinear_feature`

Coefficients are individually unstable (huge SEs, sign flips) even though
joint fit is fine.

1. Check the per-feature VIFs in the assumption detail; the offenders cluster.
2. Drop one feature from each collinear cluster — keep the one with the
   clearer interpretation.
3. Alternatively switch to ridge regression, which trades a little bias for a
   lot of variance reduction (regularized-regression skill, phase 2).
4. If two features are near-duplicates by construction (e.g. sqft and rooms),
   combine them into one.

## NON_NORMAL_RESIDUALS → `transform_target`

1. Skewed residuals: log (positive targets) or Box-Cox transform of the target.
2. Heavy tails: consider robust regression; inspect whether the tails are
   data errors.
3. If n is large (hundreds+), the CLT protects coefficient inference — note
   the caveat and move on; only prediction intervals stay unreliable.

## AUTOCORRELATION → `use_time_series_model`

Only meaningful when rows have a natural order (time, space, sequence).

1. If the data is a time series: use OLS with autoregressive errors
   (e.g. Cochrane-Orcutt / `GLSAR`) or move to an explicit time-series model.
2. Newey-West (HAC) standard errors are the inference patch when the mean
   model is otherwise right.
3. If rows are *not* ordered, the Durbin-Watson statistic is meaningless —
   ignore this flag.

## HIGH_COOKS_D / INFLUENTIAL_POINTS → `inspect_row`

1. Look at the flagged row(s) in the original data. Is it a typo, unit error,
   or merge artifact? Fix the data, refit.
2. If the point is legitimate, refit with and without it and compare
   coefficients. If conclusions flip, report both fits — the data doesn't
   support a single answer.
3. Never silently delete a legitimate observation to make diagnostics pass.

## HIGH_VARIANCE → `regularize`

The model memorises the training data and fails out of sample.

1. Regularise: ridge (keeps all features) or lasso (also selects).
2. Get more data if you can — variance shrinks with n.
3. Simplify: drop weak features, especially high-degree polynomial and
   interaction terms.

## HIGH_BIAS → `add_features`

The model is too simple even for the training data.

1. Add informative features, polynomial terms, or interactions.
2. Reconsider the functional form — maybe the relationship isn't linear in
   any of the current features.
3. Check whether the target is even predictable from this feature set; a low
   ceiling on R² may be a property of the problem, not the model.
