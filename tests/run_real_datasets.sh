#!/usr/bin/env bash
# Smoke test: run all skills on their primary real datasets.
# Fails loudly on the first non-zero exit code.
set -euo pipefail

OUT="/tmp/regpack_real"
echo "Output directory: $OUT"
echo ""

echo "--- Generating datasets ---"
uv run python tests/make_datasets.py

# ─── linear-regression ────────────────────────────────────────────────────────
echo ""
echo "=== linear-regression ==="

echo "  tips (categoricals)"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/tips.csv --target tip \
    --features total_bill,size,sex,smoker,day,time \
    --output "$OUT/tips"

echo "  engel (simple, textbook)"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/engel.csv --target foodexp \
    --features income \
    --output "$OUT/engel"

echo "  diabetes (10 features)"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/diabetes.csv --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --output "$OUT/diabetes_ols"

echo "  statecrime (small-n)"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/statecrime.csv --target violent \
    --features hs_grad,poverty,single,white,urban \
    --output "$OUT/crime"

echo "  macro (time series)"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/macro.csv --target realcons \
    --features realgdp,unemp \
    --output "$OUT/macro"

# ─── diagnostics ──────────────────────────────────────────────────────────────
echo ""
echo "=== diagnostics ==="

echo "  tips → expect HETEROSCEDASTICITY"
uv run python .agents/skills/diagnostics/scripts/diagnose.py \
    --fit-report "$OUT/tips/report.json" \
    --data tests/data/tips.csv \
    --output "$OUT/tips"

echo "  statecrime → expect HIGH_COOKS_D (DC, Hawaii, Mississippi)"
uv run python .agents/skills/diagnostics/scripts/diagnose.py \
    --fit-report "$OUT/crime/report.json" \
    --data tests/data/statecrime.csv \
    --output "$OUT/crime"

echo "  macro → expect AUTOCORRELATION despite R² ≈ 0.999"
uv run python .agents/skills/diagnostics/scripts/diagnose.py \
    --fit-report "$OUT/macro/report.json" \
    --data tests/data/macro.csv \
    --output "$OUT/macro"

# ─── pre-analysis ─────────────────────────────────────────────────────────────
echo ""
echo "=== pre-analysis ==="

echo "  diabetes → expect HIGH_VIF, recommended: regularized_regression"
uv run python .agents/skills/pre-analysis/scripts/audit.py \
    --data tests/data/diabetes.csv --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --output "$OUT/diabetes_pre"

echo "  tips → expect flags, box_cox recommendation"
uv run python .agents/skills/pre-analysis/scripts/audit.py \
    --data tests/data/tips.csv --target tip \
    --features total_bill,size,sex,smoker,day,time \
    --output "$OUT/tips_pre"

# ─── regularized-regression ───────────────────────────────────────────────────
echo ""
echo "=== regularized-regression ==="

echo "  diabetes Lasso 1-SE → expect ~5 features retained"
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data tests/data/diabetes.csv --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --method lasso --alpha-rule 1se \
    --output "$OUT/diabetes_lasso"

echo "  diabetes Ridge → expect all features retained"
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data tests/data/diabetes.csv --target progression \
    --features age,sex,bmi,bp,s1,s2,s3,s4,s5,s6 \
    --method ridge \
    --output "$OUT/diabetes_ridge"

# ─── logistic-regression ──────────────────────────────────────────────────────
echo ""
echo "=== logistic-regression ==="

echo "  affairs → expect AUC ≈ 0.74, rate_marriage AME negative"
uv run python .agents/skills/logistic-regression/scripts/fit.py \
    --data tests/data/affairs.csv --target had_affair --positive-class 1 \
    --features rate_marriage,age,yrs_married,children,religious,educ,occupation \
    --output "$OUT/affairs" \
    --dataset-name "Fair (1978) affairs"

echo "  breast_cancer → expect AUC ≈ 0.99, CONVERGENCE_ISSUE (near-separation)"
uv run python .agents/skills/logistic-regression/scripts/fit.py \
    --data tests/data/breast_cancer.csv --target malignant --positive-class 1 \
    --features mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension \
    --output "$OUT/bc_logistic" \
    --dataset-name "Breast cancer"

# ─── model-comparison ─────────────────────────────────────────────────────────
echo ""
echo "=== model-comparison ==="

echo "  diabetes OLS vs Lasso vs Ridge → expect complementary_strengths"
uv run python .agents/skills/model-comparison/scripts/compare.py \
    --reports \
        "$OUT/diabetes_ols/report.json" \
        "$OUT/diabetes_lasso/report.json" \
        "$OUT/diabetes_ridge/report.json" \
    --names "OLS" "Lasso (1-SE)" "Ridge" \
    --output "$OUT/diabetes_comparison" \
    --dataset-name "Diabetes"

echo "  tips nested models → bill vs bill+size; expect clear_winner for bill+size"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/tips.csv --target tip \
    --features total_bill \
    --output "$OUT/tips_simple"

uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data tests/data/tips.csv --target tip \
    --features total_bill,size \
    --output "$OUT/tips_size"

uv run python .agents/skills/model-comparison/scripts/compare.py \
    --reports "$OUT/tips_simple/report.json" "$OUT/tips_size/report.json" \
    --names "OLS (bill)" "OLS (bill+size)" \
    --output "$OUT/tips_comparison" \
    --dataset-name "Tips"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== All smoke tests passed ==="
echo "Reports written to $OUT/"
echo ""
echo "Key outputs:"
echo "  $OUT/tips/report.html           (linear + diagnostics)"
echo "  $OUT/diabetes_pre/pre_analysis.html  (pre-analysis)"
echo "  $OUT/diabetes_lasso/report.html  (Lasso)"
echo "  $OUT/affairs/report.html         (logistic)"
echo "  $OUT/bc_logistic/report.html     (logistic, near-separation)"
echo "  $OUT/diabetes_comparison/report.html  (model-comparison)"
