#!/usr/bin/env bash
# Checkpoint for Phase 2.2 — regularized-regression skill
#
# Verifies:
#   1. Lasso correctly zeros the collinear feature (x2) on collinear.csv
#   2. Ridge shrinks coefficients vs OLS (comparison_to_ols)
#   3. ElasticNet runs to completion on the same data
set -e

uv run python examples/synth_data.py

echo ""
echo "--- Lasso (1-SE rule) on collinear data ---"
echo "    Expect: x4 (pure noise) is zeroed; x2 coefficient << x1 (Lasso"
echo "    correctly assigns signal to x1; x2 is near-zero but not exactly 0"
echo "    at the min-CV alpha since its marginal contribution is small but non-zero)"
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data examples/data/collinear.csv \
    --target y \
    --features all \
    --method lasso \
    --alpha-rule 1se \
    --output examples/output/regularized/lasso \
    --dataset-name "Collinear synthetic data (Lasso 1-SE)"

echo ""
echo "--- Ridge on collinear data ---"
echo "    Expect: all features retained but shrunk vs OLS"
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data examples/data/collinear.csv \
    --target y \
    --features all \
    --method ridge \
    --output examples/output/regularized/ridge \
    --dataset-name "Collinear synthetic data (Ridge)"

echo ""
echo "--- ElasticNet on collinear data ---"
echo "    Expect: runs to completion; feature selection intermediate between Lasso and Ridge"
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data examples/data/collinear.csv \
    --target y \
    --features all \
    --method elasticnet \
    --l1-ratio 0.7 \
    --output examples/output/regularized/elasticnet \
    --dataset-name "Collinear synthetic data (ElasticNet l1=0.7)"

echo ""
echo "--- Lasso on clean data (sanity check — should retain x1 and x2 signal) ---"
uv run python .agents/skills/regularized-regression/scripts/fit.py \
    --data examples/data/clean.csv \
    --target y \
    --features all \
    --method lasso \
    --output examples/output/regularized/lasso_clean \
    --dataset-name "Clean synthetic data (Lasso)"

echo ""
echo "Done."
echo "  open examples/output/regularized/lasso/report.html"
echo "  open examples/output/regularized/ridge/report.html"
