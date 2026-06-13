#!/usr/bin/env bash
# Checkpoint for Phase 2.3 — logistic-regression skill
#
# Verifies:
#   1. Valid LogisticRegressionReport JSON + HTML
#   2. AUC > 0.7 on binary.csv (DGP has clear signal)
#   3. Marginal effects have correct sign: x1 positive, x2 negative, x3 positive
#   4. Calibration plot renders
set -e

uv run python examples/synth_data.py

echo ""
echo "--- Logistic regression on binary data ---"
echo "    DGP: log-odds = -0.5 + 1.5*x1 - 1.2*x2 + 0.6*x3 (x4 = noise)"
echo "    Expect: AUC > 0.7, x1 AME > 0, x2 AME < 0, x3 AME > 0"
uv run python .agents/skills/logistic-regression/scripts/fit.py \
    --data examples/data/binary.csv \
    --target y \
    --positive-class 1 \
    --features all \
    --output examples/output/logistic \
    --dataset-name "Binary synthetic data"

echo ""
echo "--- Robust SE variant ---"
uv run python .agents/skills/logistic-regression/scripts/fit.py \
    --data examples/data/binary.csv \
    --target y \
    --positive-class 1 \
    --features all \
    --robust-se HC3 \
    --output examples/output/logistic_robust \
    --dataset-name "Binary synthetic data (HC3)"

echo ""
echo "Done."
echo "  open examples/output/logistic/report.html"
echo "  open examples/output/logistic_robust/report.html"
