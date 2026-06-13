#!/usr/bin/env bash
set -e

uv run python examples/synth_data.py

echo ""
echo "--- Clean data (expect: no flags, preferred estimator = linear_regression) ---"
uv run python .agents/skills/pre-analysis/scripts/audit.py \
    --data examples/data/clean.csv \
    --target y \
    --features all \
    --output examples/output/pre_analysis/clean \
    --dataset-name "Clean synthetic data"

echo ""
echo "--- Skewed data (expect: TARGET_SKEW flag, log_transform recommended) ---"
uv run python .agents/skills/pre-analysis/scripts/audit.py \
    --data examples/data/skewed.csv \
    --target y \
    --features all \
    --output examples/output/pre_analysis/skewed \
    --dataset-name "Skewed synthetic data (log-normal target)"

echo ""
echo "Done."
echo "  open examples/output/pre_analysis/clean/pre_analysis.html"
echo "  open examples/output/pre_analysis/skewed/pre_analysis.html"
