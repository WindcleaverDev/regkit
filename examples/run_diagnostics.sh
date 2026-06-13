#!/usr/bin/env bash
set -e
# Assumes run_linear_regression.sh already ran
uv run python .agents/skills/diagnostics/scripts/diagnose.py \
    --fit-report examples/output/clean/report.json \
    --data examples/data/clean.csv \
    --output examples/output/clean
echo "Done. Open examples/output/clean/diagnostics.html"
