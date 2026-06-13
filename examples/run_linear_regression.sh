#!/usr/bin/env bash
set -e
uv run python examples/synth_data.py
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data examples/data/clean.csv \
    --target y \
    --features x1,x2 \
    --output examples/output/clean \
    --dataset-name "Clean synthetic data"
echo "Done. Open examples/output/clean/report.html"
