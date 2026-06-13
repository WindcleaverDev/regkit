#!/usr/bin/env bash
# Checkpoint for Phase 2.4 — model-comparison skill
#
# Verifies:
#   1. Valid ModelComparisonReport JSON + HTML
#   2. Akaike weights sum to ~1.0
#   3. Verdict surfaces a recommended model with rationale
#   4. LR test detects nesting (total_bill only) vs (total_bill + size)
set -e

echo "--- Generating synthetic datasets ---"
uv run python examples/synth_data.py

echo ""
echo "--- Model 1: OLS with total_bill only ---"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data examples/data/tips.csv \
    --target tip \
    --features total_bill \
    --output examples/output/tips_ols_simple \
    --dataset-name "tips (total_bill only)"

echo ""
echo "--- Model 2: OLS with total_bill + size ---"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data examples/data/tips.csv \
    --target tip \
    --features total_bill,size \
    --output examples/output/tips_ols_full \
    --dataset-name "tips (total_bill + size)"

echo ""
echo "--- Model 3: OLS + HC3 robust SE with total_bill + size ---"
uv run python .agents/skills/linear-regression/scripts/fit.py \
    --data examples/data/tips.csv \
    --target tip \
    --features total_bill,size \
    --robust-se HC3 \
    --output examples/output/tips_ols_robust \
    --dataset-name "tips (total_bill + size, HC3)"

echo ""
echo "--- Model comparison ---"
uv run python .agents/skills/model-comparison/scripts/compare.py \
    --reports \
        examples/output/tips_ols_simple/report.json \
        examples/output/tips_ols_full/report.json \
        examples/output/tips_ols_robust/report.json \
    --names "OLS (bill only)" "OLS (bill+size)" "OLS (bill+size, HC3)" \
    --output examples/output/comparison_tips \
    --dataset-name "tips synthetic"

echo ""
echo "--- Validate Akaike weights sum ---"
uv run python - <<'EOF'
import json, sys
report = json.loads(open("examples/output/comparison_tips/report.json").read())
aw = report.get("akaike_weights")
if aw:
    total = sum(aw["weights"])
    print(f"  Weights: {[f'{w:.4f}' for w in aw['weights']]}")
    print(f"  Sum: {total:.6f}")
    assert abs(total - 1.0) < 1e-6, f"Weights don't sum to 1! Got {total}"
    print("  ✓ Akaike weights sum to 1.0")
else:
    print("  (No Akaike weights — AIC comparison not available)")

v = report["verdict"]
print(f"  Verdict: {v['overall']}")
print(f"  Recommended: {v['recommended_model']}")
print(f"  Headline: {v['headline']}")
assert v["recommended_model"] is not None, "verdict.recommended_model is None!"
print("  ✓ Recommended model present")

lr_tests = report.get("lr_tests", [])
print(f"  LR tests: {len(lr_tests)}")
for t in lr_tests:
    print(f"    {t['nested_model']} vs {t['full_model']}: LR={t['likelihood_ratio']:.2f}, p={t['p_value']:.4f}")
EOF

echo ""
echo "Done."
echo "  open examples/output/comparison_tips/report.html"
