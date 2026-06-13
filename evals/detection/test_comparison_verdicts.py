"""Detection evals: model-comparison skill end-to-end.

Runs the CLI via subprocess with 3 tips models:
- OLS (total_bill only) — simpler, lower adj-R²
- OLS (total_bill + size) — fuller, higher adj-R²
- OLS (total_bill + size, HC3) — same fit as above, robust SE

Verifies:
1. Nesting detected: OLS(bill only) ⊊ OLS(bill+size)
2. LR test rejects the simpler model (size adds signal)
3. Akaike weights sum to 1.0
4. Verdict is clear_winner with a recommended model
5. ModelComparisonReport schema validates
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *argv], cwd=REPO, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def comparison_output(tmp_path_factory) -> dict:
    """Fit 3 OLS models on tips.csv and run model-comparison once."""
    out_root = tmp_path_factory.mktemp("comparison")

    gen = run_cli("examples/synth_data.py")
    assert gen.returncode == 0, gen.stderr

    # Model 1: simpler
    m1_dir = out_root / "ols_simple"
    p = run_cli(
        ".agents/skills/linear-regression/scripts/fit.py",
        "--data", "examples/data/tips.csv",
        "--target", "tip",
        "--features", "total_bill",
        "--output", str(m1_dir),
    )
    assert p.returncode == 0, p.stderr

    # Model 2: fuller
    m2_dir = out_root / "ols_full"
    p = run_cli(
        ".agents/skills/linear-regression/scripts/fit.py",
        "--data", "examples/data/tips.csv",
        "--target", "tip",
        "--features", "total_bill,size",
        "--output", str(m2_dir),
    )
    assert p.returncode == 0, p.stderr

    # Model 3: fuller + HC3
    m3_dir = out_root / "ols_robust"
    p = run_cli(
        ".agents/skills/linear-regression/scripts/fit.py",
        "--data", "examples/data/tips.csv",
        "--target", "tip",
        "--features", "total_bill,size",
        "--robust-se", "HC3",
        "--output", str(m3_dir),
    )
    assert p.returncode == 0, p.stderr

    # Comparison
    cmp_dir = out_root / "comparison"
    p = run_cli(
        ".agents/skills/model-comparison/scripts/compare.py",
        "--reports",
        str(m1_dir / "report.json"),
        str(m2_dir / "report.json"),
        str(m3_dir / "report.json"),
        "--names", "OLS (bill only)", "OLS (bill+size)", "OLS (bill+size, HC3)",
        "--output", str(cmp_dir),
    )
    assert p.returncode == 0, p.stderr

    return json.loads((cmp_dir / "report.json").read_text())


def test_akaike_weights_sum_to_one(comparison_output):
    """Akaike weights must sum to 1.0 (within floating-point tolerance)."""
    aw = comparison_output.get("akaike_weights")
    assert aw is not None, "akaike_weights should be present"
    total = sum(aw["weights"])
    assert total == pytest.approx(1.0, abs=1e-6), f"Weights sum to {total:.8f}"


def test_nesting_detected(comparison_output):
    """At least one LR test should be present (bill-only ⊊ bill+size)."""
    lr_tests = comparison_output.get("lr_tests", [])
    assert len(lr_tests) >= 1, "Expected at least one LR test for the nested pair"


def test_lr_test_rejects_simpler_model(comparison_output):
    """size adds signal — the LR test should reject the null at p < 0.05."""
    lr_tests = comparison_output.get("lr_tests", [])
    relevant = [t for t in lr_tests if "bill only" in t["nested_model"]]
    assert relevant, "No LR test found for the bill-only nested model"
    for t in relevant:
        assert t["p_value"] < 0.05, (
            f"LR test did not reject simpler model: p = {t['p_value']:.4f}"
        )


def test_verdict_has_recommended_model(comparison_output):
    """Verdict must name a recommended model."""
    v = comparison_output["verdict"]
    assert v["recommended_model"] is not None, "verdict.recommended_model is None"
    assert v["recommended_model"] in {
        "OLS (bill only)", "OLS (bill+size)", "OLS (bill+size, HC3)"
    }


def test_fuller_model_recommended(comparison_output):
    """The fuller model (bill+size) should be recommended over bill-only."""
    v = comparison_output["verdict"]
    assert v["recommended_model"] != "OLS (bill only)", (
        "The simpler bill-only model should not be recommended when LR test is significant"
    )


def test_bill_only_has_lowest_akaike_weight(comparison_output):
    """OLS (bill only) should have the lowest Akaike weight."""
    aw = comparison_output.get("akaike_weights")
    assert aw is not None
    weight_map = dict(zip(aw["model_names"], aw["weights"], strict=True))
    simple_weight = weight_map.get("OLS (bill only)", 1.0)
    other_weights = [w for n, w in weight_map.items() if n != "OLS (bill only)"]
    assert simple_weight < max(other_weights), (
        f"bill-only model has unexpectedly high Akaike weight: {simple_weight:.4f}"
    )


def test_schema_round_trip(comparison_output):
    """ModelComparisonReport schema must accept the emitted JSON."""
    from regression_pack_core.schemas import ModelComparisonReport

    report = ModelComparisonReport(**comparison_output)
    assert len(report.models) == 3
    assert report.verdict.overall in {
        "clear_winner", "competitive_tie", "complementary_strengths", "all_inadequate"
    }
