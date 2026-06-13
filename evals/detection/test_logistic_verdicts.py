"""Detection evals: logistic-regression skill end-to-end.

Runs the CLI via subprocess on binary.csv:
- AUC > 0.70 on the known DGP
- Marginal effect signs correct (x1+, x2-, x3+)
- No spurious convergence flag
- Standard and HC3 variants both produce valid schema
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
def logistic_outputs(tmp_path_factory) -> dict[str, dict]:
    """Generate data and fit logistic (standard + HC3) once for all tests."""
    out_root = tmp_path_factory.mktemp("logistic")

    gen = run_cli("examples/synth_data.py")
    assert gen.returncode == 0, gen.stderr

    results: dict[str, dict] = {}
    for variant, extra in [("standard", []), ("hc3", ["--robust-se", "HC3"])]:
        out_dir = out_root / variant
        proc = run_cli(
            ".agents/skills/logistic-regression/scripts/fit.py",
            "--data", "examples/data/binary.csv",
            "--target", "y",
            "--positive-class", "1",
            "--features", "all",
            "--output", str(out_dir),
            *extra,
        )
        assert proc.returncode == 0, f"logistic fit ({variant}) failed:\n{proc.stderr}"
        results[variant] = json.loads((out_dir / "report.json").read_text())
    return results


def test_auc_above_threshold(logistic_outputs):
    """AUC > 0.70 on the planted DGP."""
    auc = logistic_outputs["standard"]["roc"]["auc"]
    assert auc > 0.70, f"AUC = {auc:.4f} is below 0.70"


def test_ame_signs(logistic_outputs):
    """AME signs: x1+ x2- x3+ (per DGP)."""
    mes = {m["feature"]: m["ame"] for m in logistic_outputs["standard"]["marginal_effects"]}
    assert mes["x1"] > 0, f"x1 AME should be positive; got {mes['x1']:.4f}"
    assert mes["x2"] < 0, f"x2 AME should be negative; got {mes['x2']:.4f}"
    assert mes["x3"] > 0, f"x3 AME should be positive; got {mes['x3']:.4f}"


def test_no_convergence_flag(logistic_outputs):
    """Clean DGP should not trigger CONVERGENCE_ISSUE."""
    flags = [f["code"] for f in logistic_outputs["standard"]["flags"]]
    assert "CONVERGENCE_ISSUE" not in flags, f"Unexpected convergence flag; flags = {flags}"


def test_hc3_variant_same_auc(logistic_outputs):
    """HC3 variant changes SEs but not log-likelihood; AUC must be identical."""
    auc_std = logistic_outputs["standard"]["roc"]["auc"]
    auc_hc3 = logistic_outputs["hc3"]["roc"]["auc"]
    assert auc_std == pytest.approx(auc_hc3, abs=1e-6), (
        f"AUC differs between standard and HC3: {auc_std:.6f} vs {auc_hc3:.6f}"
    )


def test_calibration_bins_present(logistic_outputs):
    """At least 8 calibration bins should be populated."""
    bin_counts = logistic_outputs["standard"]["calibration"]["bin_counts"]
    assert len(bin_counts) >= 8, f"Only {len(bin_counts)} calibration bins"


def test_schema_round_trip(logistic_outputs):
    """LogisticRegressionReport schema must accept the emitted JSON without error."""
    from regression_pack_core.schemas import LogisticRegressionReport

    for variant, data in logistic_outputs.items():
        report = LogisticRegressionReport(**data)
        assert report.roc.auc > 0, f"{variant}: AUC should be positive"
