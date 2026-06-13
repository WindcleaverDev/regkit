"""Detection evals: each planted violation in the synthetic datasets must be
detected with the correct status, and clean data must come back clean.

Runs the actual skill CLIs end-to-end via subprocess, exactly as a user would.
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
def outputs(tmp_path_factory) -> dict[str, dict]:
    """Generate data, fit and diagnose all three synthetic datasets once."""
    out_root = tmp_path_factory.mktemp("synthetic")
    gen = run_cli("examples/synth_data.py")
    assert gen.returncode == 0, gen.stderr

    results: dict[str, dict] = {}
    for name in ("clean", "heteroscedastic", "influential"):
        out_dir = out_root / name
        fit = run_cli(
            ".agents/skills/linear-regression/scripts/fit.py",
            "--data", f"examples/data/{name}.csv",
            "--target", "y",
            "--features", "x1,x2",
            "--output", str(out_dir),
        )
        assert fit.returncode == 0, fit.stderr
        diag = run_cli(
            ".agents/skills/diagnostics/scripts/diagnose.py",
            "--fit-report", str(out_dir / "report.json"),
            "--data", f"examples/data/{name}.csv",
            "--output", str(out_dir),
        )
        assert diag.returncode == 0, diag.stderr
        results[name] = {
            "fit": json.loads((out_dir / "report.json").read_text()),
            "diag": json.loads((out_dir / "diagnostics.json").read_text()),
        }
    return results


def assumption(diag: dict, name: str) -> dict:
    return next(a for a in diag["assumptions"] if a["name"] == name)


def test_clean_diagnoses_clean(outputs):
    diag = outputs["clean"]["diag"]
    assert diag["verdict"]["overall"] == "clean"
    assert all(a["status"] == "ok" for a in diag["assumptions"])
    assert diag["bias_variance"]["verdict"] == "good_fit"
    assert diag["flags"] == []


def test_heteroscedastic_fails_homoscedasticity(outputs):
    diag = outputs["heteroscedastic"]["diag"]
    assert assumption(diag, "homoscedasticity")["status"] == "fail"
    assert any(r["action"] == "use_robust_se" for r in diag["recommendations"])


def test_influential_point_flagged(outputs):
    diag = outputs["influential"]["diag"]
    n = outputs["influential"]["fit"]["fit_statistics"]["n_observations"]
    row0 = next(
        (p for p in diag["influence"]["cooks_d_outliers"] if p["row_index"] == 0), None
    )
    assert row0 is not None, "row 0 must appear among Cook's D outliers"
    assert row0["cooks_distance"] > 5 * (4 / n)
    assert any(f["code"] == "HIGH_COOKS_D" for f in diag["flags"])


def test_fit_recovers_planted_coefficients(outputs):
    coefs = {c["feature"]: c["coefficient"] for c in outputs["clean"]["fit"]["coefficients"]}
    assert abs(coefs["const"] - 2.0) < 0.2
    assert abs(coefs["x1"] - 1.5) < 0.2
    assert abs(coefs["x2"] - (-0.8)) < 0.2
