"""diagnostics skill — main entry point.

Takes a fit report (LinearRegressionReport JSON) plus the original data,
refits the model, runs the full diagnostic battery, and writes
diagnostics.json (DiagnosticsReport) and diagnostics.html.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))

import assumptions as assumptions_mod  # noqa: E402
import bias_variance as bias_variance_mod  # noqa: E402
import influence as influence_mod  # noqa: E402
from render import render_report  # noqa: E402
from residuals import residual_data  # noqa: E402
from triage import ASSUMPTION_CODES, build_verdict  # noqa: E402

from regression_pack_core import validators  # noqa: E402
from regression_pack_core.schemas import (  # noqa: E402
    DiagnosticsReport,
    Flag,
    LinearRegressionReport,
    Recommendation,
    Severity,
    Status,
)

R2_TOLERANCE = 1e-6

# Remediation patterns (see references/remediation.md), keyed by flag code
REMEDIATIONS: dict[str, tuple[str, str]] = {
    "HETEROSCEDASTICITY": (
        "use_robust_se",
        "Refit with heteroscedasticity-robust standard errors (HC3 for small samples), "
        "or log/sqrt-transform the target. See linear-regression/references/robust_se.md.",
    ),
    "MISSED_NONLINEARITY": (
        "add_polynomial_term",
        "Add a polynomial term for the offending feature, identified via partial residual plots.",
    ),
    "HIGH_VIF": (
        "drop_collinear_feature",
        "Drop one of the collinear features, or switch to ridge regression.",
    ),
    "NON_NORMAL_RESIDUALS": (
        "transform_target",
        "Transform the target (log or Box-Cox), or rely on the CLT if n is large.",
    ),
    "AUTOCORRELATION": (
        "use_time_series_model",
        "If the data is time-ordered, use OLS with AR errors or a time-series model.",
    ),
    "HIGH_COOKS_D": (
        "inspect_row",
        "Inspect the flagged observation; if it is a data error, fix it; if legitimate, "
        "refit with and without it and compare estimates.",
    ),
    "INFLUENTIAL_POINTS": (
        "inspect_row",
        "Inspect the flagged observations; refit with and without them to gauge sensitivity.",
    ),
    "HIGH_VARIANCE": (
        "regularize",
        "Regularise (ridge/lasso), gather more data, or simplify the model.",
    ),
    "HIGH_BIAS": (
        "add_features",
        "Add features, polynomial terms, or interactions — the model underfits.",
    ),
}

SEVERITY_BY_STATUS = {Status.FAIL: Severity.HIGH, Status.WARN: Severity.WARN}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full diagnostic battery on a fitted regression model."
    )
    parser.add_argument("--fit-report", required=True, help="Path to LinearRegressionReport JSON")
    parser.add_argument("--data", required=True, help="Path to the original data file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--learning-curve", action="store_true", help="Include learning curve (slower)"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Held-out fraction for train/test gap"
    )
    parser.add_argument("--dataset-name", default="", help="Dataset name for the report header")
    return parser.parse_args(argv)


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file format '{p.suffix}' — expected .csv or .parquet")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 1. Load and validate the fit report
    fit_path = Path(args.fit_report)
    if not fit_path.exists():
        print(f"Error: fit report not found: {fit_path}", file=sys.stderr)
        return 1
    try:
        fit_report = LinearRegressionReport.model_validate_json(fit_path.read_text())
    except ValidationError as e:
        print(f"Error: fit report does not validate as LinearRegressionReport:\n{e}", file=sys.stderr)
        return 1

    if not fit_report.target or not fit_report.features:
        print(
            "Error: fit report lacks target/features fields — re-run the "
            "linear-regression skill to regenerate it.",
            file=sys.stderr,
        )
        return 1

    # 2. Reconstruct X and y exactly as the fit skill did
    try:
        df = load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    target, features = fit_report.target, fit_report.features
    missing = [c for c in [target, *features] if c not in df.columns]
    if missing:
        print(f"Error: column(s) missing from data: {', '.join(missing)}", file=sys.stderr)
        return 1

    complete = df[[target, *features]].dropna().reset_index(drop=True)
    X, _ = validators.coerce_features(complete, features)
    y = complete[target].astype(float)
    if fit_report.target_transform == "log":
        if (y <= 0).any():
            print("Error: log target transform requires all target values > 0.", file=sys.stderr)
            return 1
        y = np.log(y)

    # 3. Refit and verify it matches the original
    cov_type = fit_report.robust_se_used or "nonrobust"
    model = sm.OLS(y, sm.add_constant(X)).fit(cov_type=cov_type)
    r2_diff = abs(float(model.rsquared) - fit_report.fit_statistics.r_squared)
    if r2_diff > R2_TOLERANCE:
        print(
            f"Error: refit does not match the original fit (R² differs by {r2_diff:.2e}). "
            "Was the data file modified since the fit?",
            file=sys.stderr,
        )
        return 1

    # 4. Run the battery
    checks = assumptions_mod.run_all(model, X, y)
    influence = influence_mod.run(model, X)
    bias_variance = bias_variance_mod.run(
        X,
        y,
        test_split=args.test_split,
        cv_folds=args.cv_folds,
        include_learning_curve=args.learning_curve,
    )

    # 5. Build flags
    flags: list[Flag] = []
    for check in checks:
        if check.status == Status.OK:
            continue
        flags.append(
            Flag(
                severity=SEVERITY_BY_STATUS[check.status],
                code=ASSUMPTION_CODES[check.name],
                message=check.evidence,
                detail=check.detail,
            )
        )
    n = len(X)
    severe_cooks = [p for p in influence.cooks_d_outliers if p.cooks_distance > 5 * (4 / n)]
    if severe_cooks:
        flags.append(
            Flag(
                severity=Severity.HIGH,
                code="HIGH_COOKS_D",
                message=(
                    f"{len(severe_cooks)} observation(s) exceed the Cook's D threshold by >5x "
                    f"(max D = {influence.max_cooks_d:.3f}); estimates may hinge on them."
                ),
                detail={"rows": [p.row_index for p in severe_cooks]},
            )
        )
    # Mild 4/n exceedances are expected by chance and stay unflagged — they
    # remain visible in the influence table of the HTML report.
    if bias_variance.verdict in ("high_variance", "high_bias"):
        flags.append(
            Flag(
                severity=Severity.HIGH,
                code=bias_variance.verdict.upper(),
                message=bias_variance.evidence,
            )
        )

    # 6. Verdict
    verdict = build_verdict(checks, influence, bias_variance)

    # 7. Recommendations — one per actionable flag, FAIL/HIGH first
    recommendations: list[Recommendation] = []
    for flag in sorted(flags, key=lambda f: 0 if f.severity == Severity.HIGH else 1):
        if flag.code in REMEDIATIONS:
            action, reason = REMEDIATIONS[flag.code]
            target_str = None
            if flag.detail and "rows" in flag.detail:
                target_str = f"rows {flag.detail['rows']}"
            recommendations.append(
                Recommendation(
                    action=action, target=target_str, reason=reason, priority=flag.severity
                )
            )

    # 8. Assemble, write JSON + HTML
    report = DiagnosticsReport(
        assumptions=checks,
        influence=influence,
        bias_variance=bias_variance,
        verdict=verdict,
        flags=flags,
        recommendations=recommendations,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "diagnostics.html"
    report.report_html_path = str(html_path)

    json_path = out_dir / "diagnostics.json"
    json_path.write_text(report.model_dump_json(indent=2))

    html_doc = render_report(
        report,
        residual_data(model),
        dataset_name=args.dataset_name,
        n_obs=n,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    html_path.write_text(html_doc)

    status_counts = {s.value: sum(1 for c in checks if c.status == s) for s in Status}
    print("✓ Diagnostics complete")
    print(f"  verdict: {verdict.overall}")
    print(f"  assumptions: {status_counts['ok']} ok, {status_counts['warn']} warn, {status_counts['fail']} fail")
    print(f"  {verdict.headline}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
