"""linear-regression skill — main entry point.

Fits an OLS model, writes report.json (LinearRegressionReport) and
report.html to the output directory.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))

from coefficients import build_coefficient_table  # noqa: E402
from interpret import generate_interpretations  # noqa: E402
from render import render_report  # noqa: E402

from regression_pack_core import validators  # noqa: E402
from regression_pack_core.schemas import (  # noqa: E402
    FitQuality,
    FitStatistics,
    LinearRegressionReport,
)

MIN_ROWS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an OLS linear regression and produce a structured report."
    )
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet data file")
    parser.add_argument("--target", required=True, help="Numeric target column")
    parser.add_argument(
        "--features", required=True, help="Comma-separated predictor column names"
    )
    parser.add_argument("--output", required=True, help="Output directory (created if missing)")
    parser.add_argument("--log-target", action="store_true", help="Apply np.log to target")
    parser.add_argument(
        "--robust-se",
        choices=["HC0", "HC1", "HC2", "HC3"],
        default=None,
        help="Heteroscedasticity-robust standard errors",
    )
    parser.add_argument(
        "--standardize", action="store_true", help="Also report standardized beta coefficients"
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


def fit_quality_from(adj_r2: float, r2: float) -> FitQuality:
    if adj_r2 >= 0.7:
        label = "very_strong"
    elif adj_r2 >= 0.5:
        label = "strong"
    elif adj_r2 >= 0.3:
        label = "moderate"
    else:
        label = "weak"
    return FitQuality(r_squared=r2, adj_r_squared=adj_r2, interpretation=label)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        df = load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    features = [f.strip() for f in args.features.split(",") if f.strip()]

    result = validators.validate_regression_inputs(df, args.target, features)
    if not result.ok:
        print(f"Error: {result.message}", file=sys.stderr)
        for issue in result.issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    # One-hot encode categoricals, then drop incomplete rows
    complete = df[[args.target, *features]].dropna().reset_index(drop=True)
    if len(complete) < MIN_ROWS:
        print(
            f"Error: only {len(complete)} rows remain after dropping NA "
            f"(minimum {MIN_ROWS}).",
            file=sys.stderr,
        )
        return 1

    X, dummy_map = validators.coerce_features(complete, features)
    y = complete[args.target].astype(float)

    target_transform = None
    if args.log_target:
        if (y <= 0).any():
            print(
                "Error: --log-target requires all target values > 0.",
                file=sys.stderr,
            )
            return 1
        y = np.log(y)
        target_transform = "log"

    cov_type = args.robust_se if args.robust_se else "nonrobust"
    model = sm.OLS(y, sm.add_constant(X)).fit(cov_type=cov_type)

    coefs = build_coefficient_table(model, X, standardize=args.standardize, y=y)

    n_obs = int(model.nobs)
    n_features = X.shape[1]
    fit_stats = FitStatistics(
        n_observations=n_obs,
        n_features=n_features,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue),
        f_p_value=float(model.f_pvalue),
        aic=float(model.aic),
        bic=float(model.bic),
        log_likelihood=float(model.llf),
        residual_std_error=float(np.sqrt(model.mse_resid)),
        df_residuals=int(model.df_resid),
    )

    quality = fit_quality_from(fit_stats.adj_r_squared, fit_stats.r_squared)

    # Map each dummy column back to its originating feature
    dummy_origin = {col: orig for orig, cols in dummy_map.items() for col in cols}
    interpretations = generate_interpretations(
        coefs,
        target=args.target,
        target_transform=target_transform,
        dummy_origin=dummy_origin,
    )

    n_sig = sum(1 for c in coefs if c.feature != "const" and c.p_value < 0.05)
    headline = (
        f"The model explains {fit_stats.adj_r_squared * 100:.1f}% of variance in "
        f"{args.target} ({n_sig} of {n_features} predictors significant at p < 0.05)."
    )

    report = LinearRegressionReport(
        fit_statistics=fit_stats,
        coefficients=coefs,
        interpretations=interpretations,
        fit_quality=quality,
        headline=headline,
        target=args.target,
        features=features,
        target_transform=target_transform,
        robust_se_used=args.robust_se,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    report.report_html_path = str(html_path)

    json_path = out_dir / "report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    html_doc = render_report(
        report,
        target=args.target,
        dataset_name=args.dataset_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    html_path.write_text(html_doc)

    print("✓ Linear regression fit complete")
    print(f"  n = {n_obs}, k = {n_features}")
    print(f"  R² = {fit_stats.r_squared:.4f}  (adj: {fit_stats.adj_r_squared:.4f})")
    print(f"  {headline}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
