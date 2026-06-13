"""regularized-regression skill — main entry point.

Fits Ridge, Lasso, or ElasticNet with cross-validated alpha selection.
Writes report.json (RegularizedRegressionReport) and report.html.

Usage:
    python fit.py --data data.csv --target y --features x1,x2,x3
                  --method lasso --output out/ --dataset-name "My data"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from compare_ols import compare_to_ols  # noqa: E402
from interpret import generate_regularized_interpretations  # noqa: E402
from path import fit_regularized  # noqa: E402
from render import render_report  # noqa: E402
from selection import build_feature_selection  # noqa: E402

from regression_pack_core import validators  # noqa: E402
from regression_pack_core.schemas import (  # noqa: E402
    CoefficientRow,
    FitStatistics,
    Flag,
    Recommendation,
    RegularizedRegressionReport,
    Severity,
)

MIN_ROWS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit Ridge / Lasso / ElasticNet and produce a structured report."
    )
    p.add_argument("--data", required=True, help="Path to CSV or Parquet data file")
    p.add_argument("--target", required=True, help="Numeric target column")
    p.add_argument(
        "--features",
        required=True,
        help="Comma-separated predictor column names, or 'all'",
    )
    p.add_argument(
        "--method",
        choices=["ridge", "lasso", "elasticnet"],
        default="lasso",
        help="Regularisation method (default: lasso)",
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--log-target", action="store_true", help="Apply np.log to target first")
    p.add_argument(
        "--l1-ratio",
        type=float,
        default=0.5,
        help="l1_ratio for ElasticNet (default: 0.5)",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for alpha selection (default: 5)",
    )
    p.add_argument(
        "--alpha-rule",
        choices=["min", "1se"],
        default="min",
        help="Alpha selection rule: min (best CV) or 1se (parsimony; default: min)",
    )
    p.add_argument("--dataset-name", default="", help="Dataset label for report header")
    return p.parse_args(argv)


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported format '{p.suffix}' — expected .csv or .parquet")


def _approx_coef_rows(
    feature_names: list[str],
    coef_orig: np.ndarray,
    intercept_orig: float,
    X_orig: np.ndarray,
    y: np.ndarray,
) -> list[CoefficientRow]:
    """Build CoefficientRow list with approximate SE/CI/p from OLS."""
    n, k = X_orig.shape
    df_resid = max(n - k - 1, 1)

    # Try OLS to get SE baseline
    try:
        ols = sm.OLS(y, sm.add_constant(X_orig, has_constant="add")).fit()
        ols_bse = ols.bse[1:].to_numpy()  # SE per feature (exclude intercept)
    except Exception:
        ols_bse = np.full(k, np.nan)

    rows: list[CoefficientRow] = []

    # Intercept row — no inference
    rows.append(
        CoefficientRow(
            feature="const",
            coefficient=intercept_orig,
            std_error=0.0,
            t_stat=0.0,
            p_value=1.0,
            ci_lower=intercept_orig,
            ci_upper=intercept_orig,
        )
    )

    for i, name in enumerate(feature_names):
        coef = float(coef_orig[i])
        se = float(ols_bse[i]) if i < len(ols_bse) and np.isfinite(ols_bse[i]) else 0.0
        if se > 0:
            t = coef / se
            p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t), df=df_resid)))
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
        else:
            t, p_val = 0.0, 1.0
            ci_lo = ci_hi = coef

        rows.append(
            CoefficientRow(
                feature=name,
                coefficient=coef,
                std_error=se,
                t_stat=round(t, 4),
                p_value=round(p_val, 6),
                ci_lower=round(ci_lo, 6),
                ci_upper=round(ci_hi, 6),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        df = load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    raw_features = args.features.strip()
    if raw_features.lower() == "all":
        feature_names_raw = [c for c in df.columns if c != args.target]
    else:
        feature_names_raw = [f.strip() for f in raw_features.split(",") if f.strip()]

    vr = validators.validate_regression_inputs(df, args.target, feature_names_raw)
    if not vr.ok:
        print(f"Error: {vr.message}", file=sys.stderr)
        for issue in vr.issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    complete = df[[args.target, *feature_names_raw]].dropna().reset_index(drop=True)
    if len(complete) < MIN_ROWS:
        print(
            f"Error: only {len(complete)} rows after dropping NA (min {MIN_ROWS}).",
            file=sys.stderr,
        )
        return 1

    X_df, _dummy_map = validators.coerce_features(complete, feature_names_raw)
    feature_names = list(X_df.columns)
    X_orig = X_df.to_numpy(dtype=float)
    y = complete[args.target].astype(float).to_numpy()

    if args.log_target:
        if (y <= 0).any():
            print("Error: --log-target requires all target values > 0.", file=sys.stderr)
            return 1
        y = np.log(y)

    n, k = X_orig.shape

    # Standardise for regularisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)

    # Fit
    fit_result = fit_regularized(
        X_scaled,
        y,
        scaler,
        feature_names,
        method=args.method,
        cv_folds=args.cv_folds,
        l1_ratio=args.l1_ratio,
        rule=args.alpha_rule,
    )

    coef_orig = fit_result["coef_orig"]
    intercept_orig = fit_result["intercept_orig"]
    selected_alpha = fit_result["selected_alpha"]

    # Fit statistics from regularised predictions
    y_pred = X_orig @ coef_orig + intercept_orig
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    df_resid = max(n - k - 1, 1)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid
    sigma = float(np.sqrt(ss_res / df_resid))

    # Enforce exact zeros from sklearn's sparsity mask (L1 methods)
    coef_zero_mask = fit_result.get("coef_zero_mask")
    if coef_zero_mask is not None:
        coef_orig = coef_orig.copy()
        coef_orig[coef_zero_mask] = 0.0

    # Approximate AIC/BIC: use non-zero coefficient count as effective df
    k_eff = int(np.sum(~coef_zero_mask)) if coef_zero_mask is not None else k
    rss_over_n = ss_res / max(n, 1)
    log_lik = -n / 2.0 * (np.log(2 * np.pi * rss_over_n) + 1.0) if rss_over_n > 0 else 0.0
    aic_approx = -2.0 * log_lik + 2.0 * (k_eff + 1)
    bic_approx = -2.0 * log_lik + (k_eff + 1) * np.log(n)

    fit_statistics = FitStatistics(
        n_observations=n,
        n_features=k,
        r_squared=round(r2, 6),
        adj_r_squared=round(adj_r2, 6),
        f_statistic=0.0,  # not meaningful for regularised fit
        f_p_value=1.0,
        aic=round(float(aic_approx), 4),
        bic=round(float(bic_approx), 4),
        log_likelihood=round(float(log_lik), 4),
        residual_std_error=round(sigma, 6),
        df_residuals=df_resid,
    )

    # Coefficient table
    coef_rows = _approx_coef_rows(feature_names, coef_orig, intercept_orig, X_orig, y)

    # Feature selection — use sklearn's exact zero mask for L1-based methods
    feature_selection = build_feature_selection(
        coef_orig,
        feature_names,
        args.method,
        coef_zero_mask=fit_result.get("coef_zero_mask"),
    )

    # OLS comparison
    comp = compare_to_ols(X_orig, y, coef_orig, intercept_orig, feature_names)

    # Interpretations (non-zero features only)
    non_zero_coefs = [c for c in coef_rows if c.feature != "const" and abs(c.coefficient) > 1e-10]
    interpretations = generate_regularized_interpretations(
        non_zero_coefs,
        args.method,
        target=args.target,
        target_transform="log" if args.log_target else None,
    )

    # Headline
    n_selected = feature_selection.n_selected if feature_selection else k
    n_sig = sum(1 for c in non_zero_coefs if c.p_value < 0.05)
    if args.method == "ridge":
        headline = (
            f"Ridge regression explains {adj_r2 * 100:.1f}% of variance "
            f"in {args.target} (α = {selected_alpha:.4g}). "
            f"{n_sig} of {k} features are nominally significant (p < 0.05; SE approx)."
        )
    else:
        method_label = "Lasso" if args.method == "lasso" else "ElasticNet"
        headline = (
            f"{method_label} regression explains {adj_r2 * 100:.1f}% of variance "
            f"in {args.target} (α = {selected_alpha:.4g}). "
            f"{n_selected} of {k} features retained after regularisation."
        )

    # Flags
    flags: list[Flag] = []
    if r2 < 0.1:
        flags.append(Flag(
            severity=Severity.WARN,
            code="LOW_R2",
            message=f"In-sample R² is very low ({r2:.3f}). The model explains little variance.",
            detail={"r2": r2},
        ))
    if feature_selection and feature_selection.n_dropped == 0 and args.method in ("lasso", "elasticnet"):
        flags.append(Flag(
            severity=Severity.INFO,
            code="NO_SELECTION",
            message="Lasso/ElasticNet did not zero any coefficients at the selected α; "
                    "try increasing α or reducing the feature count.",
            detail={"selected_alpha": selected_alpha},
        ))
    if k >= n // 5:
        flags.append(Flag(
            severity=Severity.WARN,
            code="HIGH_DIMENSIONALITY",
            message=f"k/n ratio is {k / n:.2f} — regularisation is especially important here.",
            detail={"k": k, "n": n},
        ))

    # Recommendations
    recommendations: list[Recommendation] = []
    if args.method == "ridge" and feature_selection is None:
        recommendations.append(Recommendation(
            action="switch to --method lasso or --method elasticnet",
            reason="Ridge never zeroes coefficients; use Lasso/ElasticNet for feature selection.",
            priority=Severity.INFO,
        ))
    if n_selected <= 2 and feature_selection and feature_selection.n_selected <= 2:
        recommendations.append(Recommendation(
            action="reduce alpha or switch to ridge",
            reason="Very few features survived selection — over-regularisation may be discarding informative features.",
            priority=Severity.WARN,
        ))

    report = RegularizedRegressionReport(
        method=args.method,
        selected_alpha=selected_alpha,
        selected_l1_ratio=args.l1_ratio if args.method == "elasticnet" else None,
        fit_statistics=fit_statistics,
        coefficients=coef_rows,
        path=fit_result["path"],
        cv_curve=fit_result["cv_curve"],
        feature_selection=feature_selection,
        interpretations=interpretations,
        headline=headline,
        comparison_to_ols=comp,
        flags=flags,
        recommendations=recommendations,
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

    print(f"✓ {args.method.capitalize()} regression fit complete")
    print(f"  n = {n}, k = {k}")
    print(f"  R² = {r2:.4f}  (adj: {adj_r2:.4f})")
    print(f"  selected α = {selected_alpha:.4g}")
    if feature_selection:
        print(f"  features retained: {feature_selection.n_selected} / {k}")
        if feature_selection.dropped_features:
            print(f"  zeroed: {', '.join(feature_selection.dropped_features)}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
