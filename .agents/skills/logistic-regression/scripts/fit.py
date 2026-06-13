"""logistic-regression skill — main entry point.

Fits a binary logistic regression (statsmodels Logit), produces
report.json (LogisticRegressionReport) and report.html.

Usage:
    python fit.py --data data.csv --target churned --positive-class yes
                  --features x1,x2,x3 --output out/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))

from calibration import compute_calibration  # noqa: E402
from classify import compute_classification_stats  # noqa: E402
from coefficients import build_odds_ratio_table  # noqa: E402
from interpret import generate_logistic_interpretations  # noqa: E402
from marginal import compute_marginal_effects  # noqa: E402
from render import render_report  # noqa: E402
from roc import compute_roc  # noqa: E402

from regression_pack_core import validators  # noqa: E402
from regression_pack_core.schemas import (  # noqa: E402
    Flag,
    LogisticRegressionReport,
    Recommendation,
    Severity,
)

MIN_ROWS = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit binary logistic regression and produce a structured report."
    )
    p.add_argument("--data", required=True, help="Path to CSV or Parquet data file")
    p.add_argument("--target", required=True, help="Binary target column")
    p.add_argument(
        "--positive-class",
        default=None,
        help=(
            "Which class to treat as 1 "
            "(default: '1' if numeric 0/1, else alphabetically last value)"
        ),
    )
    p.add_argument(
        "--features", required=True, help="Comma-separated predictor columns, or 'all'"
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--robust-se",
        choices=["HC0", "HC1", "HC2", "HC3"],
        default=None,
        help="Heteroscedasticity-robust standard errors",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification stats (default: 0.5)",
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
    raise ValueError(f"Unsupported format '{p.suffix}'")


def _resolve_positive_class(series: pd.Series, user_arg: str | None):
    """Determine which value is the positive class (mapped to 1)."""
    unique_vals = sorted(series.dropna().unique().tolist())
    if len(unique_vals) != 2:
        raise ValueError(
            f"Target must have exactly 2 unique values; found {len(unique_vals)}: {unique_vals}"
        )
    if user_arg is not None:
        # Try to coerce user_arg to the right type
        if str(unique_vals[0]) == user_arg:
            return unique_vals[0]
        if str(unique_vals[1]) == user_arg:
            return unique_vals[1]
        raise ValueError(
            f"--positive-class '{user_arg}' not found in target; "
            f"available values: {unique_vals}"
        )
    # Default: if values are 0 and 1, positive = 1; else alphabetically last
    if set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}:
        return 1
    return unique_vals[-1]  # alphabetically last


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

    # Validate (non-numeric target OK — we recode below)
    vr = validators.validate_regression_inputs(
        df, args.target, feature_names_raw, require_numeric_target=False
    )
    if not vr.ok:
        print(f"Error: {vr.message}", file=sys.stderr)
        for issue in vr.issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    # Resolve positive class
    try:
        positive_class = _resolve_positive_class(df[args.target], args.positive_class)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Drop NA, encode features
    complete = df[[args.target, *feature_names_raw]].dropna().reset_index(drop=True)
    if len(complete) < MIN_ROWS:
        print(
            f"Error: only {len(complete)} rows after dropping NA (min {MIN_ROWS}).",
            file=sys.stderr,
        )
        return 1

    # Encode target as 0/1
    y = (complete[args.target] == positive_class).astype(int).to_numpy()
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    pos_rate = n_pos / n

    if n_pos < 5 or n_neg < 5:
        print(
            f"Error: too few observations in one class "
            f"(positive={n_pos}, negative={n_neg}). Min 5 per class.",
            file=sys.stderr,
        )
        return 1

    X_df, _dummy_map = validators.coerce_features(complete, feature_names_raw)
    feature_names = list(X_df.columns)
    k = len(feature_names)
    # Build a named DataFrame so statsmodels carries column names through
    X_df_const = X_df.copy()
    X_df_const.insert(0, "const", 1.0)

    # Fit
    cov_type = args.robust_se if args.robust_se else "nonrobust"
    try:
        model = sm.Logit(y, X_df_const).fit(
            cov_type=cov_type, maxiter=200, disp=False
        )
    except Exception as e:
        print(f"Error: logistic fit failed — {e}", file=sys.stderr)
        return 1

    # Predicted probabilities
    y_pred_prob = model.predict()

    # Sub-module outputs
    coefs = build_odds_ratio_table(model)
    marginal_effects = compute_marginal_effects(model)
    roc_data = compute_roc(y, y_pred_prob)
    calibration_data = compute_calibration(y, y_pred_prob)
    class_stats = compute_classification_stats(y, y_pred_prob, threshold=args.threshold)
    interpretations = generate_logistic_interpretations(
        coefs,
        marginal_effects,
        target=args.target,
        positive_class_label=str(positive_class),
    )

    # Fit statistics dict
    ll_null = model.llnull
    ll_full = model.llf
    pseudo_r2 = 1.0 - (ll_full / ll_null) if ll_null != 0 else 0.0
    fit_statistics = {
        "n_obs": n,
        "n_features": k,
        "log_likelihood": round(float(ll_full), 4),
        "ll_null": round(float(ll_null), 4),
        "pseudo_r_squared": round(float(pseudo_r2), 6),
        "aic": round(float(model.aic), 4),
        "bic": round(float(model.bic), 4),
        "converged": bool(model.mle_retvals.get("converged", True)),
        "n_iterations": int(model.mle_retvals.get("iterations", 0)),
    }

    # Headline
    n_sig = sum(1 for c in coefs if c.feature != "const" and c.p_value < 0.05)
    headline = (
        f"The model achieves AUC = {roc_data.auc:.3f} on n = {n:,} "
        f"({pos_rate * 100:.1f}% positive class); "
        f"{n_sig} of {k} predictor(s) significant at p < 0.05."
    )

    # Flags
    flags: list[Flag] = []
    if pos_rate < 0.20 or pos_rate > 0.80:
        flags.append(Flag(
            severity=Severity.WARN,
            code="CLASS_IMBALANCE",
            message=(
                f"Class imbalance detected: {pos_rate * 100:.1f}% positive. "
                "Accuracy may be misleading; consider balanced accuracy and AUC instead."
            ),
            detail={"pos_rate": pos_rate},
        ))

    large_coef = max((abs(c.log_odds_coefficient) for c in coefs if c.feature != "const"), default=0)
    converged = fit_statistics["converged"]
    if not converged or large_coef > 10:
        flags.append(Flag(
            severity=Severity.HIGH,
            code="CONVERGENCE_ISSUE",
            message=(
                "Model may have convergence issues or near-perfect separation "
                f"(max |β| = {large_coef:.1f}, converged = {converged}). "
                "Coefficient estimates may be unreliable."
            ),
            detail={"converged": converged, "max_abs_coef": large_coef},
        ))

    if roc_data.auc < 0.65:
        flags.append(Flag(
            severity=Severity.WARN,
            code="LOW_AUC",
            message=f"AUC = {roc_data.auc:.3f} is below 0.65 — model discriminates weakly.",
            detail={"auc": roc_data.auc},
        ))

    # Recommendations
    recommendations: list[Recommendation] = []
    if pos_rate < 0.10 or pos_rate > 0.90:
        recommendations.append(Recommendation(
            action="consider threshold tuning or reweighting",
            reason=(
                "Severe imbalance; default threshold 0.5 may miss the minority class. "
                "Try --threshold or class-weight balancing."
            ),
            priority=Severity.WARN,
        ))
    if large_coef > 8:
        recommendations.append(Recommendation(
            action="inspect for perfect separation and consider Firth regression",
            reason="Very large coefficient magnitudes suggest quasi-perfect separation.",
            priority=Severity.WARN,
        ))

    report = LogisticRegressionReport(
        fit_statistics=fit_statistics,
        coefficients=coefs,
        marginal_effects=marginal_effects,
        interpretations=interpretations,
        classification_stats=class_stats,
        roc=roc_data,
        calibration=calibration_data,
        headline=headline,
        target_name=args.target,
        positive_class=positive_class,
        robust_se_used=args.robust_se,
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

    # Summary
    print("✓ Logistic regression fit complete")
    print(f"  n = {n:,}  k = {k}  pos_rate = {pos_rate * 100:.1f}%")
    print(f"  AUC = {roc_data.auc:.4f}  Brier = {calibration_data.brier_score:.4f}"
          f"  McFadden R² = {pseudo_r2:.4f}")
    print(f"  converged = {converged}  {n_sig}/{k} significant at p < 0.05")
    if flags:
        for f in flags:
            print(f"  [{f.severity.value}] {f.code}: {f.message[:80]}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
