"""pre-analysis skill — main entry point.

Audits tabular data before fitting: target distribution, feature health,
multicollinearity, suspected nonlinearity, and concrete modeling recommendations.
Writes pre_analysis.json (PreAnalysisReport) and pre_analysis.html.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import features as feat_mod  # noqa: E402
import multicollinearity as mc_mod  # noqa: E402
import recommend as rec_mod  # noqa: E402
import relationships as rel_mod  # noqa: E402
import target as tgt_mod  # noqa: E402
from render import render_report  # noqa: E402

from regression_pack_core.schemas import Flag, PreAnalysisReport, Severity  # noqa: E402
from regression_pack_core.validators import (  # noqa: E402
    coerce_features,
    validate_regression_inputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit tabular data before fitting a regression model."
    )
    p.add_argument("--data", required=True, help="Path to CSV or Parquet file")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument(
        "--features",
        required=True,
        help="Comma-separated predictor columns, or 'all' to use every non-target column",
    )
    p.add_argument("--output", required=True, help="Output directory (created if missing)")
    p.add_argument("--dataset-name", default="", help="Dataset label for the report header")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 1. Load data
    try:
        df = load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # 2. Resolve --features all
    raw_features = args.features.strip()
    if raw_features.lower() == "all":
        features = [c for c in df.columns if c != args.target]
    else:
        features = [f.strip() for f in raw_features.split(",")]

    # 3. Validate inputs
    vr = validate_regression_inputs(df, args.target, features, require_numeric_target=False)
    if not vr.ok:
        print(f"Error: {vr.message}", file=sys.stderr)
        for issue in vr.issues:
            print(f"  • {issue}", file=sys.stderr)
        return 1

    n_total = len(df)

    # 4. Target audit
    target_audit = tgt_mod.audit_target(df[args.target], args.target)

    # 5. Per-feature audits
    feature_audits = [feat_mod.audit_feature(df[f], f, n_total) for f in features]

    # 6. Multicollinearity — computed on complete cases only
    complete = df[[args.target, *features]].dropna().reset_index(drop=True)
    X_raw = complete[features]
    X_encoded, _ = coerce_features(complete, features)

    vif_dict = mc_mod.compute_vif(X_encoded)
    flagged_vif = [f for f, v in vif_dict.items() if v > 5]
    max_vif = max(vif_dict.values()) if vif_dict else 0.0
    corr_names, corr_matrix = mc_mod.correlation_matrix(X_raw)

    multicollinearity = {
        "max_vif": max_vif,
        "flagged": flagged_vif,
        "vif_by_feature": vif_dict,
        "correlation_names": corr_names,
        "matrix": corr_matrix,
    }

    # 7. Suspected nonlinearity (continuous features vs numeric target only)
    y_complete = complete[args.target]
    suspected_nl: list[str] = []
    nl_plot_data: dict[str, dict] = {}
    if pd.api.types.is_numeric_dtype(y_complete):
        y_for_nl = y_complete.astype(float)
        for feat in features:
            col = complete[feat]
            if pd.api.types.is_numeric_dtype(col):
                if rel_mod.detect_nonlinearity(col.astype(float), y_for_nl):
                    suspected_nl.append(feat)
                    nl_plot_data[feat] = rel_mod.nonlinearity_plot_data(
                        col.astype(float), y_for_nl
                    )

    # 8. Synthesise modeling recommendations
    modeling_recs = rec_mod.synthesise_recommendations(
        target_audit, feature_audits, multicollinearity, suspected_nl
    )

    # 9. Build flags
    flags: list[Flag] = []

    # Target skewness
    if target_audit.recommendations and target_audit.skewness is not None:
        severity = Severity.HIGH if abs(target_audit.skewness) > 2.0 else Severity.WARN
        flags.append(
            Flag(
                severity=severity,
                code="TARGET_SKEW",
                message=(
                    f"Target '{args.target}' is skewed (skewness = {target_audit.skewness:.2f}); "
                    f"consider {target_audit.recommendations[0].replace('_', ' ')}."
                ),
                detail={
                    "skewness": target_audit.skewness,
                    "recommendations": target_audit.recommendations,
                },
            )
        )

    # Target outliers (separate from skew flag)
    if (
        target_audit.outlier_count is not None
        and target_audit.outlier_count > 0.05 * n_total
        and "winsorize" in target_audit.recommendations
        and "TARGET_SKEW" not in {f.code for f in flags}
    ):
        flags.append(
            Flag(
                severity=Severity.WARN,
                code="TARGET_OUTLIERS",
                message=(
                    f"Target has {target_audit.outlier_count} outliers "
                    f"({target_audit.outlier_count / n_total * 100:.1f}%); "
                    "consider winsorisation."
                ),
                detail={"outlier_count": target_audit.outlier_count},
            )
        )

    # Feature-level flags (one flag per feature flag code, deduped by feature+code)
    _FLAG_SEVERITY = {
        "quasi_id": Severity.HIGH,
        "high_missing": Severity.WARN,
        "high_cardinality": Severity.WARN,
        "near_constant": Severity.WARN,
    }
    _FLAG_MSG = {
        "high_missing": lambda fa: (
            f"Feature '{fa.name}' is missing {fa.missing_pct * 100:.1f}% of values."
        ),
        "high_cardinality": lambda fa: (
            f"Feature '{fa.name}' has {fa.n_unique} unique values (high cardinality)."
        ),
        "near_constant": lambda fa: (
            f"Feature '{fa.name}' is near-constant and may not contribute."
        ),
        "quasi_id": lambda fa: (
            f"Feature '{fa.name}' has {fa.n_unique} unique values — likely an ID column."
        ),
    }
    for fa in feature_audits:
        for flag_code in fa.flags:
            severity = _FLAG_SEVERITY.get(flag_code, Severity.INFO)
            msg_fn = _FLAG_MSG.get(flag_code)
            message = (
                msg_fn(fa)
                if msg_fn is not None
                else f"Issue in feature '{fa.name}': {flag_code}."
            )
            flags.append(
                Flag(
                    severity=severity,
                    code=flag_code.upper(),
                    message=message,
                    detail={"feature": fa.name},
                )
            )

    # Multicollinearity
    if max_vif > 10:
        flags.append(
            Flag(
                severity=Severity.HIGH,
                code="HIGH_VIF",
                message=(
                    f"Severe multicollinearity (max VIF = {max_vif:.1f}); "
                    "drop or combine flagged features."
                ),
                detail={"max_vif": max_vif, "flagged": flagged_vif},
            )
        )
    elif max_vif > 5:
        flags.append(
            Flag(
                severity=Severity.WARN,
                code="HIGH_VIF",
                message=f"Multicollinearity detected (max VIF = {max_vif:.1f}).",
                detail={"max_vif": max_vif, "flagged": flagged_vif},
            )
        )

    # Nonlinearity
    if suspected_nl:
        flags.append(
            Flag(
                severity=Severity.WARN,
                code="SUSPECTED_NONLINEARITY",
                message=f"Nonlinear relationship suspected for: {', '.join(suspected_nl)}.",
                detail={"features": suspected_nl},
            )
        )

    # 10. Assemble report
    report = PreAnalysisReport(
        n_samples=n_total,
        target=target_audit,
        features=feature_audits,
        multicollinearity=multicollinearity,
        suspected_nonlinearity=suspected_nl,
        flags=flags,
        modeling_recommendations=modeling_recs,
    )

    # 11. Write JSON + HTML
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "pre_analysis.html"
    report.report_html_path = str(html_path)

    json_path = out_dir / "pre_analysis.json"
    json_path.write_text(report.model_dump_json(indent=2))

    html_doc = render_report(
        report,
        y_raw=df[args.target],
        X_raw=X_raw,
        nonlinearity_plots=nl_plot_data,
        dataset_name=args.dataset_name,
        n_obs=n_total,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    html_path.write_text(html_doc)

    # 12. Print summary
    sev_counts = {s: sum(1 for f in flags if f.severity == s) for s in Severity}
    print("✓ Pre-analysis complete")
    print(f"  n = {n_total}, {len(features)} feature(s)")
    print(f"  target: {args.target} ({target_audit.type})")
    print(
        f"  flags: {sev_counts[Severity.HIGH]} high, "
        f"{sev_counts[Severity.WARN]} warn, "
        f"{sev_counts[Severity.INFO]} info"
    )
    print(f"  preferred estimator: {modeling_recs['preferred_estimator']}")
    if modeling_recs["transform_target"]:
        print(f"  → apply {modeling_recs['transform_target']} to target before fitting")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
