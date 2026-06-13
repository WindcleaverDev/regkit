"""model-comparison skill — main entry point.

Accepts two or more report.json files from any regression skill and
produces a ModelComparisonReport with Akaike weights, LR tests for
nested pairs, and a verdict.

Usage:
    python compare.py \\
        --reports out/ols/report.json out/ridge/report.json out/lasso/report.json \\
        --names "OLS" "Ridge" "Lasso" \\
        --output out/comparison/ \\
        [--alpha 0.05] [--dataset-name "tips"]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingest import load_model_entry  # noqa: E402
from lr_test import run_lr_test  # noqa: E402
from nesting import find_nested_pairs  # noqa: E402
from render import render_report  # noqa: E402
from verdict import build_verdict  # noqa: E402
from weights import compute_akaike_weights  # noqa: E402

from regression_pack_core.schemas import (  # noqa: E402
    Flag,
    ModelComparisonReport,
    Recommendation,
    Severity,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple regression model reports."
    )
    p.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Paths to report.json files (at least 2)",
    )
    p.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=(
            "Human-readable names for each model (same order as --reports). "
            "Default: basename of each report directory."
        ),
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for LR tests (default: 0.05)",
    )
    p.add_argument("--dataset-name", default="", help="Dataset label for report header")
    return p.parse_args(argv)


def _default_name(path: str, idx: int) -> str:
    p = Path(path)
    parent = p.parent.name
    return parent if parent not in (".", "") else f"model_{idx + 1}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if len(args.reports) < 2:
        print("Error: --reports requires at least 2 report paths.", file=sys.stderr)
        return 1

    names = args.names or [_default_name(r, i) for i, r in enumerate(args.reports)]
    if len(names) != len(args.reports):
        print(
            f"Error: --names has {len(names)} items but --reports has {len(args.reports)}.",
            file=sys.stderr,
        )
        return 1

    entries = []
    raw_reports = []
    for report_path, name in zip(args.reports, names, strict=True):
        try:
            entry, raw = load_model_entry(report_path, name)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        entries.append(entry)
        raw_reports.append(raw)

    # Nesting + LR tests
    nested_pairs = find_nested_pairs(raw_reports)
    lr_tests = []
    for i, j in nested_pairs:
        result = run_lr_test(raw_reports[i], raw_reports[j], alpha=args.alpha)
        if result is not None:
            lr_tests.append(result)

    # Akaike weights
    akaike_weights = compute_akaike_weights(entries)

    # Flags
    flags: list[Flag] = []
    n_obs_vals = [e.n_observations for e in entries]
    if max(n_obs_vals) > 0 and (max(n_obs_vals) - min(n_obs_vals)) > 0:
        flags.append(Flag(
            severity=Severity.WARN,
            code="SAMPLE_SIZE_MISMATCH",
            message=(
                "Models were fit on different sample sizes. "
                "AIC comparison is only valid for models fit on the same observations."
            ),
            detail={"n_obs_by_model": {e.name: e.n_observations for e in entries}},
        ))

    families = {e.family for e in entries}
    comparable_families = {f if f in ("linear",) else ("linear" if f in ("ridge", "lasso", "elasticnet") else f) for f in families}
    if len(comparable_families) > 1:
        flags.append(Flag(
            severity=Severity.INFO,
            code="CROSS_FAMILY_COMPARISON",
            message=(
                "Models are from different families. "
                "Direct AIC comparison across families (e.g., linear vs logistic) "
                "is not statistically valid; use CV scores or domain knowledge instead."
            ),
            detail={"families": list(families)},
        ))

    if akaike_weights is None and not lr_tests:
        flags.append(Flag(
            severity=Severity.INFO,
            code="NO_FORMAL_COMPARISON",
            message=(
                "Neither Akaike weights nor LR tests are available for these models. "
                "Ranking is based on primary fit quality (adj-R² or pseudo-R²)."
            ),
            detail={},
        ))

    # Recommendations
    recommendations: list[Recommendation] = []
    if len(entries) > 4:
        recommendations.append(Recommendation(
            action="narrow down candidates before final selection",
            reason="Comparing many models inflates the chance of false model selection; prefer theory-guided pre-screening.",
            priority=Severity.WARN,
        ))

    # Verdict
    comparison_verdict = build_verdict(entries, lr_tests, akaike_weights, alpha=args.alpha)

    report = ModelComparisonReport(
        models=entries,
        lr_tests=lr_tests,
        akaike_weights=akaike_weights,
        verdict=comparison_verdict,
        flags=flags,
        recommendations=recommendations,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    html_path = out_dir / "report.html"
    report.report_html_path = str(html_path)

    html_doc = render_report(
        report,
        raw_reports,
        dataset_name=args.dataset_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    html_path.write_text(html_doc)

    # Summary
    print("✓ Model comparison complete")
    print(f"  Models compared: {', '.join(names)}")
    print(f"  Verdict: {comparison_verdict.overall}")
    print(f"  Recommended: {comparison_verdict.recommended_model}")
    if akaike_weights:
        weight_strs = [
            f"{n} = {w:.3f}"
            for n, w in zip(akaike_weights.model_names, akaike_weights.weights, strict=True)
        ]
        print(f"  Akaike weights: {'; '.join(weight_strs)}")
        print(f"  Weights sum: {sum(akaike_weights.weights):.6f}")
    if lr_tests:
        for t in lr_tests:
            print(f"  LR test {t.nested_model} vs {t.full_model}: p = {t.p_value:.4f}")
    if flags:
        for f in flags:
            print(f"  [{f.severity.value}] {f.code}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
