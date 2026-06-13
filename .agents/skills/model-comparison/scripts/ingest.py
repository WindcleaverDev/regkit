"""Load report JSONs and derive ModelEntry objects for comparison."""

from __future__ import annotations

import json
from pathlib import Path

from regression_pack_core.schemas import ModelEntry


def _sniff_family(data: dict) -> str:
    """Determine model family from report JSON fields."""
    method = data.get("method")
    if method in ("ridge", "lasso", "elasticnet"):
        return method
    if "roc" in data or "marginal_effects" in data:
        return "logistic"
    return "linear"


def _extract_features(data: dict, family: str) -> list[str]:
    """Extract predictor names from a report."""
    coefs = data.get("coefficients", [])
    if not coefs:
        return []
    if family == "logistic":
        return [c["feature"] for c in coefs if c.get("feature") != "const"]
    # linear / regularized: CoefficientRow has "feature"
    return [c["feature"] for c in coefs if c.get("feature") not in ("const", "(Intercept)")]


def _extract_target(data: dict, family: str) -> str | None:
    if family == "logistic":
        return data.get("target_name")
    return data.get("target")


def load_model_entry(report_path: str | Path, name: str) -> tuple[ModelEntry, dict]:
    """Load a report JSON and return (ModelEntry, raw_dict).

    The raw_dict is kept for downstream callers (nesting, LR test, etc.).
    """
    path = Path(report_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    data = json.loads(path.read_text())
    family = _sniff_family(data)

    # Extract fit statistics
    fs = data.get("fit_statistics") or {}
    if isinstance(fs, dict):
        n_obs = int(fs.get("n_obs") or fs.get("n_observations") or 0)
        n_features = int(fs.get("n_features") or 0)
        log_likelihood = fs.get("log_likelihood")
        aic = fs.get("aic")
        bic = fs.get("bic")
    else:
        # FitStatistics is a Pydantic model serialised as dict
        n_obs = int(getattr(fs, "n_observations", 0))
        n_features = int(getattr(fs, "n_features", 0))
        log_likelihood = getattr(fs, "log_likelihood", None)
        aic = getattr(fs, "aic", None)
        bic = getattr(fs, "bic", None)

    # Linear family uses FitStatistics object (serialised as nested dict)
    if family == "linear":
        n_obs = int(fs.get("n_observations", n_obs))
        n_features = int(fs.get("n_features", n_features))
        log_likelihood = fs.get("log_likelihood", log_likelihood)
        aic = fs.get("aic", aic)
        bic = fs.get("bic", bic)

    # Regularized stores FitStatistics same as linear
    if family in ("ridge", "lasso", "elasticnet"):
        n_obs = int(fs.get("n_observations", n_obs))
        n_features = int(fs.get("n_features", n_features))
        log_likelihood = fs.get("log_likelihood", log_likelihood)
        aic = fs.get("aic", aic)
        bic = fs.get("bic", bic)

    # fit_quality_primary
    if family == "logistic":
        fq = float(fs.get("pseudo_r_squared", 0.0))
    elif family == "linear":
        fq_block = data.get("fit_quality") or {}
        fq = float(fq_block.get("adj_r_squared", fs.get("adj_r_squared", 0.0)))
    else:
        # regularized
        fq = float(fs.get("adj_r_squared", 0.0))

    # Build notes
    notes: list[str] = []
    if data.get("robust_se_used"):
        notes.append(f"robust SE: {data['robust_se_used']}")
    if data.get("log_transform_target"):
        notes.append("log-target")
    if family in ("ridge", "lasso", "elasticnet"):
        alpha = data.get("selected_alpha")
        if alpha is not None:
            notes.append(f"α = {alpha:.4g}")

    entry = ModelEntry(
        name=name,
        source_report_path=str(path),
        family=family,
        n_observations=n_obs,
        n_features=n_features,
        fit_quality_primary=round(fq, 6),
        aic=float(aic) if aic is not None else None,
        bic=float(bic) if bic is not None else None,
        notes=notes,
    )
    # Attach derived fields used by other modules
    data["_family"] = family
    data["_features"] = _extract_features(data, family)
    data["_target"] = _extract_target(data, family)
    data["_log_likelihood"] = float(log_likelihood) if log_likelihood is not None else None
    data["_n_obs"] = n_obs
    data["_name"] = name
    data["_path"] = str(path)

    return entry, data
