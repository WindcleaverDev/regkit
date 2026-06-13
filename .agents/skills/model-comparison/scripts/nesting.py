"""Detect nested model pairs for LR-test eligibility."""

from __future__ import annotations


def _comparable_family(f: str) -> str:
    """Normalise regularised families to 'linear' for nesting checks."""
    if f in ("ridge", "lasso", "elasticnet"):
        return "linear"
    return f


def find_nested_pairs(reports: list[dict]) -> list[tuple[int, int]]:
    """Return (i, j) index pairs where report[i] is nested in report[j].

    Criteria:
    - Same comparison family (linear/logistic)
    - Same target name (target_name / target field)
    - features(i) ⊊ features(j)  — strict subset

    Returns indices sorted so the simpler model is first.
    """
    nested: list[tuple[int, int]] = []
    n = len(reports)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fi = _comparable_family(reports[i]["_family"])
            fj = _comparable_family(reports[j]["_family"])
            if fi != fj:
                continue
            ti = reports[i].get("_target")
            tj = reports[j].get("_target")
            if ti is not None and tj is not None and ti != tj:
                continue
            feats_i = set(reports[i]["_features"])
            feats_j = set(reports[j]["_features"])
            if feats_i < feats_j:  # strict subset
                nested.append((i, j))
    return nested
