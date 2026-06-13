"""Thin wrapper that re-uses linear-regression's interpretation engine.

Regularised coefficient point estimates are passed as if they were OLS
coefficients. SE, CI, and p-values come from an approximate OLS fit on the
same data (noted in caveats). The wrapper adds regularisation-specific caveats.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load linear-regression interpret.py by path to avoid name collision with this module
_LR_INTERPRET = Path(__file__).parents[2] / "linear-regression" / "scripts" / "interpret.py"
_spec = importlib.util.spec_from_file_location("lr_interpret", _LR_INTERPRET)
_lr_mod = importlib.util.module_from_spec(_spec)
sys.modules["lr_interpret"] = _lr_mod
_spec.loader.exec_module(_lr_mod)
from regression_pack_core.schemas import CoefficientRow, InterpretationFact  # noqa: E402

generate_interpretations = _lr_mod.generate_interpretations

_REG_CAVEATS = {
    "ridge": "ridge shrinks all coefficients — direction is reliable, magnitude is biased toward zero",
    "lasso": "lasso may have zeroed correlated features — present coefficients survive L1 selection",
    "elasticnet": "elasticnet combines L1 selection and L2 shrinkage — coefficients are biased toward zero",
}


def generate_regularized_interpretations(
    coefs: list[CoefficientRow],
    method: str,
    *,
    target: str = "the target",
    target_transform: str | None = None,
    feature_transforms: dict[str, str] | None = None,
    dummy_origin: dict[str, str] | None = None,
) -> list[InterpretationFact]:
    """Generate interpretations with an extra regularisation caveat."""
    facts = generate_interpretations(
        coefs,
        target=target,
        target_transform=target_transform,
        feature_transforms=feature_transforms or {},
        dummy_origin=dummy_origin or {},
    )
    reg_caveat = _REG_CAVEATS.get(method, "regularisation biases coefficients toward zero")
    se_caveat = "SE / CI / p-values are approximate (from OLS on full data — ignores regularisation bias)"
    for fact in facts:
        if reg_caveat not in fact.caveats:
            fact.caveats.insert(0, reg_caveat)
        if se_caveat not in fact.caveats:
            fact.caveats.append(se_caveat)
    return facts
