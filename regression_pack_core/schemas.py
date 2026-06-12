"""All Pydantic models used across the regression pack.

Every cross-script payload validates against one of these models; raw dicts
are never passed between scripts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ─── Shared primitives ──────────────────────────────────────────────────────


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    HIGH = "high"


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class Flag(BaseModel):
    """A flagged issue raised by any skill in the pack."""

    severity: Severity
    code: str  # e.g. "TARGET_SKEW", "HIGH_VIF", "HETEROSCEDASTICITY"
    message: str  # short human-readable summary
    detail: dict | None = None  # arbitrary supporting data


class Recommendation(BaseModel):
    """An actionable suggestion produced by a skill."""

    action: str  # e.g. "log_transform", "use_robust_se", "inspect_row"
    target: str | None = None  # feature name, row index, model element
    reason: str
    priority: Severity = Severity.INFO


# ─── Linear regression schemas ──────────────────────────────────────────────


class CoefficientRow(BaseModel):
    feature: str
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    standardized_coefficient: float | None = None  # None for intercept


class FitStatistics(BaseModel):
    n_observations: int
    n_features: int  # excludes intercept
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    aic: float
    bic: float
    log_likelihood: float
    residual_std_error: float
    df_residuals: int


class InterpretationFact(BaseModel):
    """One coefficient interpretation, ready for Claude to verbalise.

    Distinguishing this from prose is the central design choice: this skill
    produces facts; Claude produces prose. Same facts, different prose for
    different audiences.
    """

    feature: str
    coefficient: float
    interpretation_type: Literal[
        "linear_linear",
        "log_log_elasticity",
        "log_linear_semi_elasticity",
        "linear_log",
        "binary_dummy",
        "categorical_dummy",
        "polynomial_term",
        "interaction_term",
    ]
    fact: str  # canonical claim
    confidence: Literal["high", "medium", "low"]  # based on p-value + CI width
    caveats: list[str] = Field(default_factory=list)


class FitQuality(BaseModel):
    r_squared: float
    adj_r_squared: float
    interpretation: Literal["weak", "moderate", "strong", "very_strong"]


class LinearRegressionReport(BaseModel):
    """The complete structured output of the linear-regression skill."""

    fit_statistics: FitStatistics
    coefficients: list[CoefficientRow]
    interpretations: list[InterpretationFact]
    fit_quality: FitQuality
    headline: str  # one-sentence summary Claude leads with
    target: str | None = None  # target column name (needed by diagnostics)
    features: list[str] = Field(default_factory=list)  # original pre-encoding feature columns
    target_transform: str | None = None  # e.g. "log"
    feature_transforms: dict[str, str] = Field(default_factory=dict)
    robust_se_used: str | None = None  # None | "HC0" | "HC1" | "HC2" | "HC3"
    flags: list[Flag] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    report_html_path: str | None = None


# ─── Diagnostics schemas ────────────────────────────────────────────────────


class AssumptionCheck(BaseModel):
    """One assumption test result."""

    name: Literal[
        "linearity",
        "homoscedasticity",
        "normality_of_residuals",
        "independence",
        "no_multicollinearity",
    ]
    status: Status
    test_name: str  # e.g. "Breusch-Pagan", "Durbin-Watson"
    statistic: float | None = None
    p_value: float | None = None
    evidence: str  # human-readable summary of the test result
    detail: dict | None = None


class InfluencePoint(BaseModel):
    row_index: int
    leverage: float
    cooks_distance: float
    studentized_residual: float
    dffits: float | None = None


class InfluenceReport(BaseModel):
    high_leverage: list[InfluencePoint]  # leverage > 2*(k+1)/n
    cooks_d_outliers: list[InfluencePoint]  # Cook's D > 4/n
    summary: str  # human-readable summary
    max_cooks_d: float
    max_leverage: float


class BiasVarianceReport(BaseModel):
    train_r_squared: float
    test_r_squared: float
    cv_r_squared_mean: float
    cv_r_squared_std: float
    gap: float  # train_r2 - test_r2
    verdict: Literal["high_bias", "good_fit", "high_variance", "inconsistent"]
    evidence: str
    learning_curve: dict | None = None  # {"sizes": [...], "train_scores": [...], "test_scores": [...]}


class DiagnosticsVerdict(BaseModel):
    overall: Literal["clean", "usable_with_caveats", "problematic", "unreliable"]
    top_issues: list[str]  # ordered codes from flags, most actionable first
    headline: str  # one-sentence verdict for Claude to lead with


class DiagnosticsReport(BaseModel):
    """The complete structured output of the diagnostics skill."""

    assumptions: list[AssumptionCheck]
    influence: InfluenceReport
    bias_variance: BiasVarianceReport
    verdict: DiagnosticsVerdict
    flags: list[Flag] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    report_html_path: str | None = None


# ─── Pre-analysis schemas (for phase 2; locked down now) ────────────────────


class TargetAudit(BaseModel):
    name: str
    type: Literal["continuous", "binary", "count", "categorical"]
    n_missing: int
    skewness: float | None = None
    kurtosis: float | None = None
    outlier_count: int | None = None
    recommendations: list[str] = Field(default_factory=list)


class FeatureAudit(BaseModel):
    name: str
    type: Literal["continuous", "binary", "categorical", "ordinal"]
    n_missing: int
    missing_pct: float
    n_unique: int
    flags: list[str] = Field(default_factory=list)


class PreAnalysisReport(BaseModel):
    n_samples: int
    target: TargetAudit
    features: list[FeatureAudit]
    multicollinearity: dict  # {"max_vif": float, "flagged": [str], "matrix": [[...]]}
    suspected_nonlinearity: list[str]
    flags: list[Flag] = Field(default_factory=list)
    modeling_recommendations: dict  # {"transform_target": str|None, "consider_polynomial": [str], ...}
    report_html_path: str | None = None
