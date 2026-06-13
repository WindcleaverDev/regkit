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
        "logistic_marginal",
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
    high_leverage: list[InfluencePoint]  # leverage > 2*(k+1)/n, top entries by leverage
    cooks_d_outliers: list[InfluencePoint]  # Cook's D > 4/n, top entries by Cook's D
    n_high_leverage: int = 0  # total flagged count (lists above may be truncated)
    n_cooks_d_outliers: int = 0
    truncated: bool = False  # True if either list was capped
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


# ─── Logistic regression schemas ─────────────────────────────────────────────


class OddsRatioRow(BaseModel):
    """One row in a logistic regression coefficient table."""

    feature: str
    log_odds_coefficient: float  # β
    odds_ratio: float  # exp(β)
    std_error: float
    z_stat: float
    p_value: float
    ci_lower_log_odds: float
    ci_upper_log_odds: float
    ci_lower_odds_ratio: float
    ci_upper_odds_ratio: float


class MarginalEffect(BaseModel):
    """Average marginal effect — average change in predicted probability
    when this feature increases by one unit (or switches to 1 for dummies)."""

    feature: str
    ame: float  # average marginal effect, on probability scale
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float


class ROCData(BaseModel):
    fpr: list[float]  # downsampled to ~100 points for plotting
    tpr: list[float]
    thresholds: list[float]
    auc: float


class CalibrationData(BaseModel):
    """Reliability diagram — predicted probability bins vs observed frequency."""

    bin_centers: list[float]  # ~10 bins
    observed_frequencies: list[float]
    bin_counts: list[int]
    brier_score: float


class ClassificationStats(BaseModel):
    accuracy: float
    balanced_accuracy: float
    precision: float  # for positive class
    recall: float  # for positive class
    f1: float
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]
    threshold: float  # decision threshold used (default 0.5)
    n_observations: int
    class_balance: float  # positive class rate in training data


class LogisticRegressionReport(BaseModel):
    """Complete structured output of the logistic-regression skill."""

    fit_statistics: dict  # n_obs, n_features, log_likelihood, ll_null, pseudo_r_squared, aic, bic, converged, n_iterations
    coefficients: list[OddsRatioRow]
    marginal_effects: list[MarginalEffect]
    interpretations: list[InterpretationFact]
    classification_stats: ClassificationStats
    roc: ROCData
    calibration: CalibrationData
    headline: str
    target_name: str
    positive_class: str | int  # which class is "1"
    feature_transforms: dict[str, str] = Field(default_factory=dict)
    robust_se_used: str | None = None
    flags: list[Flag] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    report_html_path: str | None = None


# ─── Regularised regression schemas ──────────────────────────────────────────


class RegularizationPath(BaseModel):
    """Coefficient path along α — how each feature's coefficient shrinks
    as regularisation strength increases."""

    alphas: list[float]  # log-spaced
    feature_names: list[str]
    coefficients: list[list[float]]  # coefficients[i][j] = β for alphas[i], feature_names[j]


class CVCurve(BaseModel):
    """Cross-validation score as α varies."""

    alphas: list[float]
    mean_scores: list[float]  # mean across folds (R² or neg MSE)
    std_scores: list[float]
    scoring: Literal["r2", "neg_mean_squared_error", "neg_root_mean_squared_error"]
    selected_alpha: float  # minimises CV error
    alpha_1se: float | None = None  # one-standard-error rule (more parsimonious)


class FeatureSelection(BaseModel):
    """For Lasso/ElasticNet — which features survived at the chosen α."""

    n_selected: int
    n_dropped: int
    selected_features: list[str]
    dropped_features: list[str]


class RegularizedRegressionReport(BaseModel):
    method: Literal["ridge", "lasso", "elasticnet"]
    selected_alpha: float
    selected_l1_ratio: float | None = None  # for elasticnet
    fit_statistics: FitStatistics
    coefficients: list[CoefficientRow]  # at the chosen α
    path: RegularizationPath
    cv_curve: CVCurve
    feature_selection: FeatureSelection | None = None  # populated for lasso/elasticnet
    interpretations: list[InterpretationFact]
    headline: str
    comparison_to_ols: dict | None = None  # {ols_r2, regularised_r2, n_shrunk, n_dropped}
    flags: list[Flag] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    report_html_path: str | None = None


# ─── Model comparison schemas ─────────────────────────────────────────────────


class ModelEntry(BaseModel):
    """One model's summary, suitable for cross-model comparison."""

    name: str  # human-readable label
    source_report_path: str  # path to the original report.json
    family: Literal["linear", "logistic", "ridge", "lasso", "elasticnet"]
    n_observations: int
    n_features: int
    fit_quality_primary: float  # adj_r² for linear family, pseudo_r² for logistic
    aic: float | None = None
    bic: float | None = None
    cv_score_mean: float | None = None
    cv_score_std: float | None = None
    notes: list[str] = Field(default_factory=list)  # e.g. "robust SE", "log-target"


class LRTestResult(BaseModel):
    """Likelihood-ratio test for nested model pairs."""

    nested_model: str  # name of the simpler model
    full_model: str  # name of the more complex model
    likelihood_ratio: float
    df: int
    p_value: float
    conclusion: str  # e.g. "Full model significantly better (p < 0.01)"


class AkaikeWeights(BaseModel):
    """For non-nested comparison: AIC differences and Akaike weights."""

    model_names: list[str]
    delta_aic: list[float]  # Δ_i = AIC_i - AIC_min
    weights: list[float]  # exp(-Δ/2) normalised


class ComparisonVerdict(BaseModel):
    overall: Literal[
        "clear_winner",
        "competitive_tie",
        "complementary_strengths",
        "all_inadequate",
    ]
    recommended_model: str | None = None
    headline: str
    rationale: str


class ModelComparisonReport(BaseModel):
    models: list[ModelEntry]
    lr_tests: list[LRTestResult] = Field(default_factory=list)
    akaike_weights: AkaikeWeights | None = None
    verdict: ComparisonVerdict
    flags: list[Flag] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    report_html_path: str | None = None
