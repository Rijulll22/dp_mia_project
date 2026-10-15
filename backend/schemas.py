"""
schemas.py — Pydantic request/response models for PrivacyGuard API

All response models reflect actual computed experiment results.
No simulated or hard-coded values are returned through these schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "PrivacyGuard Differential Privacy API is running"
    version: str = "2.0.0"
    timestamp: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    datasetId: str
    filename: str
    filesize: str
    records: int
    columns: int
    columnNames: List[str]
    targetColumn: str
    missingValues: int
    previewRows: List[Dict[str, Any]]


# ─── Experiment Request ────────────────────────────────────────────────────

class PrivacyProcessRequest(BaseModel):
    datasetId: Optional[str] = None
    epsilon: float = Field(
        default=1.0, ge=0.01, le=20.0,
        description="Record-level privacy budget (total epsilon for one record)"
    )
    delta: Optional[float] = Field(
        default=0.00001, ge=1e-9, le=0.1,
        description="Privacy parameter delta (used for Gaussian mechanism only)"
    )
    sensitivity: float = Field(
        default=1.0, ge=0.01, le=10.0,
        description="Global sensitivity of a single feature (1.0 after MinMax normalization)"
    )
    mechanism: str = Field(
        default="laplace_clipped",
        description="DP mechanism: laplace_clipped, plain_laplace, gaussian, non_private"
    )
    mechanism_type: Optional[str] = Field(
        default="laplace_clipped",
        description="Alias for mechanism"
    )
    classifier_type: Optional[str] = Field(
        default="logistic_regression",
        description="Classifier: logistic_regression, random_forest"
    )
    seed: Optional[int] = Field(default=42, description="Random seed")
    n_shadow: Optional[int] = Field(
        default=5, ge=2, le=10,
        description="Number of shadow models for MIA (default 5)"
    )


class ExperimentRunRequest(BaseModel):
    """Request for running a batch of experiments (multiple seeds × epsilon grid)."""
    mode: str = Field(
        default="fast",
        description="'fast' (2 seeds, 4 epsilon) or 'full' (10 seeds, 6 epsilon)"
    )
    epsilon_grid: Optional[List[float]] = Field(
        default=None,
        description="Custom epsilon values. If None, uses mode defaults."
    )
    seeds: Optional[List[int]] = Field(
        default=None,
        description="Custom seeds. If None, uses mode defaults."
    )
    mechanisms: Optional[List[str]] = Field(
        default=None,
        description="Mechanisms to test. Default: all three."
    )
    classifier_types: Optional[List[str]] = Field(
        default=None,
        description="Classifiers to test. Default: both."
    )
    n_shadow: Optional[int] = Field(default=5, description="Shadow models per MIA")


# ─── Metric Models ─────────────────────────────────────────────────────────

class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class UtilityMetrics(BaseModel):
    """All utility metrics with fraud-class breakdown."""
    # Overall metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    rocAuc: Optional[float] = None
    prAuc: Optional[float] = None            # Precision-Recall AUC (important for imbalanced)
    # Fraud class (minority class = 1)
    fraud_precision: Optional[float] = None
    fraud_recall: Optional[float] = None
    fraud_f1: Optional[float] = None
    fraud_support: Optional[int] = None
    confusionMatrix: Optional[ConfusionMatrix] = None
    # Legacy fields (kept for schema compat)
    miaRisk: Optional[float] = None         # = attack_accuracy (from actual MIA)
    reconRisk: Optional[float] = None       # DEPRECATED — always None


class MIAMetrics(BaseModel):
    """Genuine shadow-model MIA metrics. No clipping applied."""
    attack_accuracy: Optional[float] = None
    attack_auc: Optional[float] = None
    attacker_advantage: Optional[float] = None  # TPR - FPR
    tpr_at_1fpr: Optional[float] = None         # TPR @ 1% FPR
    tpr_at_01fpr: Optional[float] = None        # TPR @ 0.1% FPR
    n_members_evaluated: Optional[int] = None
    n_nonmembers_evaluated: Optional[int] = None
    error: Optional[str] = None                 # Set if MIA failed


class EpsilonAccounting(BaseModel):
    """Correct epsilon composition under basic sequential composition."""
    epsilon_record: float    # Total record-level epsilon (input from user)
    epsilon_feature: float   # Per-feature epsilon = epsilon_record / feature_count
    feature_count: int       # Number of numerical features perturbed
    composition_method: str  # "basic sequential composition"


# ─── Tradeoff Curve ───────────────────────────────────────────────────────

class TradeoffPoint(BaseModel):
    """Single point on the privacy-utility tradeoff curve from real experiments."""
    epsilon: float
    epsilon_feature: Optional[float] = None
    # Utility
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    rocAuc: Optional[float] = None
    prAuc: Optional[float] = None
    fraud_f1: Optional[float] = None
    fraud_recall: Optional[float] = None
    # MIA (from actual shadow attack, not formula)
    miaAccuracy: Optional[float] = None
    miaAuc: Optional[float] = None
    attackerAdvantage: Optional[float] = None
    tprAt1Fpr: Optional[float] = None
    # Legacy
    miaRisk: Optional[float] = None
    reconRisk: Optional[float] = None


# ─── Full Response ─────────────────────────────────────────────────────────

class PrivacyProcessResponse(BaseModel):
    success: bool
    experimentId: str
    is_simulated: bool = False  # Always False for real experiments

    # DP configuration
    epsilon: float              # Record-level epsilon
    delta: float
    sensitivity: float
    mechanism: str
    mechanism_type: Optional[str] = None
    mechanism_label: Optional[str] = None  # Human-readable mechanism name
    classifier_type: Optional[str] = None
    noiseScale: float

    # Epsilon accounting
    epsilon_accounting: Optional[EpsilonAccounting] = None

    # Dataset split metadata
    split_seed: Optional[int] = None
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    fraud_count_train: Optional[int] = None
    fraud_count_test: Optional[int] = None
    feature_list: Optional[List[str]] = None

    # Utility metrics
    metrics: UtilityMetrics

    # MIA results
    mia_result: Optional[MIAMetrics] = None
    mia_per_class: Optional[Dict[str, Any]] = None

    # Privacy-utility tradeoff curve
    tradeoffCurve: List[TradeoffPoint] = []
    pareto_curve: Optional[List[Dict[str, Any]]] = None

    # Baselines
    baselines: Optional[Dict[str, Any]] = None

    # Model comparison
    model_comparison: Optional[Dict[str, Any]] = None

    # Other
    processed_rows: Optional[int] = None
    relative_utility_loss: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None

    # Legacy compat
    mia_risk: Optional[Dict[str, Any]] = None
    reconstruction_risk: Optional[Dict[str, Any]] = None
    empirical_attacks: Optional[Dict[str, Any]] = None


class ExperimentListItem(BaseModel):
    """Summary of an experiment for the history list."""
    id: str
    dataset: str
    epsilon: float
    delta: Optional[float] = None
    mechanism: str
    classifier_type: Optional[str] = None
    date: str
    timestamp: Optional[str] = None
    # Utility
    accuracy: Optional[str] = None   # formatted as "xx.x%"
    precision: Optional[str] = None
    recall: Optional[str] = None
    f1: Optional[str] = None
    rocAuc: Optional[str] = None
    prAuc: Optional[str] = None
    # Fraud class
    fraud_f1: Optional[str] = None
    fraud_recall: Optional[str] = None
    # MIA
    mia_accuracy: Optional[str] = None
    mia_auc: Optional[str] = None
    attacker_advantage: Optional[str] = None
    # Status
    privacyRisk: Optional[str] = None
    status: str = "Completed"
    hasDataset: bool = False
    hasModel: bool = False
    is_simulated: bool = False
