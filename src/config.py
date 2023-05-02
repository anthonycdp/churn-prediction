"""Configuration settings for the churn prediction project."""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Configuration for data generation and preprocessing."""

    random_state: int = 42
    n_samples: int = 5000

    # Feature generation parameters
    numerical_features: List[str] = field(default_factory=lambda: [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "avg_call_duration",
        "num_support_tickets",
        "num_complaints",
        "data_usage_gb",
        "contract_changes",
    ])

    categorical_features: List[str] = field(default_factory=lambda: [
        "contract_type",
        "payment_method",
        "internet_service",
        "online_security",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ])

    # Churn probability base rates by segment
    base_churn_rate: float = 0.20

    # Output paths
    raw_data_path: str = "data/raw/churn_data.csv"
    processed_data_path: str = "data/processed/churn_data_processed.csv"


@dataclass
class ModelConfig:
    """Configuration for model training."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Model hyperparameters
    logistic_regression_params: dict = field(default_factory=lambda: {
        "max_iter": 1000,
        "solver": "lbfgs",
    })

    random_forest_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    })

    xgboost_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
    })

    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # Options: f1, precision, recall, balanced_accuracy


@dataclass
class CostConfig:
    """Configuration for cost analysis."""

    # Cost of false negatives (missed churners - we lose the customer)
    cost_false_negative: float = 500.0  # Lost customer lifetime value

    # Cost of false positives (unnecessary retention offers)
    cost_false_positive: float = 50.0  # Retention offer cost

    # Value of true positives (successful retention)
    retention_success_value: float = 450.0  # Saved CLV minus retention cost

    # Retention campaign effectiveness
    retention_success_rate: float = 0.30  # 30% of targeted customers stay

    # Budget constraints
    monthly_retention_budget: float = 10000.0


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    # Project paths
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir: str = "outputs"
    figures_dir: str = "outputs/figures"
    models_dir: str = "outputs/models"
    reports_dir: str = "outputs/reports"

    def __post_init__(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.output_dir, self.figures_dir, self.models_dir, self.reports_dir]:
            full_path = os.path.join(self.project_root, dir_path)
            os.makedirs(full_path, exist_ok=True)


# Default configuration instance
config = Config()
