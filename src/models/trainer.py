"""Model training module for churn prediction.

Supports multiple ML algorithms with hyperparameter tuning and threshold optimization.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import joblib
import os

# Optional: XGBoost support
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. XGBoost models will not be available.")


@dataclass
class ModelResult:
    """Container for model training results."""

    model_name: str
    model: Any
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    optimal_threshold: float = 0.5
    feature_importance: Optional[pd.DataFrame] = None
    training_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding model object)."""
        return {
            "model_name": self.model_name,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "optimal_threshold": self.optimal_threshold,
            "training_time": self.training_time,
        }


class ChurnModelTrainer:
    """Train and evaluate multiple churn prediction models.

    This class provides:
    - Multiple model algorithms (Logistic Regression, Random Forest, XGBoost)
    - Cross-validation
    - Hyperparameter tuning via grid search
    - Probability threshold optimization
    - Model persistence

    Example:
        >>> trainer = ChurnModelTrainer()
        >>> results = trainer.train_all_models(X_train, y_train, X_val, y_val)
        >>> best_model = trainer.get_best_model()
    """

    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        optimize_threshold: bool = True,
        threshold_metric: str = "f1",
        n_jobs: int = -1,
    ):
        """Initialize the model trainer.

        Args:
            random_state: Random seed for reproducibility.
            cv_folds: Number of cross-validation folds.
            scoring: Metric for model selection ('roc_auc', 'f1', 'accuracy').
            optimize_threshold: Whether to find optimal probability threshold.
            threshold_metric: Metric for threshold optimization.
            n_jobs: Number of parallel jobs (-1 for all cores).
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.optimize_threshold = optimize_threshold
        self.threshold_metric = threshold_metric
        self.n_jobs = n_jobs

        # Storage for trained models
        self.results_: Dict[str, ModelResult] = {}
        self.best_model_name_: Optional[str] = None

        # Define available models
        self._setup_model_configs()

    def _setup_model_configs(self):
        """Define model configurations and hyperparameter grids."""
        self.model_configs = {
            "logistic_regression": {
                "class": LogisticRegression,
                "params": {
                    "max_iter": 1000,
                    "random_state": self.random_state,
                    "solver": "lbfgs",
                },
                "param_grid": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l2"],
                },
            },
            "random_forest": {
                "class": RandomForestClassifier,
                "params": {
                    "n_estimators": 100,
                    "random_state": self.random_state,
                    "n_jobs": self.n_jobs,
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                },
            },
            "gradient_boosting": {
                "class": GradientBoostingClassifier,
                "params": {
                    "n_estimators": 100,
                    "random_state": self.random_state,
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
            },
        }

        # Add XGBoost if available
        if HAS_XGBOOST:
            self.model_configs["xgboost"] = {
                "class": XGBClassifier,
                "params": {
                    "n_estimators": 100,
                    "random_state": self.random_state,
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                    "n_jobs": self.n_jobs,
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
            }

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.

        Returns:
            Dictionary of metric names and values.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
        }

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """Find optimal probability threshold for classification.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities for positive class.
            metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced').

        Returns:
            Optimal threshold value.
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "balanced":
                # Balance precision and recall
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def _get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Optional[pd.DataFrame]:
        """Extract feature importance from a trained model.

        Args:
            model: Trained model.
            feature_names: List of feature names.

        Returns:
            DataFrame with feature importance scores.
        """
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_).ravel()
        else:
            return None

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_pct"] = df["importance"] / df["importance"].sum() * 100

        return df

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune_hyperparams: bool = False,
        feature_names: Optional[List[str]] = None,
    ) -> ModelResult:
        """Train a single model.

        Args:
            model_name: Name of the model to train.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            tune_hyperparams: Whether to perform grid search.
            feature_names: Feature names (uses X_train columns if None).

        Returns:
            ModelResult object with training results.

        Raises:
            ValueError: If model_name is not recognized.
        """
        import time

        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")

        config = self.model_configs[model_name]
        feature_names = feature_names or list(X_train.columns)

        start_time = time.time()

        # Create base model
        model = config["class"](**config["params"])

        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # Hyperparameter tuning if requested
        if tune_hyperparams:
            grid_search = GridSearchCV(
                model,
                config["param_grid"],
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        # Fit final model
        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Get predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        # Training metrics
        train_metrics = self._calculate_metrics(
            y_train.values if hasattr(y_train, 'values') else y_train,
            y_train_pred,
            y_train_proba
        )

        # Validation metrics
        val_metrics = {}
        optimal_threshold = 0.5

        if X_val is not None and y_val is not None:
            y_val_proba = model.predict_proba(X_val)[:, 1]

            # Find optimal threshold
            if self.optimize_threshold:
                optimal_threshold = self._find_optimal_threshold(
                    y_val.values if hasattr(y_val, 'values') else y_val,
                    y_val_proba,
                    self.threshold_metric
                )

            y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
            val_metrics = self._calculate_metrics(
                y_val.values if hasattr(y_val, 'values') else y_val,
                y_val_pred,
                y_val_proba
            )

        # Get CV scores from final model
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )

        # Create result
        result = ModelResult(
            model_name=model_name,
            model=model,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            optimal_threshold=optimal_threshold,
            feature_importance=self._get_feature_importance(model, feature_names),
            training_time=training_time,
        )

        self.results_[model_name] = result
        return result

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune_hyperparams: bool = False,
        models: Optional[List[str]] = None,
    ) -> Dict[str, ModelResult]:
        """Train all available models.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            tune_hyperparams: Whether to tune hyperparameters.
            models: List of model names to train (all if None).

        Returns:
            Dictionary of model results.
        """
        models = models or list(self.model_configs.keys())

        for model_name in models:
            print(f"Training {model_name}...")
            self.train_model(
                model_name,
                X_train, y_train,
                X_val, y_val,
                tune_hyperparams
            )

        # Determine best model
        if X_val is not None and y_val is not None:
            best_score = -np.inf
            for name, result in self.results_.items():
                score = result.val_metrics.get(self.scoring, result.cv_mean)
                if score > best_score:
                    best_score = score
                    self.best_model_name_ = name
        else:
            best_score = -np.inf
            for name, result in self.results_.items():
                if result.cv_mean > best_score:
                    best_score = result.cv_mean
                    self.best_model_name_ = name

        return self.results_

    def get_best_model(self) -> Tuple[Any, str]:
        """Get the best performing model.

        Returns:
            Tuple of (model, model_name).

        Raises:
            ValueError: If no models have been trained.
        """
        if not self.results_:
            raise ValueError("No models trained. Call train_model() or train_all_models() first.")

        if self.best_model_name_ is None:
            # Find best by CV score
            best_name = max(self.results_, key=lambda x: self.results_[x].cv_mean)
            self.best_model_name_ = best_name

        return self.results_[self.best_model_name_].model, self.best_model_name_

    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary of all model results.

        Returns:
            DataFrame with model comparison metrics.
        """
        if not self.results_:
            raise ValueError("No models trained yet.")

        rows = []
        for name, result in self.results_.items():
            row = {
                "model": name,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                **{f"train_{k}": v for k, v in result.train_metrics.items()},
                **{f"val_{k}": v for k, v in result.val_metrics.items()},
                "optimal_threshold": result.optimal_threshold,
                "training_time": result.training_time,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(f"val_{self.scoring}" if f"val_{self.scoring}" in df.columns else "cv_mean", ascending=False)
        return df

    def save_model(
        self,
        model_name: str,
        filepath: str,
        include_preprocessor: bool = False,
        preprocessor: Any = None,
    ) -> str:
        """Save a trained model to disk.

        Args:
            model_name: Name of the model to save.
            filepath: Path to save the model.
            include_preprocessor: Whether to include preprocessor.
            preprocessor: Preprocessor object to save.

        Returns:
            Path to saved model.
        """
        if model_name not in self.results_:
            raise ValueError(f"Model '{model_name}' not found in trained models.")

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        save_data = {
            "model": self.results_[model_name].model,
            "model_name": model_name,
            "optimal_threshold": self.results_[model_name].optimal_threshold,
            "feature_importance": self.results_[model_name].feature_importance,
        }

        if include_preprocessor and preprocessor is not None:
            save_data["preprocessor"] = preprocessor

        joblib.dump(save_data, filepath)
        return filepath

    def load_model(self, filepath: str) -> Tuple[Any, float]:
        """Load a model from disk.

        Args:
            filepath: Path to the saved model.

        Returns:
            Tuple of (model, optimal_threshold).
        """
        data = joblib.load(filepath)

        if isinstance(data, dict):
            return data.get("model"), data.get("optimal_threshold", 0.5)
        else:
            # Legacy format
            return data, 0.5


def train_churn_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models: Optional[List[str]] = None,
    tune: bool = False,
    random_state: int = 42,
) -> Tuple[ChurnModelTrainer, pd.DataFrame]:
    """Convenience function to train churn prediction models.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        models: List of model names to train.
        tune: Whether to tune hyperparameters.
        random_state: Random seed.

    Returns:
        Tuple of (trainer, results_summary).
    """
    trainer = ChurnModelTrainer(random_state=random_state)
    trainer.train_all_models(X_train, y_train, X_val, y_val, tune, models)
    summary = trainer.get_results_summary()

    return trainer, summary


if __name__ == "__main__":
    # Test the trainer
    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    X_train = pd.DataFrame(X[:800])
    y_train = pd.Series(y[:800])
    X_val = pd.DataFrame(X[800:])
    y_val = pd.Series(y[800:])

    # Train models
    trainer = ChurnModelTrainer(random_state=42)
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)

    print("\nModel Results:")
    print(trainer.get_results_summary())

    best_model, best_name = trainer.get_best_model()
    print(f"\nBest model: {best_name}")
