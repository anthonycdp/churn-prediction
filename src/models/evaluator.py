"""Model evaluation module for churn prediction.

Comprehensive evaluation metrics, visualizations, and analysis tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import CalibrationDisplay, calibration_curve
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Dict[str, float]]
    threshold: float
    y_pred: np.ndarray
    y_proba: np.ndarray


class ChurnModelEvaluator:
    """Comprehensive model evaluation for churn prediction.

    This class provides:
    - Classification metrics (accuracy, precision, recall, F1, AUC)
    - Confusion matrix analysis
    - ROC and PR curves
    - Calibration analysis
    - Threshold analysis
    - Segmented evaluation by customer groups

    Example:
        >>> evaluator = ChurnModelEvaluator()
        >>> result = evaluator.evaluate(model, X_test, y_test)
        >>> evaluator.plot_roc_curve(result)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        positive_label: int = 1,
    ):
        """Initialize the evaluator.

        Args:
            threshold: Probability threshold for positive prediction.
            positive_label: Label for the positive class.
        """
        self.threshold = threshold
        self.positive_label = positive_label

    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None,
    ) -> EvaluationResult:
        """Evaluate a model on test data.

        Args:
            model: Trained model with predict/predict_proba methods.
            X: Feature DataFrame.
            y: True labels.
            threshold: Override threshold (uses instance threshold if None).

        Returns:
            EvaluationResult object with all metrics.
        """
        threshold = threshold or self.threshold

        # Get predictions
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        y_true = y.values if hasattr(y, 'values') else y

        # Calculate metrics
        metrics = self._calculate_all_metrics(y_true, y_pred, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        return EvaluationResult(
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=report,
            threshold=threshold,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    def _calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.

        Returns:
            Dictionary of metric names and values.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "brier_score": brier_score_loss(y_true, y_proba),
            "true_positives": int(np.sum((y_true == 1) & (y_pred == 1))),
            "true_negatives": int(np.sum((y_true == 0) & (y_pred == 0))),
            "false_positives": int(np.sum((y_true == 0) & (y_pred == 1))),
            "false_negatives": int(np.sum((y_true == 1) & (y_pred == 0))),
        }

        # Calculate rates
        metrics["specificity"] = (
            metrics["true_negatives"] /
            max(metrics["true_negatives"] + metrics["false_positives"], 1)
        )
        metrics["npv"] = (
            metrics["true_negatives"] /
            max(metrics["true_negatives"] + metrics["false_negatives"], 1)
        )

        # Lift score (ratio of true positive rate to base rate)
        base_rate = np.mean(y_true)
        if base_rate > 0:
            captured_rate = metrics["true_positives"] / max(metrics["true_positives"] + metrics["false_positives"], 1)
            metrics["lift"] = captured_rate / base_rate
        else:
            metrics["lift"] = 0.0

        return metrics

    def threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Analyze metrics across different thresholds.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            thresholds: Array of thresholds to evaluate.

        Returns:
            DataFrame with metrics at each threshold.
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "fp_rate": np.mean((y_true == 0) & (y_pred == 1)),
                "fn_rate": np.mean((y_true == 1) & (y_pred == 0)),
            }
            results.append(metrics)

        return pd.DataFrame(results)

    def segment_evaluation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        segment_column: str,
    ) -> pd.DataFrame:
        """Evaluate model performance by customer segment.

        Args:
            model: Trained model.
            X: Feature DataFrame.
            y: True labels.
            segment_column: Column name to segment by.

        Returns:
            DataFrame with metrics per segment.
        """
        # Exclude segment column from features for prediction
        feature_cols = [col for col in X.columns if col != segment_column]
        X_features = X[feature_cols]

        y_proba = model.predict_proba(X_features)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)
        y_true = y.values if hasattr(y, 'values') else y

        results = []
        segments = X[segment_column].unique() if segment_column in X.columns else []

        for segment in segments:
            mask = X[segment_column] == segment
            if mask.sum() == 0:
                continue

            segment_y_true = y_true[mask]
            segment_y_pred = y_pred[mask]
            segment_y_proba = y_proba[mask]

            metrics = {
                "segment": segment,
                "n_samples": mask.sum(),
                "actual_churn_rate": segment_y_true.mean(),
                "predicted_churn_rate": segment_y_pred.mean(),
                "precision": precision_score(segment_y_true, segment_y_pred, zero_division=0),
                "recall": recall_score(segment_y_true, segment_y_pred, zero_division=0),
                "f1": f1_score(segment_y_true, segment_y_pred, zero_division=0),
            }

            if len(np.unique(segment_y_true)) > 1:
                metrics["roc_auc"] = roc_auc_score(segment_y_true, segment_y_proba)
            else:
                metrics["roc_auc"] = np.nan

            results.append(metrics)

        return pd.DataFrame(results)

    def plot_confusion_matrix(
        self,
        result: EvaluationResult,
        ax: Optional[plt.Axes] = None,
        normalize: bool = False,
        cmap: str = "Blues",
    ) -> plt.Figure:
        """Plot confusion matrix.

        Args:
            result: Evaluation result object.
            ax: Matplotlib axes (creates new figure if None).
            normalize: Whether to normalize by true class.
            cmap: Colormap name.

        Returns:
            Matplotlib figure.
        """
        cm = result.confusion_matrix.copy()

        if normalize:
            cm = cm.astype(float)
            cm = cm / cm.sum(axis=1, keepdims=True)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )

        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title(f"Confusion Matrix (threshold={result.threshold:.2f})")

        return fig

    def plot_roc_curve(
        self,
        result: EvaluationResult,
        y_true: np.ndarray,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot ROC curve.

        Args:
            result: Evaluation result object.
            y_true: True labels.
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        fpr, tpr, thresholds = roc_curve(y_true, result.y_proba)
        auc = result.metrics["roc_auc"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
        ax.scatter(
            1 - result.metrics["specificity"],
            result.metrics["recall"],
            c="red",
            s=100,
            zorder=5,
            label=f"Threshold = {result.threshold:.2f}",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_precision_recall_curve(
        self,
        result: EvaluationResult,
        y_true: np.ndarray,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot Precision-Recall curve.

        Args:
            result: Evaluation result object.
            y_true: True labels.
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, result.y_proba)
        pr_auc = result.metrics["pr_auc"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})", linewidth=2)
        ax.axhline(y=np.mean(y_true), color="k", linestyle="--", label="Baseline", linewidth=1)
        ax.scatter(
            result.metrics["recall"],
            result.metrics["precision"],
            c="red",
            s=100,
            zorder=5,
            label=f"Threshold = {result.threshold:.2f}",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot metrics across thresholds.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        threshold_df = self.threshold_analysis(y_true, y_proba)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", linewidth=2)
        ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", linewidth=2)
        ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1", linewidth=2)
        ax.axvline(x=self.threshold, color="red", linestyle="--", label=f"Current ({self.threshold:.2f})")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Metrics vs. Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        return fig

    def plot_probability_distribution(
        self,
        result: EvaluationResult,
        y_true: np.ndarray,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot distribution of predicted probabilities by actual class.

        Args:
            result: Evaluation result object.
            y_true: True labels.
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Separate probabilities by class
        proba_churn = result.y_proba[y_true == 1]
        proba_no_churn = result.y_proba[y_true == 0]

        # Plot distributions
        ax.hist(proba_no_churn, bins=50, alpha=0.6, label="No Churn", density=True)
        ax.hist(proba_churn, bins=50, alpha=0.6, label="Churn", density=True)
        ax.axvline(x=result.threshold, color="red", linestyle="--", label=f"Threshold ({result.threshold:.2f})")

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Predicted Probabilities")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot calibration curve.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration.
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        ax.plot(prob_pred, prob_true, "s-", label="Model", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        return fig

    def generate_report(
        self,
        result: EvaluationResult,
        y_true: np.ndarray,
    ) -> str:
        """Generate a text summary report.

        Args:
            result: Evaluation result object.
            y_true: True labels.

        Returns:
            Formatted text report.
        """
        report = []
        report.append("=" * 60)
        report.append("CHURN MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall metrics
        report.append("OVERALL METRICS")
        report.append("-" * 40)
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                report.append(f"{metric:.<25} {value:.4f}")
            else:
                report.append(f"{metric:.<25} {value}")
        report.append("")

        # Confusion matrix
        report.append("CONFUSION MATRIX")
        report.append("-" * 40)
        cm = result.confusion_matrix
        report.append(f"              Predicted")
        report.append(f"              No    Yes")
        report.append(f"Actual No   {cm[0, 0]:5d} {cm[0, 1]:5d}")
        report.append(f"       Yes  {cm[1, 0]:5d} {cm[1, 1]:5d}")
        report.append("")

        # Key insights
        report.append("KEY INSIGHTS")
        report.append("-" * 40)
        report.append(f"Threshold used: {result.threshold:.2f}")
        report.append(f"AUC-ROC: {result.metrics['roc_auc']:.3f}")
        report.append(f"AUC-PR: {result.metrics['pr_auc']:.3f}")
        report.append(f"")
        report.append(f"Of {int(cm.sum())} customers evaluated:")
        report.append(f"  - {result.metrics['true_positives']} churners correctly identified")
        report.append(f"  - {result.metrics['false_negatives']} churners missed (costly!)")
        report.append(f"  - {result.metrics['false_positives']} non-churners incorrectly flagged")
        report.append(f"  - Lift: {result.metrics['lift']:.2f}x better than random")

        return "\n".join(report)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> Tuple[EvaluationResult, str]:
    """Convenience function to evaluate a model.

    Args:
        model: Trained model.
        X: Feature DataFrame.
        y: True labels.
        threshold: Classification threshold.

    Returns:
        Tuple of (result, report_string).
    """
    evaluator = ChurnModelEvaluator(threshold=threshold)
    result = evaluator.evaluate(model, X, y)
    report = evaluator.generate_report(result, y.values if hasattr(y, 'values') else y)

    return result, report


if __name__ == "__main__":
    # Test the evaluator
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    # Generate test data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Train a simple model
    model = LogisticRegression(random_state=42)
    model.fit(X[:800], y[:800])

    # Evaluate
    evaluator = ChurnModelEvaluator(threshold=0.5)
    result = evaluator.evaluate(model, X[800:], y[800:])

    print(evaluator.generate_report(result, y[800:].values))
