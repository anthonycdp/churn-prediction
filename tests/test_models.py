"""Tests for model training and evaluation modules."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import ChurnDataGenerator
from src.data.preprocessor import ChurnDataPreprocessor
from src.models.trainer import ChurnModelTrainer, train_churn_models
from src.models.evaluator import ChurnModelEvaluator, evaluate_model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    generator = ChurnDataGenerator(random_state=42)
    return generator.generate(n_samples=500)


@pytest.fixture
def preprocessed_data(sample_data):
    """Create preprocessed data for testing."""
    preprocessor = ChurnDataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)
    return X_train, X_test, y_train, y_test


class TestChurnModelTrainer:
    """Test cases for ChurnModelTrainer."""

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        trainer = ChurnModelTrainer(random_state=42)
        assert trainer.random_state == 42

    def test_train_single_model(self, preprocessed_data):
        """Test training a single model."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        result = trainer.train_model(
            "logistic_regression",
            X_train, y_train,
            X_test, y_test
        )

        assert result.model_name == "logistic_regression"
        assert result.model is not None
        assert result.cv_mean > 0

    def test_train_all_models(self, preprocessed_data):
        """Test training all available models."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        results = trainer.train_all_models(
            X_train, y_train,
            X_test, y_test,
            models=["logistic_regression", "random_forest"]
        )

        assert len(results) == 2
        assert "logistic_regression" in results
        assert "random_forest" in results

    def test_get_best_model(self, preprocessed_data):
        """Test getting the best model."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        trainer.train_all_models(
            X_train, y_train,
            X_test, y_test,
            models=["logistic_regression", "random_forest"]
        )

        best_model, best_name = trainer.get_best_model()

        assert best_model is not None
        assert best_name in ["logistic_regression", "random_forest"]

    def test_get_best_model_without_training_raises(self):
        """Test that getting best model without training raises error."""
        trainer = ChurnModelTrainer()

        with pytest.raises(ValueError, match="No models trained"):
            trainer.get_best_model()

    def test_get_results_summary(self, preprocessed_data):
        """Test getting results summary."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        trainer.train_all_models(
            X_train, y_train,
            X_test, y_test,
            models=["logistic_regression"]
        )

        summary = trainer.get_results_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1
        assert "model" in summary.columns

    def test_results_summary_without_training_raises(self):
        """Test that getting summary without training raises error."""
        trainer = ChurnModelTrainer()

        with pytest.raises(ValueError, match="No models trained"):
            trainer.get_results_summary()

    def test_train_unknown_model_raises(self, preprocessed_data):
        """Test that training unknown model raises error."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer()

        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train_model("unknown_model", X_train, y_train)

    def test_cv_scores_shape(self, preprocessed_data):
        """Test that CV scores have correct shape."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(cv_folds=5, random_state=42)
        result = trainer.train_model(
            "logistic_regression",
            X_train, y_train
        )

        assert len(result.cv_scores) == 5

    def test_train_metrics_present(self, preprocessed_data):
        """Test that training metrics are present."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        result = trainer.train_model(
            "logistic_regression",
            X_train, y_train,
            X_test, y_test
        )

        assert "accuracy" in result.train_metrics
        assert "roc_auc" in result.train_metrics
        assert "f1" in result.train_metrics

    def test_val_metrics_present(self, preprocessed_data):
        """Test that validation metrics are present."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        result = trainer.train_model(
            "logistic_regression",
            X_train, y_train,
            X_test, y_test
        )

        assert "accuracy" in result.val_metrics
        assert "roc_auc" in result.val_metrics
        assert "f1" in result.val_metrics

    def test_optimal_threshold_found(self, preprocessed_data):
        """Test that optimal threshold is found."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(
            optimize_threshold=True,
            random_state=42
        )
        result = trainer.train_model(
            "logistic_regression",
            X_train, y_train,
            X_test, y_test
        )

        assert 0 < result.optimal_threshold < 1

    def test_feature_importance_extracted(self, preprocessed_data):
        """Test that feature importance is extracted."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        result = trainer.train_model(
            "random_forest",
            X_train, y_train
        )

        assert result.feature_importance is not None
        assert "feature" in result.feature_importance.columns
        assert "importance" in result.feature_importance.columns

    def test_save_and_load_model(self, preprocessed_data, tmp_path):
        """Test saving and loading models."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        trainer.train_model(
            "logistic_regression",
            X_train, y_train
        )

        filepath = str(tmp_path / "model.joblib")
        trainer.save_model("logistic_regression", filepath)

        loaded_model, threshold = trainer.load_model(filepath)

        assert loaded_model is not None
        assert isinstance(loaded_model, LogisticRegression)

    def test_model_result_to_dict(self, preprocessed_data):
        """Test ModelResult to_dict method."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer = ChurnModelTrainer(random_state=42)
        result = trainer.train_model(
            "logistic_regression",
            X_train, y_train
        )

        result_dict = result.to_dict()

        assert "model_name" in result_dict
        assert "cv_mean" in result_dict
        # Model object should not be in dict
        assert "model" not in result_dict


class TestChurnModelEvaluator:
    """Test cases for ChurnModelEvaluator."""

    @pytest.fixture
    def trained_model(self, preprocessed_data):
        """Create a trained model for testing."""
        X_train, X_test, y_train, y_test = preprocessed_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        evaluator = ChurnModelEvaluator(threshold=0.5)
        assert evaluator.threshold == 0.5

    def test_evaluate_model(self, trained_model):
        """Test evaluating a model."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator(threshold=0.5)
        result = evaluator.evaluate(model, X_test, y_test)

        assert result.metrics is not None
        assert result.confusion_matrix is not None
        assert result.threshold == 0.5

    def test_metrics_calculated(self, trained_model):
        """Test that all metrics are calculated."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        required_metrics = [
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc"
        ]

        for metric in required_metrics:
            assert metric in result.metrics

    def test_confusion_matrix_shape(self, trained_model):
        """Test confusion matrix has correct shape."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        assert result.confusion_matrix.shape == (2, 2)

    def test_classification_report(self, trained_model):
        """Test classification report is generated."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        assert result.classification_report is not None
        assert "0" in result.classification_report or "False" in result.classification_report

    def test_threshold_analysis(self, trained_model):
        """Test threshold analysis."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        threshold_df = evaluator.threshold_analysis(
            y_test.values, result.y_proba
        )

        assert isinstance(threshold_df, pd.DataFrame)
        assert "threshold" in threshold_df.columns
        assert "precision" in threshold_df.columns
        assert "recall" in threshold_df.columns

    def test_segment_evaluation(self, trained_model, preprocessed_data):
        """Test segment evaluation."""
        model, X_test, y_test = trained_model
        X_train, _, _, _ = preprocessed_data

        # Add a segment column
        X_test_with_segment = X_test.copy()
        X_test_with_segment["segment"] = np.random.choice(["A", "B"], len(X_test))

        evaluator = ChurnModelEvaluator()
        segment_df = evaluator.segment_evaluation(
            model, X_test_with_segment, y_test, "segment"
        )

        assert isinstance(segment_df, pd.DataFrame)
        assert "segment" in segment_df.columns

    def test_generate_report(self, trained_model):
        """Test generating text report."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        report = evaluator.generate_report(result, y_test.values)

        assert isinstance(report, str)
        assert "ACCURACY" in report.upper() or "accuracy" in report

    def test_plot_confusion_matrix(self, trained_model):
        """Test plotting confusion matrix."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        fig = evaluator.plot_confusion_matrix(result)

        assert fig is not None

    def test_plot_roc_curve(self, trained_model):
        """Test plotting ROC curve."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        fig = evaluator.plot_roc_curve(result, y_test.values)

        assert fig is not None

    def test_plot_precision_recall_curve(self, trained_model):
        """Test plotting PR curve."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        fig = evaluator.plot_precision_recall_curve(result, y_test.values)

        assert fig is not None

    def test_metrics_values_reasonable(self, trained_model):
        """Test that metric values are in reasonable ranges."""
        model, X_test, y_test = trained_model

        evaluator = ChurnModelEvaluator()
        result = evaluator.evaluate(model, X_test, y_test)

        # All metrics should be between 0 and 1
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0 <= result.metrics[metric] <= 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_train_churn_models(self, preprocessed_data):
        """Test train_churn_models function."""
        X_train, X_test, y_train, y_test = preprocessed_data

        trainer, summary = train_churn_models(
            X_train, y_train, X_test, y_test,
            models=["logistic_regression"],
            random_state=42
        )

        assert trainer is not None
        assert isinstance(summary, pd.DataFrame)

    def test_evaluate_model(self, preprocessed_data):
        """Test evaluate_model function."""
        X_train, X_test, y_train, y_test = preprocessed_data

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        result, report = evaluate_model(model, X_test, y_test)

        assert result is not None
        assert isinstance(report, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
