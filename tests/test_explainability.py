"""Tests for SHAP explainability module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import ChurnDataGenerator
from src.data.preprocessor import ChurnDataPreprocessor


# Check if SHAP is available
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP not installed")
class TestSHAPAnalyzer:
    """Test cases for SHAPAnalyzer."""

    @pytest.fixture
    def trained_model_data(self):
        """Create a trained model with data for testing."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=300)

        preprocessor = ChurnDataPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test

    def test_analyzer_initialization(self, trained_model_data):
        """Test analyzer initializes correctly."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(
            model, X_train.head(50),
            model_type="tree",
            random_state=42
        )

        assert analyzer.model is not None
        assert analyzer.explainer_ is not None

    def test_compute_shap_values(self, trained_model_data):
        """Test computing SHAP values."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)
        shap_values = analyzer.compute_shap_values(X_test.head(20))

        assert shap_values is not None
        assert shap_values.shape[0] == 20

    def test_get_feature_importance(self, trained_model_data):
        """Test getting feature importance."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)
        importance = analyzer.get_feature_importance(X_test.head(20))

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_explain_prediction(self, trained_model_data):
        """Test explaining a single prediction."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)
        explanation = analyzer.explain_prediction(X_test.iloc[0])

        assert "base_value" in explanation
        assert "predicted_probability" in explanation
        assert "top_features" in explanation

    def test_explain_prediction_top_n(self, trained_model_data):
        """Test explaining prediction with custom top_n."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)
        explanation = analyzer.explain_prediction(X_test.iloc[0], top_n=5)

        assert len(explanation["top_features"]) <= 5

    def test_features_increasing_decreasing_churn(self, trained_model_data):
        """Test getting features that increase/decrease churn."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)
        explanation = analyzer.explain_prediction(X_test.iloc[0])

        assert "features_increasing_churn" in explanation
        assert "features_decreasing_churn" in explanation

    def test_get_high_risk_customers(self, trained_model_data):
        """Test identifying high risk customers."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, y_test = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)

        y_proba = model.predict_proba(X_test)[:, 1]
        high_risk = analyzer.get_high_risk_customers(
            X_test, y_proba, top_pct=0.1
        )

        assert isinstance(high_risk, pd.DataFrame)
        assert "churn_probability" in high_risk.columns
        assert "top_feature_1" in high_risk.columns

    def test_plot_summary(self, trained_model_data):
        """Test plotting SHAP summary."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)

        fig = analyzer.plot_summary(
            X_test.head(30),
            max_display=10,
            show=False
        )

        assert fig is not None

    def test_plot_bar(self, trained_model_data):
        """Test plotting feature importance bar chart."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, X_test, _, _ = trained_model_data

        analyzer = SHAPAnalyzer(model, X_train.head(50), random_state=42)

        fig = analyzer.plot_bar(
            X_test.head(30),
            max_display=10,
            show=False
        )

        assert fig is not None

    def test_model_type_auto_detection(self, trained_model_data):
        """Test auto-detection of model type."""
        from src.explainability.shap_analyzer import SHAPAnalyzer

        model, X_train, _, _, _ = trained_model_data

        # Random Forest should be detected as tree model
        analyzer = SHAPAnalyzer(
            model, X_train.head(50),
            model_type="auto",
            random_state=42
        )

        assert analyzer.model_type == "tree"

    def test_linear_model(self):
        """Test with linear model."""
        from src.explainability.shap_analyzer import SHAPAnalyzer
        from sklearn.linear_model import LogisticRegression

        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=200)

        preprocessor = ChurnDataPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        analyzer = SHAPAnalyzer(
            model, X_train.head(50),
            model_type="linear",
            random_state=42
        )

        importance = analyzer.get_feature_importance(X_test.head(20))

        assert isinstance(importance, pd.DataFrame)


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP not installed")
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_analyze_model(self):
        """Test analyze_model function."""
        from src.explainability.shap_analyzer import analyze_model

        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=200)

        preprocessor = ChurnDataPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        analyzer, importance = analyze_model(
            model, X_train.head(50), X_test.head(20)
        )

        assert analyzer is not None
        assert isinstance(importance, pd.DataFrame)


class TestSHAPNotInstalled:
    """Test behavior when SHAP is not installed."""

    @pytest.mark.skipif(HAS_SHAP, reason="SHAP is installed")
    def test_import_warning(self):
        """Test that import warning is raised when SHAP not installed."""
        # This test only runs when SHAP is NOT installed
        # The module should still import but show a warning
        import importlib
        import sys

        # Remove cached module if exists
        if "src.explainability.shap_analyzer" in sys.modules:
            del sys.modules["src.explainability.shap_analyzer"]

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from src.explainability.shap_analyzer import SHAPAnalyzer

            # Should have warning about SHAP not installed
            assert any("SHAP not installed" in str(x.message) for x in w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
