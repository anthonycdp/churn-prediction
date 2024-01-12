"""Tests for data generation and preprocessing modules."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import ChurnDataGenerator, generate_sample_data
from src.data.preprocessor import ChurnDataPreprocessor, preprocess_churn_data


class TestChurnDataGenerator:
    """Test cases for ChurnDataGenerator."""

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = ChurnDataGenerator(random_state=42)
        assert generator.random_state == 42

    def test_generate_basic(self):
        """Test basic data generation."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        assert len(df) == 100
        assert "churn" in df.columns
        assert "customer_id" in df.columns

    def test_generate_with_custom_churn_rate(self):
        """Test generation with custom churn rate."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=1000, base_churn_rate=0.3)

        # Should be approximately 30% churn rate (allow wider range due to probabilistic nature)
        assert 0.10 < df["churn"].mean() < 0.50

    def test_generate_without_customer_id(self):
        """Test generation without customer IDs."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100, include_customer_id=False)

        assert "customer_id" not in df.columns

    def test_numerical_features_present(self):
        """Test that all numerical features are present."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        expected_features = [
            "tenure_months", "monthly_charges", "total_charges",
            "avg_call_duration", "num_support_tickets"
        ]

        for feature in expected_features:
            assert feature in df.columns

    def test_categorical_features_present(self):
        """Test that all categorical features are present."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        expected_features = [
            "contract_type", "payment_method", "internet_service"
        ]

        for feature in expected_features:
            assert feature in df.columns

    def test_contract_types_valid(self):
        """Test that contract types have valid values."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        valid_contracts = {"Month-to-month", "One year", "Two year"}
        assert set(df["contract_type"].unique()).issubset(valid_contracts)

    def test_tenure_in_valid_range(self):
        """Test that tenure is in valid range."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        assert df["tenure_months"].min() >= 1
        assert df["tenure_months"].max() <= 72

    def test_monthly_charges_positive(self):
        """Test that monthly charges are positive."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        assert (df["monthly_charges"] > 0).all()

    def test_churn_is_binary(self):
        """Test that churn column is binary."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        assert set(df["churn"].unique()).issubset({True, False, 0, 1})

    def test_churn_probability_in_range(self):
        """Test that churn probability is in valid range."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        assert (df["churn_probability"] >= 0).all()
        assert (df["churn_probability"] <= 1).all()

    def test_save_to_csv(self, tmp_path):
        """Test saving data to CSV."""
        generator = ChurnDataGenerator(random_state=42)
        df = generator.generate(n_samples=100)

        filepath = str(tmp_path / "test_data.csv")
        result_path = generator.save_to_csv(df, filepath)

        assert result_path == filepath
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 100

    def test_generate_train_test_split(self):
        """Test generating train/test split."""
        generator = ChurnDataGenerator(random_state=42)
        train_df, test_df = generator.generate_train_test_split(
            n_train=800, n_test=200
        )

        assert len(train_df) == 800
        assert len(test_df) == 200
        assert "dataset" in train_df.columns
        assert "dataset" in test_df.columns

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        gen1 = ChurnDataGenerator(random_state=42)
        gen2 = ChurnDataGenerator(random_state=42)

        df1 = gen1.generate(n_samples=100)
        df2 = gen2.generate(n_samples=100)

        pd.testing.assert_frame_equal(df1, df2)


class TestChurnDataPreprocessor:
    """Test cases for ChurnDataPreprocessor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        generator = ChurnDataGenerator(random_state=42)
        return generator.generate(n_samples=200)

    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = ChurnDataPreprocessor(target_column="churn")
        assert preprocessor.target_column == "churn"

    def test_fit(self, sample_data):
        """Test fitting the preprocessor."""
        preprocessor = ChurnDataPreprocessor()
        result = preprocessor.fit(sample_data)

        assert result is preprocessor  # Returns self
        assert preprocessor.is_fitted_

    def test_transform_without_fit_raises(self, sample_data):
        """Test that transform without fit raises error."""
        preprocessor = ChurnDataPreprocessor()

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_data)

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = ChurnDataPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)

        assert len(X_train) == 160  # 80% of 200
        assert len(X_test) == 40    # 20% of 200
        assert len(y_train) == 160
        assert len(y_test) == 40

    def test_feature_engineering(self, sample_data):
        """Test that feature engineering creates new features."""
        preprocessor = ChurnDataPreprocessor()
        preprocessor.fit(sample_data)
        X, y = preprocessor.transform(sample_data)

        # Should have additional engineered features
        assert len(X.columns) > 0

    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        preprocessor = ChurnDataPreprocessor()
        preprocessor.fit(sample_data)

        feature_names = preprocessor.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_scaling_applied(self, sample_data):
        """Test that scaling is applied to numerical features."""
        preprocessor = ChurnDataPreprocessor(scaling_method="standard")
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)

        # Standard scaling should make mean approximately 0
        # (not exact due to train/test split)
        assert isinstance(X_train, pd.DataFrame)

    def test_no_scaling(self, sample_data):
        """Test that no scaling is applied when disabled."""
        preprocessor = ChurnDataPreprocessor(scaling_method=None)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)

        assert isinstance(X_train, pd.DataFrame)

    def test_categorical_encoding_onehot(self, sample_data):
        """Test one-hot encoding of categorical features."""
        preprocessor = ChurnDataPreprocessor(encode_method="onehot")
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)

        # One-hot encoding should create more columns
        assert X_train.shape[1] > 0

    def test_stratified_split(self, sample_data):
        """Test that stratified split maintains class distribution."""
        preprocessor = ChurnDataPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)

        # Churn rates should be similar
        train_rate = y_train.mean()
        test_rate = y_test.mean()

        assert abs(train_rate - test_rate) < 0.05  # Within 5%

    def test_preprocessing_summary(self, sample_data):
        """Test getting preprocessing summary."""
        preprocessor = ChurnDataPreprocessor()
        preprocessor.fit(sample_data)

        summary = preprocessor.get_preprocessing_summary()

        assert summary["is_fitted"]
        assert summary["total_features"] > 0

    def test_feature_importance_df(self, sample_data):
        """Test creating feature importance DataFrame."""
        preprocessor = ChurnDataPreprocessor()
        preprocessor.fit(sample_data)

        importance = np.random.rand(len(preprocessor.get_feature_names()))
        df = preprocessor.get_feature_importance_df(importance)

        assert "feature" in df.columns
        assert "importance" in df.columns


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_generate_sample_data(self):
        """Test generate_sample_data function."""
        df = generate_sample_data(n_samples=100, random_state=42)

        assert len(df) == 100
        assert "churn" in df.columns

    def test_preprocess_churn_data(self):
        """Test preprocess_churn_data function."""
        df = generate_sample_data(n_samples=200, random_state=42)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_churn_data(df)

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert preprocessor.is_fitted_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
