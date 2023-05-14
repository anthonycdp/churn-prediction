"""Data preprocessing module for churn prediction.

Handles feature engineering, encoding, scaling, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Optional, List, Tuple, Dict, Any
import warnings


class ChurnDataPreprocessor:
    """Preprocess customer data for churn prediction modeling.

    This class handles:
    - Missing value imputation
    - Categorical encoding (one-hot and label)
    - Feature scaling
    - Feature engineering
    - Train/test splitting

    Example:
        >>> preprocessor = ChurnDataPreprocessor()
        >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    """

    def __init__(
        self,
        target_column: str = "churn",
        id_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        scaling_method: str = "standard",
        encode_method: str = "onehot",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize the preprocessor.

        Args:
            target_column: Name of the target variable column.
            id_columns: Columns to exclude from features but keep as identifiers.
            drop_columns: Additional columns to drop from features.
            numerical_columns: List of numerical feature columns (auto-detected if None).
            categorical_columns: List of categorical columns (auto-detected if None).
            scaling_method: Method for scaling numerical features ('standard', 'minmax', None).
            encode_method: Method for encoding categorical features ('onehot', 'label').
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
        """
        self.target_column = target_column
        self.id_columns = id_columns or ["customer_id"]
        self.drop_columns = drop_columns or ["churn_probability", "snapshot_date", "dataset"]
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.scaling_method = scaling_method
        self.encode_method = encode_method
        self.test_size = test_size
        self.random_state = random_state

        # Fitted components
        self.scaler_: Optional[StandardScaler] = None
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.onehot_encoder_: Optional[OneHotEncoder] = None
        self.imputer_: Optional[SimpleImputer] = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

        # Store preprocessing metadata
        self.metadata_: Dict[str, Any] = {}

    def _identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical columns.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (numerical_columns, categorical_columns).
        """
        exclude_cols = (
            {self.target_column} |
            set(self.id_columns) |
            set(self.drop_columns)
        )
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if self.numerical_columns is not None and self.categorical_columns is not None:
            return self.numerical_columns, self.categorical_columns

        # Auto-detect column types
        numerical_cols = []
        categorical_cols = []

        for col in feature_cols:
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                # Check if it's actually categorical (few unique values)
                if df[col].nunique() <= 10 and df[col].min() >= 0:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            else:
                categorical_cols.append(col)

        return numerical_cols, categorical_cols

    def _compute_feature_names(self) -> None:
        """Compute feature names after fitting encoders."""
        feature_names = []

        # Add numerical feature names
        if self.numerical_columns_:
            feature_names.extend(self.numerical_columns_)

        # Add categorical feature names (encoded)
        if self.categorical_columns_:
            if self.encode_method == "onehot" and self.onehot_encoder_:
                cat_names = self.onehot_encoder_.get_feature_names_out(
                    self.categorical_columns_
                )
                feature_names.extend(list(cat_names))
            else:
                feature_names.extend(self.categorical_columns_)

        self.feature_names_ = feature_names

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from existing columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with additional engineered features.
        """
        df = df.copy()

        # Average monthly charge (total / tenure)
        if "total_charges" in df.columns and "tenure_months" in df.columns:
            df["avg_monthly_charge"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
            df["charge_change_ratio"] = df["monthly_charges"] / df["avg_monthly_charge"].replace(0, 1)

        # Engagement score (composite metric)
        if all(col in df.columns for col in ["num_support_tickets", "num_complaints", "avg_call_duration"]):
            # Normalize and combine engagement metrics
            df["engagement_score"] = (
                (df["num_support_tickets"] / df["num_support_tickets"].max()) +
                (df["num_complaints"] / max(df["num_complaints"].max(), 1)) -
                (df["avg_call_duration"] / df["avg_call_duration"].max())
            )

        # Tenure bucket
        if "tenure_months" in df.columns:
            df["tenure_bucket"] = pd.cut(
                df["tenure_months"],
                bins=[0, 6, 12, 24, 48, float("inf")],
                labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"]
            )

        # High-risk indicator
        if all(col in df.columns for col in ["num_complaints", "contract_changes", "num_support_tickets"]):
            df["high_risk_flag"] = (
                (df["num_complaints"] >= 2) |
                (df["contract_changes"] >= 2) |
                (df["num_support_tickets"] >= 5)
            ).astype(int)

        # Service count (how many services they have)
        service_cols = ["online_security", "tech_support", "streaming_tv", "streaming_movies"]
        if all(col in df.columns for col in service_cols):
            df["service_count"] = sum(
                (df[col] == "Yes").astype(int) for col in service_cols
            )

        return df

    def fit(self, df: pd.DataFrame) -> "ChurnDataPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            df: Training DataFrame.

        Returns:
            Fitted preprocessor instance.
        """
        # Engineer features first
        df = self._engineer_features(df)

        # Identify column types
        self.numerical_columns_, self.categorical_columns_ = self._identify_columns(df)

        self.metadata_["numerical_columns"] = self.numerical_columns_
        self.metadata_["categorical_columns"] = self.categorical_columns_
        self.metadata_["n_samples"] = len(df)

        # Fit numerical imputer
        if self.numerical_columns_:
            self.imputer_ = SimpleImputer(strategy="median")
            self.imputer_.fit(df[self.numerical_columns_])

        # Fit scaler
        if self.scaling_method and self.numerical_columns_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(df[self.numerical_columns_].fillna(df[self.numerical_columns_].median()))

        # Fit categorical encoder
        if self.categorical_columns_:
            if self.encode_method == "onehot":
                self.onehot_encoder_ = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    drop="first"  # Avoid multicollinearity
                )
                # Handle missing values in categorical columns
                cat_data = df[self.categorical_columns_].fillna("Unknown")
                self.onehot_encoder_.fit(cat_data)
            else:
                for col in self.categorical_columns_:
                    le = LabelEncoder()
                    le.fit(df[col].fillna("Unknown").astype(str))
                    self.label_encoders_[col] = le

        # Compute feature names after fitting encoders
        self._compute_feature_names()

        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform data using fitted preprocessor.

        Args:
            df: Input DataFrame.
            include_target: Whether to return the target variable.

        Returns:
            Tuple of (features DataFrame, target Series or None).

        Raises:
            ValueError: If preprocessor hasn't been fitted.
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")

        # Engineer features
        df = self._engineer_features(df)

        # Extract target
        y = None
        if include_target and self.target_column in df.columns:
            y = df[self.target_column].astype(int)

        # Process numerical features
        numerical_df = pd.DataFrame()
        if self.numerical_columns_:
            num_data = df[self.numerical_columns_].copy()
            if self.imputer_:
                num_data = pd.DataFrame(
                    self.imputer_.transform(num_data),
                    columns=self.numerical_columns_,
                    index=num_data.index
                )
            if self.scaler_:
                num_data = pd.DataFrame(
                    self.scaler_.transform(num_data),
                    columns=self.numerical_columns_,
                    index=num_data.index
                )
            numerical_df = num_data

        # Process categorical features
        categorical_df = pd.DataFrame()
        if self.categorical_columns_:
            if self.encode_method == "onehot" and self.onehot_encoder_:
                cat_data = df[self.categorical_columns_].fillna("Unknown")
                encoded = self.onehot_encoder_.transform(cat_data)
                feature_names = self.onehot_encoder_.get_feature_names_out(self.categorical_columns_)
                categorical_df = pd.DataFrame(
                    encoded,
                    columns=feature_names,
                    index=df.index
                )
            else:
                for col in self.categorical_columns_:
                    if col in self.label_encoders_:
                        le = self.label_encoders_[col]
                        # Handle unknown categories
                        col_data = df[col].fillna("Unknown").astype(str)
                        col_encoded = col_data.apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                        categorical_df[col] = col_encoded

        # Combine features
        X = pd.concat([numerical_df, categorical_df], axis=1)
        self.feature_names_ = list(X.columns)

        return X, y

    def fit_transform(
        self,
        df: pd.DataFrame,
        split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fit preprocessor and transform data with train/test split.

        Args:
            df: Input DataFrame.
            split: Whether to split into train/test sets.

        Returns:
            If split=True: (X_train, X_test, y_train, y_test)
            If split=False: raises ValueError (use fit() and transform() instead)
        """
        if not split:
            self.fit(df)
            X, y = self.transform(df)
            return X, y

        # Split first to avoid data leakage
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column] if self.target_column in df.columns else None
        )

        # Fit on training data only
        self.fit(train_df)

        # Transform both sets
        X_train, y_train = self.transform(train_df)
        X_test, y_test = self.transform(test_df)

        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names after preprocessing.

        Returns:
            List of feature column names.

        Raises:
            ValueError: If preprocessor hasn't been fitted.
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted first.")
        return self.feature_names_.copy()

    def get_feature_importance_df(
        self,
        importance_values: np.ndarray,
        sort: bool = True
    ) -> pd.DataFrame:
        """Create a feature importance DataFrame.

        Args:
            importance_values: Array of feature importance values.
            sort: Whether to sort by importance.

        Returns:
            DataFrame with feature names and importance values.
        """
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance_values
        })

        if sort:
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get a summary of the preprocessing steps.

        Returns:
            Dictionary with preprocessing metadata.
        """
        return {
            "is_fitted": self.is_fitted_,
            "numerical_features": len(self.numerical_columns_) if self.is_fitted_ else 0,
            "categorical_features": len(self.categorical_columns_) if self.is_fitted_ else 0,
            "total_features": len(self.feature_names_) if self.is_fitted_ else 0,
            "scaling_method": self.scaling_method,
            "encoding_method": self.encode_method,
            "feature_names": self.feature_names_ if self.is_fitted_ else [],
        }


def preprocess_churn_data(
    df: pd.DataFrame,
    target_column: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ChurnDataPreprocessor]:
    """Convenience function to preprocess churn data.

    Args:
        df: Input DataFrame with customer data.
        target_column: Name of the target column.
        test_size: Proportion of test data.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor).
    """
    preprocessor = ChurnDataPreprocessor(
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test the preprocessor
    from generator import ChurnDataGenerator

    # Generate sample data
    gen = ChurnDataGenerator(random_state=42)
    df = gen.generate(n_samples=1000)

    print("Original data shape:", df.shape)
    print("Columns:", list(df.columns))

    # Preprocess
    preprocessor = ChurnDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")
    print(f"\nFeature names: {preprocessor.get_feature_names()[:10]}...")
