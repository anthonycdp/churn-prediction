"""Sample data generator for churn prediction.

Generates realistic synthetic customer data for churn prediction modeling.
The data includes realistic correlations between features and churn probability.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import os


class ChurnDataGenerator:
    """Generate synthetic customer churn data.

    This class creates realistic customer data with features that correlate
    with churn probability based on typical business patterns.

    Example:
        >>> generator = ChurnDataGenerator(random_state=42)
        >>> df = generator.generate(n_samples=1000)
        >>> df.shape
        (1000, 16)
    """

    def __init__(self, random_state: int = 42):
        """Initialize the data generator.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Define feature distributions and their impact on churn
        self._setup_feature_distributions()

    def _setup_feature_distributions(self):
        """Define feature distributions and their relationship to churn."""
        # Contract types with their base churn rates
        self.contract_types = {
            "Month-to-month": {"weight": 0.55, "churn_modifier": 2.0},
            "One year": {"weight": 0.25, "churn_modifier": 0.5},
            "Two year": {"weight": 0.20, "churn_modifier": 0.2},
        }

        # Payment methods
        self.payment_methods = {
            "Electronic check": {"weight": 0.35, "churn_modifier": 1.3},
            "Mailed check": {"weight": 0.20, "churn_modifier": 1.1},
            "Bank transfer (automatic)": {"weight": 0.25, "churn_modifier": 0.7},
            "Credit card (automatic)": {"weight": 0.20, "churn_modifier": 0.6},
        }

        # Internet services
        self.internet_services = {
            "DSL": {"weight": 0.35, "churn_modifier": 0.8},
            "Fiber optic": {"weight": 0.45, "churn_modifier": 1.4},
            "No": {"weight": 0.20, "churn_modifier": 0.3},
        }

        # Binary services (Yes/No)
        self.binary_services = {
            "Yes": {"weight": 0.35, "churn_modifier": 0.7},
            "No": {"weight": 0.55, "churn_modifier": 1.2},
            "No internet service": {"weight": 0.10, "churn_modifier": 0.3},
        }

    def generate(
        self,
        n_samples: int = 5000,
        base_churn_rate: float = 0.20,
        include_customer_id: bool = True,
        include_timestamp: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic churn dataset.

        Args:
            n_samples: Number of customer records to generate.
            base_churn_rate: Base probability of churn (modified by features).
            include_customer_id: Whether to include customer IDs.
            include_timestamp: Whether to include a timestamp column.

        Returns:
            DataFrame with customer data including churn labels.
        """
        data = {}

        # Generate customer IDs
        if include_customer_id:
            data["customer_id"] = [f"CUST-{str(i).zfill(5)}" for i in range(1, n_samples + 1)]

        # Generate tenure (months) - exponential-ish distribution
        # Longer tenure customers are less likely to churn
        data["tenure_months"] = self._generate_tenure(n_samples)

        # Generate charges
        monthly_charges = self._generate_monthly_charges(n_samples)
        data["monthly_charges"] = monthly_charges
        data["total_charges"] = self._calculate_total_charges(
            data["tenure_months"], monthly_charges
        )

        # Generate usage metrics
        data["avg_call_duration"] = self._generate_call_duration(n_samples)
        data["num_support_tickets"] = self._generate_support_tickets(n_samples)
        data["num_complaints"] = self._generate_complaints(n_samples)
        data["data_usage_gb"] = self._generate_data_usage(n_samples)
        data["contract_changes"] = self._generate_contract_changes(n_samples)

        # Generate categorical features
        data["contract_type"] = self._sample_categorical(
            n_samples, self.contract_types
        )
        data["payment_method"] = self._sample_categorical(
            n_samples, self.payment_methods
        )
        data["internet_service"] = self._sample_categorical(
            n_samples, self.internet_services
        )

        # Generate service subscriptions
        data["online_security"] = self._sample_categorical(
            n_samples, self.binary_services
        )
        data["tech_support"] = self._sample_categorical(
            n_samples, self.binary_services
        )
        data["streaming_tv"] = self._sample_categorical(
            n_samples, self.binary_services
        )
        data["streaming_movies"] = self._sample_categorical(
            n_samples, self.binary_services
        )

        # Calculate churn probability based on all features
        churn_probability = self._calculate_churn_probability(
            data, base_churn_rate
        )

        # Generate churn labels based on probability
        data["churn"] = self.rng.random(n_samples) < churn_probability
        data["churn_probability"] = churn_probability

        # Add timestamp
        if include_timestamp:
            base_date = datetime.now() - timedelta(days=30)
            data["snapshot_date"] = base_date.strftime("%Y-%m-%d")

        df = pd.DataFrame(data)

        # Round numerical columns
        numerical_cols = [
            "monthly_charges", "total_charges", "avg_call_duration",
            "churn_probability"
        ]
        for col in numerical_cols:
            df[col] = df[col].round(2)

        return df

    def _generate_tenure(self, n_samples: int) -> np.ndarray:
        """Generate tenure in months."""
        # Mix of distributions to create realistic spread
        tenure = np.where(
            self.rng.random(n_samples) < 0.3,
            self.rng.randint(1, 12, n_samples),  # New customers
            np.clip(
                self.rng.exponential(30, n_samples) + 3,
                1, 72
            ).astype(int)
        )
        return tenure

    def _generate_monthly_charges(self, n_samples: int) -> np.ndarray:
        """Generate monthly charges."""
        # Bimodal distribution for different service tiers
        low_tier = self.rng.normal(35, 10, n_samples)
        high_tier = self.rng.normal(75, 15, n_samples)
        mask = self.rng.random(n_samples) < 0.6
        charges = np.where(mask, low_tier, high_tier)
        return np.clip(charges, 18, 120)

    def _calculate_total_charges(
        self, tenure: np.ndarray, monthly: np.ndarray
    ) -> np.ndarray:
        """Calculate total charges with some variance."""
        base_total = tenure * monthly
        # Add some noise for promotions, rate changes, etc.
        noise = self.rng.normal(0, base_total * 0.05)
        return np.maximum(base_total + noise, monthly)  # At least one month

    def _generate_call_duration(self, n_samples: int) -> np.ndarray:
        """Generate average call duration in minutes."""
        return np.clip(
            self.rng.exponential(5, n_samples) + 1,
            0.5, 30
        )

    def _generate_support_tickets(self, n_samples: int) -> np.ndarray:
        """Generate number of support tickets."""
        # Most customers have few tickets
        return np.clip(
            self.rng.poisson(2, n_samples),
            0, 20
        )

    def _generate_complaints(self, n_samples: int) -> np.ndarray:
        """Generate number of complaints."""
        # Very few complaints typically
        return np.clip(
            self.rng.poisson(0.5, n_samples),
            0, 10
        )

    def _generate_data_usage(self, n_samples: int) -> np.ndarray:
        """Generate data usage in GB."""
        return np.clip(
            self.rng.lognormal(3, 1, n_samples),
            0.1, 500
        ).round(1)

    def _generate_contract_changes(self, n_samples: int) -> np.ndarray:
        """Generate number of contract changes."""
        return np.clip(
            self.rng.poisson(0.3, n_samples),
            0, 5
        )

    def _sample_categorical(
        self, n_samples: int, categories: Dict
    ) -> np.ndarray:
        """Sample from categorical distribution."""
        choices = list(categories.keys())
        weights = [c["weight"] for c in categories.values()]
        return self.rng.choice(choices, n_samples, p=weights)

    def _calculate_churn_probability(
        self, data: Dict, base_rate: float
    ) -> np.ndarray:
        """Calculate churn probability based on all features."""
        n_samples = len(data["tenure_months"])
        log_odds = np.log(base_rate / (1 - base_rate))

        # Tenure effect (logarithmic decrease)
        tenure_effect = -0.5 * np.log1p(data["tenure_months"])
        log_odds += tenure_effect

        # Monthly charges effect (higher = more churn)
        charges_effect = (data["monthly_charges"] - 50) / 50
        log_odds += charges_effect * 0.3

        # Support tickets effect
        tickets_effect = (data["num_support_tickets"] - 2) / 5
        log_odds += tickets_effect * 0.2

        # Complaints effect (strong predictor)
        complaints_effect = data["num_complaints"] * 0.3
        log_odds += complaints_effect

        # Contract changes effect
        changes_effect = data["contract_changes"] * 0.2
        log_odds += changes_effect

        # Categorical feature effects
        for cat_col, cat_dict in [
            ("contract_type", self.contract_types),
            ("payment_method", self.payment_methods),
            ("internet_service", self.internet_services),
        ]:
            for category, props in cat_dict.items():
                mask = data[cat_col] == category
                log_odds[mask] += np.log(props["churn_modifier"]) * 0.3

        # Convert log-odds to probability
        probability = 1 / (1 + np.exp(-log_odds))

        # Add some random noise
        noise = self.rng.normal(0, 0.1, n_samples)
        probability = np.clip(probability + noise, 0.02, 0.98)

        return probability

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        create_dirs: bool = True
    ) -> str:
        """Save generated data to CSV file.

        Args:
            df: DataFrame to save.
            filepath: Path to save file.
            create_dirs: Whether to create directories if they don't exist.

        Returns:
            Path to saved file.
        """
        if create_dirs:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath, index=False)
        return filepath

    def generate_train_test_split(
        self,
        n_train: int = 4000,
        n_test: int = 1000,
        base_churn_rate: float = 0.20,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate separate train and test datasets.

        Args:
            n_train: Number of training samples.
            n_test: Number of test samples.
            base_churn_rate: Base churn rate for generation.

        Returns:
            Tuple of (train_df, test_df).
        """
        # Use different random states for train/test
        train_gen = ChurnDataGenerator(random_state=self.random_state)
        test_gen = ChurnDataGenerator(random_state=self.random_state + 1000)

        train_df = train_gen.generate(n_train, base_churn_rate)
        train_df["dataset"] = "train"

        test_df = test_gen.generate(n_test, base_churn_rate)
        test_df["dataset"] = "test"

        return train_df, test_df


def generate_sample_data(
    n_samples: int = 5000,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Convenience function to generate sample data.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed.
        output_path: Optional path to save CSV.

    Returns:
        Generated DataFrame.
    """
    generator = ChurnDataGenerator(random_state=random_state)
    df = generator.generate(n_samples=n_samples)

    if output_path:
        generator.save_to_csv(df, output_path)

    return df


if __name__ == "__main__":
    # Generate sample data when run directly
    df = generate_sample_data(n_samples=5000, random_state=42)
    print(f"Generated {len(df)} samples")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"\nFeature columns: {list(df.columns)}")
    print(f"\nSample data:\n{df.head()}")
