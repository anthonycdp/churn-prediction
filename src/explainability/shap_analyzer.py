"""SHAP-based model explainability for churn prediction.

Provides comprehensive model interpretability using SHAP values.
"""

import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple

# SHAP is required
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


class SHAPAnalyzer:
    """Analyze and explain churn prediction models using SHAP values.

    This class provides:
    - Global feature importance
    - Individual prediction explanations
    - Feature interaction effects
    - Visualizations for model interpretability

    Example:
        >>> analyzer = SHAPAnalyzer(model, X_train)
        >>> analyzer.plot_summary()
        >>> explanation = analyzer.explain_prediction(X_test.iloc[0])
    """

    def __init__(
        self,
        model: Any,
        background_data: pd.DataFrame,
        model_type: str = "auto",
        random_state: int = 42,
    ):
        """Initialize the SHAP analyzer.

        Args:
            model: Trained model (sklearn, XGBoost, etc.).
            background_data: Reference data for SHAP background distribution.
            model_type: Type of model ('tree', 'linear', 'auto').
            random_state: Random seed.

        Raises:
            ImportError: If SHAP is not installed.
        """
        if not HAS_SHAP:
            raise ImportError(
                "SHAP is required. Install with: pip install shap"
            )

        self.model = model
        self.background_data = background_data
        self.model_type = model_type
        self.random_state = random_state

        self.explainer_: Optional[shap.Explainer] = None
        self.explanation_: Optional[Any] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = list(background_data.columns)

        # Initialize explainer
        self._create_explainer()

    def _create_explainer(self):
        """Create the appropriate SHAP explainer for the model type."""
        # Sample background data if too large
        if len(self.background_data) > 100:
            background = shap.sample(
                self.background_data,
                100,
                random_state=self.random_state,
            )
        else:
            background = self.background_data

        if self.model_type == "auto":
            # Try to detect model type
            model_class = type(self.model).__name__.lower()

            if any(x in model_class for x in ["tree", "forest", "xgb", "lgbm", "gradient", "catboost"]):
                self.model_type = "tree"
            elif "logistic" in model_class or "linear" in model_class:
                self.model_type = "linear"
            else:
                self.model_type = "kernel"

        if self.model_type == "tree":
            try:
                self.explainer_ = shap.TreeExplainer(
                    self.model,
                    data=background,
                    feature_perturbation="interventional",
                )
            except Exception:
                self.model_type = "kernel"
                self.explainer_ = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
        elif self.model_type == "linear":
            try:
                masker = shap.maskers.Independent(
                    background,
                    max_samples=min(len(background), 100),
                )
                self.explainer_ = shap.LinearExplainer(
                    self.model,
                    masker,
                )
            except Exception:
                self.model_type = "kernel"
                self.explainer_ = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
        else:
            self.explainer_ = shap.KernelExplainer(
                self.model.predict_proba, background
            )

    def _normalize_shap_values(self, shap_output: Any) -> np.ndarray:
        """Convert SHAP output to a consistent 2D array."""
        if hasattr(shap_output, "values"):
            shap_values = shap_output.values
        else:
            shap_values = shap_output

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values = np.asarray(shap_values)
        n_features = len(self.feature_names_)

        if shap_values.ndim == 3:
            if shap_values.shape[1] == n_features:
                shap_values = shap_values[:, :, -1]
            elif shap_values.shape[2] == n_features:
                shap_values = shap_values[-1, :, :]

        if shap_values.ndim == 2 and shap_values.shape[0] == n_features and shap_values.shape[1] != n_features:
            shap_values = shap_values.T

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        if shap_values.ndim == 2 and shap_values.shape[1] == 2 and n_features != 2:
            shap_values = shap_values[:, 1:2]

        return shap_values

    def _extract_base_value(self, shap_output: Any) -> float:
        """Extract the positive-class base value from SHAP output."""
        if hasattr(shap_output, "base_values"):
            base_value = shap_output.base_values
        else:
            base_value = self.explainer_.expected_value

        base_value = np.asarray(base_value)

        if base_value.ndim == 0:
            return float(base_value)

        if base_value.ndim == 1:
            return float(base_value[-1])

        return float(base_value.reshape(-1)[-1])

    def _compute_explanation(
        self,
        X: pd.DataFrame,
        check_additivity: bool = True,
    ) -> Any:
        """Compute SHAP output using the most appropriate API for the explainer."""
        X = X[self.feature_names_]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_type == "kernel":
                shap_values_fn = getattr(self.explainer_, "shap_values")
                shap_values_kwargs: Dict[str, Any] = {}
                signature = inspect.signature(shap_values_fn)

                if "silent" in signature.parameters:
                    shap_values_kwargs["silent"] = True
                if "check_additivity" in signature.parameters:
                    shap_values_kwargs["check_additivity"] = check_additivity

                return shap_values_fn(X, **shap_values_kwargs)

            explainer_kwargs: Dict[str, Any] = {}
            signature = inspect.signature(self.explainer_.__call__)
            if "check_additivity" in signature.parameters:
                explainer_kwargs["check_additivity"] = check_additivity
            if "silent" in signature.parameters:
                explainer_kwargs["silent"] = True

            return self.explainer_(X, **explainer_kwargs)

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        check_additivity: bool = True,
    ) -> np.ndarray:
        """Compute SHAP values for a dataset.

        Args:
            X: Data to compute SHAP values for.
            check_additivity: Whether to check SHAP additivity.

        Returns:
            Array of SHAP values.
        """
        explanation = self._compute_explanation(X, check_additivity=check_additivity)
        shap_values = self._normalize_shap_values(explanation)

        self.explanation_ = explanation
        self.shap_values_ = shap_values
        return shap_values

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        aggregate: str = "mean_abs",
    ) -> pd.DataFrame:
        """Calculate global feature importance from SHAP values.

        Args:
            X: Data to compute importance from.
            aggregate: Aggregation method ('mean_abs', 'mean', 'max').

        Returns:
            DataFrame with feature importance scores.
        """
        shap_values = (
            self.compute_shap_values(X)
            if self.shap_values_ is None or len(self.shap_values_) != len(X)
            else self.shap_values_
        )

        # Ensure shap_values is 2D (samples x features)
        if shap_values.ndim > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)

        if aggregate == "mean_abs":
            importance = np.abs(shap_values).mean(axis=0)
        elif aggregate == "mean":
            importance = shap_values.mean(axis=0)
        elif aggregate == "max":
            importance = np.abs(shap_values).max(axis=0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        # Ensure importance is 1D
        importance = np.asarray(importance).flatten()

        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_pct"] = df["importance"] / df["importance"].sum() * 100

        return df

    def explain_prediction(
        self,
        x: Union[pd.Series, pd.DataFrame],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Explain a single prediction.

        Args:
            x: Single observation to explain.
            top_n: Number of top features to include.

        Returns:
            Dictionary with explanation details.
        """
        # Convert to DataFrame if Series
        if isinstance(x, pd.Series):
            x_df = pd.DataFrame([x])
        else:
            x_df = x

        # Ensure column order
        x_df = x_df[self.feature_names_]

        explanation = self._compute_explanation(x_df, check_additivity=False)
        shap_values = self._normalize_shap_values(explanation)[0]
        base_value = self._extract_base_value(explanation)

        # Get model prediction
        proba = self.model.predict_proba(x_df)[0, 1]

        # Get feature values as 1D array
        feature_values = np.asarray(x_df.values[0]).flatten()

        # Create feature contribution DataFrame
        contributions = pd.DataFrame({
            "feature": self.feature_names_,
            "value": feature_values,
            "shap_value": shap_values,
            "abs_shap": np.abs(shap_values),
        })
        contributions = contributions.sort_values("abs_shap", ascending=False)
        contributions["direction"] = np.where(
            contributions["shap_value"] > 0, "Increase Churn", "Decrease Churn"
        )

        # Get top features
        top_features = contributions.head(top_n)

        return {
            "base_value": base_value,
            "predicted_probability": proba,
            "predicted_class": int(proba >= 0.5),
            "top_features": top_features.to_dict("records"),
            "features_increasing_churn": contributions[
                contributions["shap_value"] > 0
            ].head(top_n).to_dict("records"),
            "features_decreasing_churn": contributions[
                contributions["shap_value"] < 0
            ].head(top_n).to_dict("records"),
        }

    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_size: Tuple[int, int] = (10, 8),
        show: bool = True,
    ) -> plt.Figure:
        """Plot SHAP summary (beeswarm) plot.

        Args:
            X: Data to plot.
            max_display: Maximum number of features to display.
            plot_size: Figure size.
            show: Whether to show the plot.

        Returns:
            Matplotlib figure.
        """
        shap_values = (
            self.compute_shap_values(X)
            if self.shap_values_ is None or len(self.shap_values_) != len(X)
            else self.shap_values_
        )

        fig, ax = plt.subplots(figsize=plot_size)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="The NumPy global RNG was seeded by calling `np.random.seed`.*",
            )
            shap.summary_plot(
                shap_values,
                X[self.feature_names_],
                max_display=max_display,
                show=False,
                plot_size=plot_size,
            )

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def plot_bar(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_size: Tuple[int, int] = (10, 8),
        show: bool = True,
    ) -> plt.Figure:
        """Plot feature importance bar chart.

        Args:
            X: Data to compute importance from.
            max_display: Maximum features to display.
            plot_size: Figure size.
            show: Whether to show the plot.

        Returns:
            Matplotlib figure.
        """
        importance = self.get_feature_importance(X)
        importance = importance.head(max_display)

        fig, ax = plt.subplots(figsize=plot_size)

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance)))[::-1]

        bars = ax.barh(
            importance["feature"][::-1],
            importance["importance"][::-1],
            color=colors,
        )

        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance (SHAP)")
        ax.grid(True, axis="x", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, importance["importance"][::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def plot_waterfall(
        self,
        x: pd.Series,
        max_display: int = 15,
        show: bool = True,
    ) -> plt.Figure:
        """Plot waterfall chart for a single prediction.

        Args:
            x: Single observation to explain.
            max_display: Maximum features to display.
            show: Whether to show the plot.

        Returns:
            Matplotlib figure.
        """
        explanation = self.explain_prediction(x, top_n=max_display)

        # Get data for waterfall
        base_value = explanation["base_value"]
        features = explanation["top_features"][:max_display]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Prepare data
        labels = [f["feature"] for f in features] + ["Final Prediction"]
        values = [f["shap_value"] for f in features] + [0]  # Last bar is for total

        # Calculate cumulative values
        cumulative = [base_value]
        for v in values[:-1]:
            cumulative.append(cumulative[-1] + v)

        # Create waterfall bars
        colors = ["red" if v > 0 else "green" for v in values[:-1]] + ["blue"]

        # Plot bars
        left_positions = cumulative[:-1] + [0]
        bar_values = values[:-1] + [explanation["predicted_probability"]]

        for i, (label, value, color) in enumerate(zip(labels, bar_values, colors)):
            if i < len(features):
                left = cumulative[i] - values[i] if values[i] < 0 else cumulative[i]
                width = abs(values[i])
            else:
                left = 0
                width = explanation["predicted_probability"]
            ax.barh(i, width, left=left, color=color, alpha=0.7)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(x=base_value, color="gray", linestyle="--", label=f"Base: {base_value:.3f}")
        ax.axvline(x=0.5, color="red", linestyle=":", alpha=0.5, label="Decision Boundary")

        ax.set_xlabel("Churn Probability")
        ax.set_title(f"Prediction Explanation (Prob: {explanation['predicted_probability']:.3f})")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot SHAP dependence plot for a feature.

        Args:
            X: Data to plot.
            feature: Feature to plot dependence for.
            interaction_feature: Feature for coloring (auto-detected if None).
            show: Whether to show the plot.

        Returns:
            Matplotlib figure.
        """
        shap_values = (
            self.compute_shap_values(X)
            if self.shap_values_ is None or len(self.shap_values_) != len(X)
            else self.shap_values_
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        # Auto-detect interaction feature
        if interaction_feature is None:
            interaction_index = "auto"
        else:
            interaction_index = self.feature_names_.index(interaction_feature)

        shap.dependence_plot(
            feature,
            shap_values,
            X[self.feature_names_],
            interaction_index=interaction_index,
            show=False,
            ax=ax,
        )

        ax.set_title(f"SHAP Dependence: {feature}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def plot_force_plot(
        self,
        x: pd.Series,
        matplotlib: bool = True,
        show: bool = True,
    ) -> Any:
        """Plot force plot for a single prediction.

        Args:
            x: Single observation to explain.
            matplotlib: Use matplotlib (True) or JavaScript (False).
            show: Whether to show the plot.

        Returns:
            Force plot object.
        """
        # Convert to DataFrame
        if isinstance(x, pd.Series):
            x_df = pd.DataFrame([x])
        else:
            x_df = x

        x_df = x_df[self.feature_names_]

        # Get SHAP explanation
        explanation = self._compute_explanation(x_df, check_additivity=False)
        base_value = self._extract_base_value(explanation)
        shap_values = self._normalize_shap_values(explanation)[0]

        if matplotlib:
            fig = shap.force_plot(
                base_value,
                shap_values,
                x_df.iloc[0],
                matplotlib=True,
                show=show,
            )
            return fig
        else:
            return shap.force_plot(
                base_value,
                shap_values,
                x_df.iloc[0],
            )

    def get_high_risk_customers(
        self,
        X: pd.DataFrame,
        y_proba: np.ndarray,
        top_pct: float = 0.10,
    ) -> pd.DataFrame:
        """Identify high-risk customers with explanations.

        Args:
            X: Customer data.
            y_proba: Predicted churn probabilities.
            top_pct: Top percentage of high-risk customers to return.

        Returns:
            DataFrame with high-risk customers and top contributing features.
        """
        # Get top-risk customers
        n_top = int(len(X) * top_pct)
        top_indices = np.argsort(y_proba)[-n_top:]

        # Compute SHAP values if needed
        if self.shap_values_ is None or len(self.shap_values_) != len(X):
            self.compute_shap_values(X)

        # Get top contributing features for each
        results = []
        for idx in top_indices:
            shap_vals = self.shap_values_[idx]

            # Flatten if needed (handle multi-dimensional arrays)
            if shap_vals.ndim > 1:
                shap_vals = shap_vals.flatten()

            # Get top 3 contributing features
            top_features_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]
            top_features = [
                (self.feature_names_[int(i)], float(shap_vals[i]), X.iloc[idx][self.feature_names_[int(i)]])
                for i in top_features_idx
            ]

            results.append({
                "index": idx,
                "churn_probability": y_proba[idx],
                "top_feature_1": top_features[0][0],
                "top_feature_1_shap": top_features[0][1],
                "top_feature_1_value": top_features[0][2],
                "top_feature_2": top_features[1][0],
                "top_feature_2_shap": top_features[1][1],
                "top_feature_2_value": top_features[1][2],
                "top_feature_3": top_features[2][0],
                "top_feature_3_shap": top_features[2][1],
                "top_feature_3_value": top_features[2][2],
            })

        return pd.DataFrame(results).sort_values("churn_probability", ascending=False)


def analyze_model(
    model: Any,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_type: str = "auto",
) -> Tuple[SHAPAnalyzer, pd.DataFrame]:
    """Convenience function to analyze a model with SHAP.

    Args:
        model: Trained model.
        X_train: Training data for background.
        X_explain: Data to explain.
        model_type: Model type hint.

    Returns:
        Tuple of (analyzer, feature_importance).
    """
    analyzer = SHAPAnalyzer(model, X_train, model_type=model_type)
    importance = analyzer.get_feature_importance(X_explain)

    return analyzer, importance


if __name__ == "__main__":
    # Test SHAP analyzer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Generate test data
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:400], y[:400])

    # Analyze
    analyzer = SHAPAnalyzer(model, X[:100])
    importance = analyzer.get_feature_importance(X[400:])

    print("Feature Importance:")
    print(importance)

    # Explain a prediction
    explanation = analyzer.explain_prediction(X.iloc[400])
    print(f"\nPrediction: {explanation['predicted_probability']:.3f}")
    print("Top features:", [f["feature"] for f in explanation["top_features"][:3]])
