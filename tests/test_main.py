"""Tests for the main pipeline orchestration helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import main
from src.analysis.retention_simulation import SimulationResult


class DummyProbabilityModel:
    """Minimal classifier stub that returns fixed probabilities."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        positive = self.probabilities[: len(X)]
        negative = 1 - positive
        return np.column_stack([negative, positive])


def make_output_dirs(base_dir: Path) -> dict:
    """Create output directories compatible with main.py helpers."""
    dirs = {
        "base": str(base_dir),
        "figures": str(base_dir / "figures"),
        "models": str(base_dir / "models"),
        "reports": str(base_dir / "reports"),
        "data": str(base_dir / "data"),
    }

    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return dirs


def test_predict_with_threshold_uses_custom_operating_point():
    """Predictions should reflect the caller-provided threshold."""
    model = DummyProbabilityModel([0.2, 0.4, 0.8])
    X = pd.DataFrame({"feature": [1, 2, 3]})

    y_proba, y_pred = main.predict_with_threshold(model, X, threshold=0.4)

    np.testing.assert_allclose(y_proba, [0.2, 0.4, 0.8])
    np.testing.assert_array_equal(y_pred, [0, 1, 1])


def test_split_training_data_creates_validation_holdout():
    """Training data should be split into train/validation without losing rows."""
    X = pd.DataFrame({"feature": np.arange(40)})
    y = pd.Series([0] * 20 + [1] * 20)

    X_fit, X_val, y_fit, y_val = main.split_training_data(
        X,
        y,
        validation_size=0.25,
        random_state=42,
    )

    assert len(X_fit) + len(X_val) == len(X)
    assert len(y_fit) + len(y_val) == len(y)
    assert abs(y_fit.mean() - y.mean()) < 0.10
    assert abs(y_val.mean() - y.mean()) < 0.10


def test_evaluate_model_uses_supplied_threshold(tmp_path):
    """The evaluation helper should use the operating threshold chosen upstream."""
    model = DummyProbabilityModel([0.1, 0.3, 0.6, 0.7])
    X_test = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y_test = pd.Series([0, 0, 1, 1])
    dirs = make_output_dirs(tmp_path)

    result = main.evaluate_model(
        model,
        X_test,
        y_test,
        threshold=0.65,
        dirs=dirs,
    )

    assert result.threshold == 0.65
    np.testing.assert_array_equal(result.y_pred, [0, 0, 0, 1])


def test_simulate_retention_uses_operating_threshold(monkeypatch, tmp_path):
    """Retention simulation should target customers with the active operating threshold."""
    captured = {}

    class DummySimulator:
        def __init__(self, offer_cost, success_rate):
            self.offer_cost = offer_cost
            self.success_rate = success_rate

        def simulate_campaign(self, y_proba, customer_values, strategy, threshold):
            captured["threshold"] = threshold
            return SimulationResult(
                strategy=strategy,
                n_targeted=1,
                n_would_churn=1,
                n_retained=1,
                total_cost=50.0,
                revenue_saved=200.0,
                net_benefit=150.0,
                roi=300.0,
                retention_rate_achieved=1.0,
            )

        def generate_report(self, result):
            return "report"

        def compare_strategies(self, y_proba, customer_values, thresholds=None, **kwargs):
            captured["thresholds"] = thresholds
            return pd.DataFrame(
                [
                    {
                        "strategy": "target_high_risk",
                        "threshold": thresholds[0],
                        "net_benefit": 150.0,
                        "roi": 300.0,
                    }
                ]
            )

        def plot_strategy_comparison(self, comparison, ax=None):
            return ax.figure

        def simulate_time_horizon(self, y_proba, customer_values, n_months=12, threshold=0.5):
            captured["time_threshold"] = threshold
            return {
                "monthly_results": pd.DataFrame(
                    {
                        "month": [1],
                        "cumulative_revenue": [200.0],
                        "cumulative_cost": [50.0],
                        "net_benefit": [150.0],
                    }
                ),
                "final_retention_rate": 1.0,
                "total_net_benefit": 150.0,
            }

        def plot_time_simulation(self, time_results, ax=None):
            return ax.figure

    monkeypatch.setattr(main, "RetentionSimulator", DummySimulator)
    monkeypatch.setattr(main.np.random, "lognormal", lambda *args, **kwargs: np.array([100.0, 200.0]))

    dirs = make_output_dirs(tmp_path)
    main.simulate_retention(np.array([0.1, 0.8]), targeting_threshold=0.15, dirs=dirs)

    assert captured["threshold"] == 0.15
    assert 0.15 in captured["thresholds"]
    assert captured["time_threshold"] == 0.15
