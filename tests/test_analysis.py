"""Tests for cost analysis and retention simulation modules."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.cost_analysis import (
    CostAnalyzer,
    CostConfig,
    CostResult,
    analyze_costs,
)
from src.analysis.retention_simulation import (
    RetentionSimulator,
    RetentionStrategy,
    SimulationResult,
    simulate_retention,
)


class TestCostAnalyzer:
    """Test cases for CostAnalyzer."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.2, n)
        y_proba = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        assert analyzer.cost_fn == 500
        assert analyzer.cost_fp == 50

    def test_calculate_cost_matrix(self):
        """Test cost matrix calculation."""
        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50, value_tp=450)
        cost_matrix = analyzer.calculate_cost_matrix()

        assert cost_matrix.shape == (2, 2)
        assert cost_matrix[1, 0] == -500  # FN cost
        assert cost_matrix[0, 1] == -50   # FP cost

    def test_analyze_predictions(self, sample_predictions):
        """Test analyzing predictions."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        result = analyzer.analyze(y_true, y_pred, y_proba)

        assert isinstance(result, CostResult)
        assert result.total_cost >= 0
        assert isinstance(result.cost_breakdown, dict)

    def test_cost_breakdown(self, sample_predictions):
        """Test cost breakdown contains all components."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        result = analyzer.analyze(y_true, y_pred, y_proba)

        required_keys = [
            "false_negatives_cost",
            "false_positives_cost",
            "true_positives_value",
            "true_negatives_value",
        ]

        for key in required_keys:
            assert key in result.cost_breakdown

    def test_find_optimal_threshold(self, sample_predictions):
        """Test finding optimal threshold."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        threshold = analyzer.find_optimal_threshold(y_true, y_proba)

        assert 0 < threshold < 1

    def test_threshold_analysis_dataframe(self, sample_predictions):
        """Test threshold analysis returns DataFrame."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        result = analyzer.analyze(y_true, y_pred, y_proba)

        assert result.threshold_analysis is not None
        assert isinstance(result.threshold_analysis, pd.DataFrame)
        assert "threshold" in result.threshold_analysis.columns
        assert "net_benefit" in result.threshold_analysis.columns

    def test_optimize_targeting(self, sample_predictions):
        """Test budget-constrained targeting optimization."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50, monthly_budget=500)
        targeting = analyzer.optimize_targeting(y_proba, budget=500)

        assert "n_targeted" in targeting
        assert "total_cost" in targeting
        assert targeting["total_cost"] <= 500

    def test_calculate_expected_value(self, sample_predictions):
        """Test expected value calculation."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        ev = analyzer.calculate_expected_value(y_proba, threshold=0.5)

        assert "expected_value" in ev
        assert "n_should_target" in ev

    def test_calculate_expected_value_respects_threshold(self):
        """Higher thresholds should target fewer customers."""
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        customer_values = np.full(len(y_proba), 500)
        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)

        low_threshold = analyzer.calculate_expected_value(
            y_proba,
            customer_values=customer_values,
            threshold=0.25,
        )
        high_threshold = analyzer.calculate_expected_value(
            y_proba,
            customer_values=customer_values,
            threshold=0.75,
        )

        assert low_threshold["n_should_target"] > high_threshold["n_should_target"]

    def test_net_benefit_calculation(self, sample_predictions):
        """Test net benefit calculation."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50, value_tp=450)
        result = analyzer.analyze(y_true, y_pred, y_proba)

        expected_net = result.total_value - result.total_cost
        assert result.net_benefit == expected_net

    def test_plot_cost_vs_threshold(self, sample_predictions):
        """Test plotting cost vs threshold."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        fig = analyzer.plot_cost_vs_threshold(y_true, y_proba)

        assert fig is not None

    def test_plot_cost_breakdown(self, sample_predictions):
        """Test plotting cost breakdown."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        result = analyzer.analyze(y_true, y_pred, y_proba)
        fig = analyzer.plot_cost_breakdown(result)

        assert fig is not None

    def test_plot_targeting_analysis(self, sample_predictions):
        """Test plotting targeting analysis."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        fig = analyzer.plot_targeting_analysis(y_proba)

        assert fig is not None

    def test_generate_report(self, sample_predictions):
        """Test generating text report."""
        y_true, y_pred, y_proba = sample_predictions

        analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        result = analyzer.analyze(y_true, y_pred, y_proba)
        report = analyzer.generate_report(result)

        assert isinstance(report, str)
        assert "COST" in report.upper()

    def test_high_cost_fn_impact(self, sample_predictions):
        """Test that high FN cost affects optimal threshold."""
        y_true, y_pred, y_proba = sample_predictions

        # Low FN cost - higher threshold acceptable
        analyzer_low = CostAnalyzer(cost_fn=100, cost_fp=50)
        threshold_low = analyzer_low.find_optimal_threshold(y_true, y_proba)

        # High FN cost - lower threshold to catch more churners
        analyzer_high = CostAnalyzer(cost_fn=1000, cost_fp=50)
        threshold_high = analyzer_high.find_optimal_threshold(y_true, y_proba)

        # Higher FN cost should lead to lower threshold (more aggressive targeting)
        assert threshold_high <= threshold_low + 0.1  # Allow some tolerance


class TestCostConfig:
    """Test cases for CostConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CostConfig()

        assert config.cost_fn == 500.0
        assert config.cost_fp == 50.0
        assert config.value_tp == 450.0
        assert config.retention_rate == 0.30

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CostConfig(
            cost_fn=1000,
            cost_fp=100,
            value_tp=900,
            retention_rate=0.4
        )

        assert config.cost_fn == 1000
        assert config.cost_fp == 100
        assert config.value_tp == 900
        assert config.retention_rate == 0.4


class TestRetentionSimulator:
    """Test cases for RetentionSimulator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        y_proba = np.random.beta(2, 8, n)
        customer_values = np.random.lognormal(6, 0.5, n)
        true_churn = np.random.binomial(1, y_proba)
        return y_proba, customer_values, true_churn

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        assert simulator.offer_cost == 50
        assert simulator.success_rate == 0.3

    def test_simulate_campaign_target_high_risk(self, sample_data):
        """Test simulating campaign with target_high_risk strategy."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        result = simulator.simulate_campaign(
            y_proba, customer_values,
            true_churn=true_churn,
            strategy="target_high_risk",
            threshold=0.5
        )

        assert isinstance(result, SimulationResult)
        assert result.n_targeted >= 0
        assert result.n_retained >= 0

    def test_simulate_campaign_target_top_n(self, sample_data):
        """Test simulating campaign with target_top_n strategy."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        result = simulator.simulate_campaign(
            y_proba, customer_values,
            true_churn=true_churn,
            strategy="target_top_n",
            n_target=20
        )

        assert result.n_targeted == 20

    def test_simulate_campaign_budget_constrained(self, sample_data):
        """Test simulating campaign with budget constraint."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        result = simulator.simulate_campaign(
            y_proba, customer_values,
            true_churn=true_churn,
            strategy="budget_constrained",
            budget=500
        )

        # Should target at most 10 customers (500 / 50)
        assert result.total_cost <= 500

    def test_compare_strategies(self, sample_data):
        """Test comparing different strategies."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        comparison = simulator.compare_strategies(
            y_proba, customer_values,
            true_churn=true_churn,
            budget=1000
        )

        assert isinstance(comparison, pd.DataFrame)
        assert "strategy" in comparison.columns
        assert "net_benefit" in comparison.columns
        assert "roi" in comparison.columns

    def test_compare_strategies_deduplicates_threshold_free_strategies(self, sample_data):
        """Strategies that do not use thresholds should not emit duplicate rows."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        comparison = simulator.compare_strategies(
            y_proba,
            customer_values,
            true_churn=true_churn,
            strategies=["target_top_n", "budget_constrained", "threshold_optimized"],
            thresholds=[0.2, 0.4, 0.6],
        )

        counts = comparison.groupby("strategy").size().to_dict()
        assert counts["target_top_n"] == 1
        assert counts["budget_constrained"] == 1
        assert counts["threshold_optimized"] == 1

    def test_threshold_optimized_uses_expected_value(self):
        """Threshold-optimized targeting should use per-customer expected benefit."""
        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        y_proba = np.array([0.2, 0.6, 0.9])
        customer_values = np.array([200, 200, 1000])

        result = simulator.simulate_campaign(
            y_proba,
            customer_values,
            strategy="threshold_optimized",
        )

        assert result.n_targeted == 1

    def test_sensitivity_analysis(self, sample_data):
        """Test sensitivity analysis."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        sensitivity = simulator.sensitivity_analysis(
            y_proba, customer_values,
            true_churn=true_churn,
            threshold=0.5
        )

        assert isinstance(sensitivity, pd.DataFrame)
        assert "success_rate" in sensitivity.columns
        assert "offer_cost" in sensitivity.columns
        assert "net_benefit" in sensitivity.columns

    def test_simulate_time_horizon(self, sample_data):
        """Test time horizon simulation."""
        y_proba, customer_values, _ = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        results = simulator.simulate_time_horizon(
            y_proba, customer_values,
            n_months=6,
            campaign_frequency=2
        )

        assert "monthly_results" in results
        assert "final_retention_rate" in results
        assert len(results["monthly_results"]) == 6

    def test_simulate_time_horizon_respects_threshold(self):
        """Higher campaign thresholds should not target more customers over time."""
        y_proba = np.array([0.2, 0.4, 0.8, 0.9])
        customer_values = np.full(len(y_proba), 500.0)

        low_threshold_sim = RetentionSimulator(offer_cost=50, success_rate=0.3, random_state=42)
        high_threshold_sim = RetentionSimulator(offer_cost=50, success_rate=0.3, random_state=42)

        low_threshold = low_threshold_sim.simulate_time_horizon(
            y_proba,
            customer_values,
            n_months=3,
            campaign_frequency=1,
            threshold=0.3,
        )
        high_threshold = high_threshold_sim.simulate_time_horizon(
            y_proba,
            customer_values,
            n_months=3,
            campaign_frequency=1,
            threshold=0.85,
        )

        assert low_threshold["monthly_results"]["campaign_cost"].sum() >= high_threshold["monthly_results"]["campaign_cost"].sum()

    def test_roi_calculation(self, sample_data):
        """Test ROI calculation."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        result = simulator.simulate_campaign(
            y_proba, customer_values,
            true_churn=true_churn,
            strategy="target_high_risk",
            threshold=0.5
        )

        if result.total_cost > 0:
            expected_roi = (result.revenue_saved / result.total_cost - 1) * 100
            assert abs(result.roi - expected_roi) < 0.01

    def test_retention_rate_achieved(self, sample_data):
        """Test retention rate achieved calculation."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        result = simulator.simulate_campaign(
            y_proba, customer_values,
            true_churn=true_churn,
            strategy="target_high_risk",
            threshold=0.5
        )

        if result.n_would_churn > 0:
            expected_rate = result.n_retained / result.n_would_churn
            assert abs(result.retention_rate_achieved - expected_rate) < 0.01

    def test_plot_strategy_comparison(self, sample_data):
        """Test plotting strategy comparison."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        comparison = simulator.compare_strategies(
            y_proba, customer_values,
            true_churn=true_churn
        )

        fig = simulator.plot_strategy_comparison(comparison)

        assert fig is not None

    def test_plot_sensitivity_heatmap(self, sample_data):
        """Test plotting sensitivity heatmap."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        sensitivity = simulator.sensitivity_analysis(
            y_proba, customer_values,
            true_churn=true_churn
        )

        fig = simulator.plot_sensitivity_heatmap(sensitivity)

        assert fig is not None

    def test_plot_time_simulation(self, sample_data):
        """Test plotting time simulation."""
        y_proba, customer_values, _ = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        time_results = simulator.simulate_time_horizon(
            y_proba, customer_values,
            n_months=6
        )

        fig = simulator.plot_time_simulation(time_results)

        assert fig is not None

    def test_generate_report(self, sample_data):
        """Test generating simulation report."""
        y_proba, customer_values, true_churn = sample_data

        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        result = simulator.simulate_campaign(
            y_proba, customer_values,
            true_churn=true_churn,
            strategy="target_high_risk"
        )

        report = simulator.generate_report(result)

        assert isinstance(report, str)
        assert "SIMULATION" in report.upper()

    def test_no_customers_targeted(self):
        """Test handling case with no customers targeted."""
        simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)

        # Very high threshold should target no one
        y_proba = np.array([0.1, 0.2, 0.3])
        customer_values = np.array([100, 200, 300])

        result = simulator.simulate_campaign(
            y_proba, customer_values,
            strategy="target_high_risk",
            threshold=0.99
        )

        assert result.n_targeted == 0
        assert result.net_benefit == 0


class TestRetentionStrategy:
    """Test cases for RetentionStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategies are defined."""
        strategies = [
            RetentionStrategy.TARGET_HIGH_RISK,
            RetentionStrategy.TARGET_TOP_N,
            RetentionStrategy.TARGET_SEGMENT,
            RetentionStrategy.THRESHOLD_OPTIMIZED,
            RetentionStrategy.BUDGET_CONSTRAINED,
        ]

        for strategy in strategies:
            assert isinstance(strategy.value, str)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_analyze_costs(self):
        """Test analyze_costs function."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.2, 100)
        y_proba = np.clip(y_true + np.random.normal(0, 0.2, 100), 0, 1)
        y_pred = (y_proba >= 0.5).astype(int)

        result, report = analyze_costs(
            y_true, y_pred, y_proba,
            cost_fn=500, cost_fp=50
        )

        assert isinstance(result, CostResult)
        assert isinstance(report, str)

    def test_simulate_retention(self):
        """Test simulate_retention function."""
        np.random.seed(42)
        y_proba = np.random.beta(2, 8, 100)
        customer_values = np.random.lognormal(6, 0.5, 100)

        result, report = simulate_retention(
            y_proba, customer_values,
            offer_cost=50, success_rate=0.3,
            threshold=0.5
        )

        assert isinstance(result, SimulationResult)
        assert isinstance(report, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
