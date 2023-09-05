"""Retention campaign simulation for churn prediction.

Simulates the impact of different retention strategies and campaigns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class RetentionStrategy(Enum):
    """Available retention strategies."""

    TARGET_HIGH_RISK = "target_high_risk"  # Target customers above threshold
    TARGET_TOP_N = "target_top_n"  # Target top N highest risk
    TARGET_SEGMENT = "target_segment"  # Target specific segments
    THRESHOLD_OPTIMIZED = "threshold_optimized"  # Use cost-optimized threshold
    BUDGET_CONSTRAINED = "budget_constrained"  # Maximize under budget


@dataclass
class CampaignConfig:
    """Configuration for retention campaign."""

    name: str = "Retention Campaign"
    offer_value: float = 50.0  # Value of retention offer to customer
    offer_cost: float = 50.0  # Cost to company
    success_rate: float = 0.30  # Base success rate
    success_rate_modifier: float = 0.0  # Additional success for high-value customers


@dataclass
class SimulationResult:
    """Results from retention simulation."""

    strategy: str
    n_targeted: int
    n_would_churn: int
    n_retained: int
    total_cost: float
    revenue_saved: float
    net_benefit: float
    roi: float
    retention_rate_achieved: float
    details: Dict[str, Any] = field(default_factory=dict)


class RetentionSimulator:
    """Simulate retention campaigns and their business impact.

    This class provides:
    - Multiple targeting strategies
    - Campaign ROI simulation
    - Sensitivity analysis
    - Budget optimization
    - What-if scenarios

    Example:
        >>> simulator = RetentionSimulator(offer_cost=50, success_rate=0.3)
        >>> result = simulator.simulate_campaign(y_proba, customer_values, strategy="target_high_risk")
        >>> print(f"Net Benefit: ${result.net_benefit:,.2f}")
    """

    def __init__(
        self,
        offer_cost: float = 50.0,
        offer_value: float = 50.0,
        success_rate: float = 0.30,
        success_rate_by_segment: Optional[Dict[str, float]] = None,
        random_state: int = 42,
    ):
        """Initialize the retention simulator.

        Args:
            offer_cost: Cost of retention offer to the company.
            offer_value: Value of offer to the customer.
            success_rate: Base success rate of retention offers.
            success_rate_by_segment: Success rates by customer segment.
            random_state: Random seed for reproducibility.
        """
        self.offer_cost = offer_cost
        self.offer_value = offer_value
        self.success_rate = success_rate
        self.success_rate_by_segment = success_rate_by_segment or {}
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def simulate_campaign(
        self,
        y_proba: np.ndarray,
        customer_values: np.ndarray,
        true_churn: Optional[np.ndarray] = None,
        strategy: str = "target_high_risk",
        threshold: float = 0.5,
        budget: Optional[float] = None,
        n_target: Optional[int] = None,
        customer_segments: Optional[np.ndarray] = None,
        n_simulations: int = 100,
    ) -> SimulationResult:
        """Simulate a retention campaign.

        Args:
            y_proba: Predicted churn probabilities.
            customer_values: Customer lifetime values.
            true_churn: Actual churn labels (for ground truth simulation).
            strategy: Targeting strategy.
            threshold: Probability threshold for targeting.
            budget: Budget constraint.
            n_target: Number of customers to target.
            customer_segments: Customer segment labels.
            n_simulations: Number of Monte Carlo simulations.

        Returns:
            SimulationResult with campaign outcomes.
        """
        # Determine which customers to target
        targeted_mask = self._get_target_mask(
            y_proba,
            customer_values,
            strategy,
            threshold,
            budget,
            n_target,
            customer_segments,
        )

        targeted_indices = np.where(targeted_mask)[0]
        n_targeted = len(targeted_indices)

        if n_targeted == 0:
            return SimulationResult(
                strategy=strategy,
                n_targeted=0,
                n_would_churn=0,
                n_retained=0,
                total_cost=0,
                revenue_saved=0,
                net_benefit=0,
                roi=0,
                retention_rate_achieved=0,
                details={"message": "No customers targeted"},
            )

        # Get target customer data
        targeted_proba = y_proba[targeted_mask]
        targeted_values = customer_values[targeted_mask]

        # Determine who would actually churn (if ground truth not provided)
        if true_churn is not None:
            would_churn = true_churn[targeted_mask]
        else:
            # Simulate based on probabilities
            would_churn = self.rng.random(n_targeted) < targeted_proba

        n_would_churn = would_churn.sum()

        # Simulate retention success
        retention_success = self._simulate_retention(
            would_churn, targeted_values, customer_segments, targeted_indices, n_simulations
        )

        # Calculate metrics
        avg_retained = retention_success.mean(axis=0)
        n_retained = int(avg_retained.sum())

        total_cost = n_targeted * self.offer_cost

        # Revenue saved = retained customers * their value
        revenue_saved = np.sum(avg_retained * targeted_values)

        net_benefit = revenue_saved - total_cost
        roi = (revenue_saved / total_cost - 1) * 100 if total_cost > 0 else 0

        # Calculate actual retention rate
        retention_rate_achieved = n_retained / n_would_churn if n_would_churn > 0 else 0

        return SimulationResult(
            strategy=strategy,
            n_targeted=n_targeted,
            n_would_churn=int(n_would_churn),
            n_retained=n_retained,
            total_cost=total_cost,
            revenue_saved=revenue_saved,
            net_benefit=net_benefit,
            roi=roi,
            retention_rate_achieved=retention_rate_achieved,
            details={
                "threshold_used": threshold,
                "avg_targeted_probability": targeted_proba.mean(),
                "avg_targeted_value": targeted_values.mean(),
                "simulation_uncertainty": retention_success.std(axis=0).mean(),
            },
        )

    def _get_target_mask(
        self,
        y_proba: np.ndarray,
        customer_values: np.ndarray,
        strategy: str,
        threshold: float,
        budget: Optional[float],
        n_target: Optional[int],
        customer_segments: Optional[np.ndarray],
    ) -> np.ndarray:
        """Determine which customers to target based on strategy.

        Args:
            y_proba: Predicted probabilities.
            strategy: Targeting strategy.
            threshold: Probability threshold.
            budget: Budget constraint.
            n_target: Number to target.
            customer_segments: Segment labels.

        Returns:
            Boolean mask of customers to target.
        """
        n_customers = len(y_proba)
        strategy_enum = RetentionStrategy(strategy)

        if strategy_enum == RetentionStrategy.TARGET_HIGH_RISK:
            return y_proba >= threshold

        elif strategy_enum == RetentionStrategy.TARGET_TOP_N:
            n_target = n_target or int(n_customers * 0.1)
            top_indices = np.argsort(y_proba)[-n_target:]
            mask = np.zeros(n_customers, dtype=bool)
            mask[top_indices] = True
            return mask

        elif strategy_enum == RetentionStrategy.THRESHOLD_OPTIMIZED:
            expected_net_benefit = (
                y_proba * self.success_rate * customer_values
            ) - self.offer_cost
            return expected_net_benefit > 0

        elif strategy_enum == RetentionStrategy.BUDGET_CONSTRAINED:
            budget = budget or 10000
            n_can_afford = int(budget / self.offer_cost)
            top_indices = np.argsort(y_proba)[-n_can_afford:]
            mask = np.zeros(n_customers, dtype=bool)
            mask[top_indices] = True
            return mask

        elif strategy_enum == RetentionStrategy.TARGET_SEGMENT:
            if customer_segments is None:
                return y_proba >= threshold
            # Target only high-risk segments
            return (y_proba >= threshold)  # Simplified

        else:
            return y_proba >= threshold

    def _strategy_thresholds(
        self,
        strategy: str,
        thresholds: List[float],
    ) -> List[float]:
        """Return the thresholds that are relevant to a strategy."""
        strategy_enum = RetentionStrategy(strategy)
        threshold_free_strategies = {
            RetentionStrategy.TARGET_TOP_N,
            RetentionStrategy.THRESHOLD_OPTIMIZED,
            RetentionStrategy.BUDGET_CONSTRAINED,
        }

        if strategy_enum in threshold_free_strategies:
            return [thresholds[0]]

        return thresholds

    def _simulate_retention(
        self,
        would_churn: np.ndarray,
        customer_values: np.ndarray,
        customer_segments: Optional[np.ndarray],
        targeted_indices: np.ndarray,
        n_simulations: int,
    ) -> np.ndarray:
        """Simulate retention outcomes using Monte Carlo.

        Args:
            would_churn: Which customers would actually churn.
            customer_values: Customer values.
            customer_segments: Segment labels.
            targeted_indices: Indices of targeted customers.
            n_simulations: Number of simulations.

        Returns:
            Array of shape (n_simulations, n_targeted) with retention outcomes.
        """
        n_targeted = len(would_churn)
        results = np.zeros((n_simulations, n_targeted))

        for sim in range(n_simulations):
            for i in range(n_targeted):
                # Only those who would churn can be retained
                if not would_churn[i]:
                    continue

                # Base success rate
                success_rate = self.success_rate

                # Modify by segment if available
                if customer_segments is not None and len(self.success_rate_by_segment) > 0:
                    segment = customer_segments[targeted_indices[i]]
                    if segment in self.success_rate_by_segment:
                        success_rate = self.success_rate_by_segment[segment]

                # Higher value customers might respond better
                value_modifier = min(customer_values[i] / 500, 0.5) * 0.1
                success_rate = min(success_rate + value_modifier, 0.8)

                # Simulate
                results[sim, i] = self.rng.random() < success_rate

        return results

    def compare_strategies(
        self,
        y_proba: np.ndarray,
        customer_values: np.ndarray,
        true_churn: Optional[np.ndarray] = None,
        strategies: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        budget: float = 10000,
    ) -> pd.DataFrame:
        """Compare different targeting strategies.

        Args:
            y_proba: Predicted probabilities.
            customer_values: Customer values.
            true_churn: Actual churn labels.
            strategies: Strategies to compare.
            thresholds: Thresholds to test.
            budget: Budget for constrained strategies.

        Returns:
            DataFrame comparing strategy outcomes.
        """
        strategies = strategies or [
            "target_high_risk",
            "target_top_n",
            "threshold_optimized",
            "budget_constrained",
        ]
        thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]

        results = []

        for strategy in strategies:
            for threshold in self._strategy_thresholds(strategy, thresholds):
                result = self.simulate_campaign(
                    y_proba,
                    customer_values,
                    true_churn,
                    strategy=strategy,
                    threshold=threshold,
                    budget=budget,
                )

                results.append({
                    "strategy": strategy,
                    "threshold": threshold,
                    "n_targeted": result.n_targeted,
                    "n_retained": result.n_retained,
                    "total_cost": result.total_cost,
                    "revenue_saved": result.revenue_saved,
                    "net_benefit": result.net_benefit,
                    "roi": result.roi,
                    "retention_rate": result.retention_rate_achieved,
                })

        df = pd.DataFrame(results)
        df = df.sort_values("net_benefit", ascending=False).reset_index(drop=True)

        return df

    def sensitivity_analysis(
        self,
        y_proba: np.ndarray,
        customer_values: np.ndarray,
        true_churn: Optional[np.ndarray] = None,
        success_rates: Optional[List[float]] = None,
        offer_costs: Optional[List[float]] = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on campaign parameters.

        Args:
            y_proba: Predicted probabilities.
            customer_values: Customer values.
            true_churn: Actual churn labels.
            success_rates: Success rates to test.
            offer_costs: Offer costs to test.
            threshold: Classification threshold.

        Returns:
            DataFrame with sensitivity analysis results.
        """
        success_rates = success_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
        offer_costs = offer_costs or [25, 50, 75, 100, 150]

        original_success = self.success_rate
        original_cost = self.offer_cost

        results = []

        for success_rate in success_rates:
            for offer_cost in offer_costs:
                # Temporarily modify parameters
                self.success_rate = success_rate
                self.offer_cost = offer_cost

                result = self.simulate_campaign(
                    y_proba,
                    customer_values,
                    true_churn,
                    strategy="target_high_risk",
                    threshold=threshold,
                )

                results.append({
                    "success_rate": success_rate,
                    "offer_cost": offer_cost,
                    "n_targeted": result.n_targeted,
                    "n_retained": result.n_retained,
                    "total_cost": result.total_cost,
                    "net_benefit": result.net_benefit,
                    "roi": result.roi,
                })

        # Restore original parameters
        self.success_rate = original_success
        self.offer_cost = original_cost

        return pd.DataFrame(results)

    def simulate_time_horizon(
        self,
        y_proba: np.ndarray,
        customer_values: np.ndarray,
        n_months: int = 12,
        monthly_churn_baseline: float = 0.02,
        campaign_frequency: int = 3,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Simulate retention over a time horizon.

        Args:
            y_proba: Initial churn probabilities.
            customer_values: Customer values.
            n_months: Number of months to simulate.
            monthly_churn_baseline: Baseline monthly churn rate.
            campaign_frequency: Months between campaigns.
            threshold: Targeting threshold for campaign months.

        Returns:
            Dictionary with time-series simulation results.
        """
        n_customers = len(y_proba)
        active_customers = np.ones(n_customers, dtype=bool)
        cumulative_revenue = 0
        cumulative_cost = 0
        cumulative_retained = 0

        monthly_results = []

        for month in range(1, n_months + 1):
            # Monthly revenue from active customers
            monthly_revenue = active_customers.sum() * customer_values.mean() / 12
            cumulative_revenue += monthly_revenue

            # Natural churn
            churn_mask = self.rng.random(n_customers) < monthly_churn_baseline
            churn_mask = churn_mask & active_customers

            # Run campaign every campaign_frequency months
            if month % campaign_frequency == 0:
                # Update probabilities for remaining customers
                current_proba = y_proba.copy()
                current_proba[~active_customers] = 0

                # Target high-risk customers
                target_mask = (current_proba >= threshold) & active_customers

                # Simulate campaign
                n_targeted = target_mask.sum()
                campaign_cost = n_targeted * self.offer_cost
                cumulative_cost += campaign_cost

                # Retention attempts on those who would churn
                would_churn = self.rng.random(n_targeted) < current_proba[target_mask]
                n_retained = (would_churn & (self.rng.random(n_targeted) < self.success_rate)).sum()
                cumulative_retained += n_retained

                # Update active customers (those who churned and weren't retained)
                newly_churned = would_churn & (self.rng.random(n_targeted) >= self.success_rate)
                target_indices = np.where(target_mask)[0]
                active_customers[target_indices[newly_churned]] = False
            else:
                # Apply natural churn
                active_customers[churn_mask] = False
                n_targeted = 0
                campaign_cost = 0
                n_retained = 0

            monthly_results.append({
                "month": month,
                "active_customers": active_customers.sum(),
                "retention_rate": active_customers.mean(),
                "monthly_revenue": monthly_revenue,
                "campaign_cost": campaign_cost,
                "n_retained": n_retained,
                "cumulative_revenue": cumulative_revenue,
                "cumulative_cost": cumulative_cost,
                "cumulative_retained": cumulative_retained,
            })

        df = pd.DataFrame(monthly_results)
        df["net_benefit"] = df["cumulative_revenue"] - df["cumulative_cost"]

        return {
            "monthly_results": df,
            "final_retention_rate": active_customers.mean(),
            "total_revenue": cumulative_revenue,
            "total_cost": cumulative_cost,
            "total_net_benefit": cumulative_revenue - cumulative_cost,
            "total_customers_retained": cumulative_retained,
        }

    def plot_strategy_comparison(
        self,
        comparison_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot strategy comparison.

        Args:
            comparison_df: DataFrame from compare_strategies().
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        # Group by strategy
        strategies = comparison_df["strategy"].unique()

        x = np.arange(len(strategies))
        width = 0.35

        net_benefits = [comparison_df[comparison_df["strategy"] == s]["net_benefit"].max()
                        for s in strategies]
        rois = [comparison_df[comparison_df["strategy"] == s]["roi"].max()
                for s in strategies]

        bars1 = ax.bar(x - width/2, net_benefits, width, label="Net Benefit ($)", color="steelblue")
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, rois, width, label="ROI (%)", color="darkorange")

        ax.set_xlabel("Strategy")
        ax.set_ylabel("Net Benefit ($)", color="steelblue")
        ax2.set_ylabel("ROI (%)", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=45, ha="right")
        ax.set_title("Retention Strategy Comparison")

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f"${height:,.0f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=8)

        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_sensitivity_heatmap(
        self,
        sensitivity_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot sensitivity analysis heatmap.

        Args:
            sensitivity_df: DataFrame from sensitivity_analysis().
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Pivot for heatmap
        pivot = sensitivity_df.pivot(
            index="success_rate",
            columns="offer_cost",
            values="net_benefit"
        )

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

        # Labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"${c}" for c in pivot.columns])
        ax.set_yticklabels([f"{r:.0%}" for r in pivot.index])

        ax.set_xlabel("Offer Cost")
        ax.set_ylabel("Success Rate")
        ax.set_title("Net Benefit by Success Rate and Offer Cost")

        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                color = "white" if abs(value) > pivot.values.max() / 2 else "black"
                text = ax.text(j, i, f"${value:,.0f}",
                              ha="center", va="center", color=color, fontsize=9)

        fig.colorbar(im, ax=ax, label="Net Benefit ($)")
        plt.tight_layout()

        return fig

    def plot_time_simulation(
        self,
        time_results: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot time horizon simulation results.

        Args:
            time_results: Results from simulate_time_horizon().
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        df = time_results["monthly_results"]

        ax.plot(df["month"], df["cumulative_revenue"], "g-", label="Cumulative Revenue", linewidth=2)
        ax.plot(df["month"], df["cumulative_cost"], "r-", label="Cumulative Cost", linewidth=2)
        ax.plot(df["month"], df["net_benefit"], "b-", label="Net Benefit", linewidth=2)

        ax.fill_between(df["month"], df["cumulative_cost"], df["cumulative_revenue"],
                        where=df["cumulative_revenue"] > df["cumulative_cost"],
                        alpha=0.3, color="green", label="Profit Zone")

        ax.set_xlabel("Month")
        ax.set_ylabel("Amount ($)")
        ax.set_title(f"Retention Simulation Over {len(df)} Months")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add final metrics
        final_text = f"Final Retention: {time_results['final_retention_rate']:.1%}\n"
        final_text += f"Total Net Benefit: ${time_results['total_net_benefit']:,.0f}"
        ax.text(0.02, 0.98, final_text, transform=ax.transAxes,
               verticalalignment="top", fontsize=10,
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        return fig

    def generate_report(
        self,
        result: SimulationResult,
    ) -> str:
        """Generate a text simulation report.

        Args:
            result: SimulationResult object.

        Returns:
            Formatted report string.
        """
        report = []
        report.append("=" * 60)
        report.append("RETENTION CAMPAIGN SIMULATION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("CAMPAIGN PARAMETERS")
        report.append("-" * 40)
        report.append(f"Strategy: {result.strategy.replace('_', ' ').title()}")
        report.append(f"Offer Cost: ${self.offer_cost:,.2f}")
        report.append(f"Success Rate: {self.success_rate:.0%}")
        report.append("")

        report.append("SIMULATION RESULTS")
        report.append("-" * 40)
        report.append(f"Customers Targeted: {result.n_targeted:,}")
        report.append(f"Would-Have Churned: {result.n_would_churn:,}")
        report.append(f"Successfully Retained: {result.n_retained:,}")
        report.append(f"Retention Rate Achieved: {result.retention_rate_achieved:.1%}")
        report.append("")

        report.append("FINANCIAL IMPACT")
        report.append("-" * 40)
        report.append(f"Total Campaign Cost: ${result.total_cost:,.2f}")
        report.append(f"Revenue Saved: ${result.revenue_saved:,.2f}")
        report.append(f"Net Benefit: ${result.net_benefit:,.2f}")
        report.append(f"ROI: {result.roi:.1f}%")
        report.append("")

        # Verdict
        report.append("VERDICT")
        report.append("-" * 40)
        if result.net_benefit > 0:
            report.append("SUCCESS: Campaign generates positive return!")
            report.append(f"For every $1 spent, you gain ${1 + result.roi/100:.2f} in value.")
        else:
            report.append("WARNING: Campaign results in net loss.")
            report.append("Consider adjusting offer cost or targeting strategy.")

        return "\n".join(report)


def simulate_retention(
    y_proba: np.ndarray,
    customer_values: np.ndarray,
    offer_cost: float = 50.0,
    success_rate: float = 0.30,
    threshold: float = 0.5,
) -> Tuple[SimulationResult, str]:
    """Convenience function to run retention simulation.

    Args:
        y_proba: Predicted churn probabilities.
        customer_values: Customer lifetime values.
        offer_cost: Cost of retention offer.
        success_rate: Success rate of retention.
        threshold: Targeting threshold.

    Returns:
        Tuple of (result, report_string).
    """
    simulator = RetentionSimulator(offer_cost=offer_cost, success_rate=success_rate)
    result = simulator.simulate_campaign(
        y_proba, customer_values,
        strategy="target_high_risk",
        threshold=threshold,
    )
    report = simulator.generate_report(result)

    return result, report


if __name__ == "__main__":
    # Test retention simulator
    np.random.seed(42)

    n = 1000
    y_proba = np.random.beta(2, 8, n)  # Skewed toward low churn
    customer_values = np.random.lognormal(6, 0.5, n)  # CLV distribution

    simulator = RetentionSimulator(offer_cost=50, success_rate=0.30)
    result = simulator.simulate_campaign(
        y_proba, customer_values,
        strategy="target_high_risk",
        threshold=0.5,
    )

    print(simulator.generate_report(result))

    # Compare strategies
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    comparison = simulator.compare_strategies(y_proba, customer_values)
    print(comparison[["strategy", "threshold", "net_benefit", "roi"]].head())
