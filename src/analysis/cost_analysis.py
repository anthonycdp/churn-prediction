"""Cost of error analysis for churn prediction.

Analyzes the business impact of prediction errors and optimal decision thresholds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CostConfig:
    """Configuration for cost analysis."""

    # Cost of false negative (missed churner)
    # This is typically the customer lifetime value we lose
    cost_fn: float = 500.0

    # Cost of false positive (unnecessary retention action)
    # This includes retention offer cost, marketing spend, etc.
    cost_fp: float = 50.0

    # Value of true positive (successfully retained customer)
    # Net benefit: saved CLV minus retention cost
    value_tp: float = 450.0

    # Value of true negative (no action needed)
    value_tn: float = 0.0

    # Retention campaign success rate
    retention_rate: float = 0.30

    # Monthly budget for retention campaigns
    monthly_budget: float = 10000.0


@dataclass
class CostResult:
    """Results from cost analysis."""

    total_cost: float
    total_value: float
    net_benefit: float
    cost_per_customer: float
    confusion_matrix_cost: np.ndarray
    cost_breakdown: Dict[str, float]
    optimal_threshold: float
    threshold_analysis: pd.DataFrame


class CostAnalyzer:
    """Analyze the business cost of model prediction errors.

    This class provides:
    - Cost-based evaluation of predictions
    - Optimal threshold selection based on costs
    - Budget-constrained targeting optimization
    - ROI analysis of retention campaigns

    Example:
        >>> analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
        >>> result = analyzer.analyze(y_true, y_pred, y_proba)
        >>> optimal_threshold = analyzer.find_optimal_threshold(y_true, y_proba)
    """

    def __init__(
        self,
        cost_fn: float = 500.0,
        cost_fp: float = 50.0,
        value_tp: float = 450.0,
        value_tn: float = 0.0,
        retention_rate: float = 0.30,
        monthly_budget: float = 10000.0,
    ):
        """Initialize the cost analyzer.

        Args:
            cost_fn: Cost of false negative (missed churner).
            cost_fp: Cost of false positive (unnecessary action).
            value_tp: Net value of true positive (retained customer).
            value_tn: Value of true negative.
            retention_rate: Success rate of retention campaigns.
            monthly_budget: Available monthly budget for retention.
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.value_tp = value_tp
        self.value_tn = value_tn
        self.retention_rate = retention_rate
        self.monthly_budget = monthly_budget

    def calculate_cost_matrix(self) -> np.ndarray:
        """Calculate the cost matrix.

        Returns:
            2x2 cost matrix [[TN_cost, FP_cost], [FN_cost, TP_cost]].
        """
        return np.array([
            [self.value_tn, -self.cost_fp],      # True Neg, False Pos
            [-self.cost_fn, self.value_tp],      # False Neg, True Pos
        ])

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> CostResult:
        """Analyze the cost of predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional, for threshold analysis).
            sample_weight: Optional sample weights.

        Returns:
            CostResult with detailed cost analysis.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate costs
        cost_matrix = self.calculate_cost_matrix()

        # Apply sample weights if provided
        if sample_weight is not None:
            weighted_cm = self._weighted_confusion_matrix(
                y_true, y_pred, sample_weight
            )
        else:
            weighted_cm = cm.astype(float)

        # Calculate total cost/value
        total_cost = fn * self.cost_fn + fp * self.cost_fp
        total_value = tp * self.value_tp + tn * self.value_tn
        net_benefit = total_value - total_cost

        n_samples = len(y_true)
        cost_per_customer = net_benefit / n_samples

        # Cost breakdown
        cost_breakdown = {
            "false_negatives_cost": fn * self.cost_fn,
            "false_positives_cost": fp * self.cost_fp,
            "true_positives_value": tp * self.value_tp,
            "true_negatives_value": tn * self.value_tn,
            "false_negatives_count": int(fn),
            "false_positives_count": int(fp),
            "true_positives_count": int(tp),
            "true_negatives_count": int(tn),
        }

        # Threshold analysis
        threshold_analysis = None
        optimal_threshold = 0.5

        if y_proba is not None:
            threshold_analysis = self._analyze_thresholds(y_true, y_proba)
            optimal_threshold = self.find_optimal_threshold(y_true, y_proba)

        return CostResult(
            total_cost=total_cost,
            total_value=total_value,
            net_benefit=net_benefit,
            cost_per_customer=cost_per_customer,
            confusion_matrix_cost=weighted_cm,
            cost_breakdown=cost_breakdown,
            optimal_threshold=optimal_threshold,
            threshold_analysis=threshold_analysis,
        )

    def _weighted_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray,
    ) -> np.ndarray:
        """Calculate weighted confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            sample_weight: Sample weights.

        Returns:
            Weighted confusion matrix.
        """
        cm = np.zeros((2, 2))
        for i in range(len(y_true)):
            cm[int(y_true[i]), int(y_pred[i])] += sample_weight[i]
        return cm

    def _analyze_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Analyze costs across different thresholds.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            thresholds: Thresholds to evaluate.

        Returns:
            DataFrame with cost analysis per threshold.
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            total_cost = fn * self.cost_fn + fp * self.cost_fp
            total_value = tp * self.value_tp + tn * self.value_tn
            net_benefit = total_value - total_cost

            results.append({
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_cost": total_cost,
                "total_value": total_value,
                "net_benefit": net_benefit,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            })

        return pd.DataFrame(results)

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "net_benefit",
    ) -> float:
        """Find optimal classification threshold based on costs.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            metric: Metric to optimize ('net_benefit', 'cost', 'roi').

        Returns:
            Optimal threshold value.
        """
        threshold_df = self._analyze_thresholds(y_true, y_proba)

        if metric == "net_benefit":
            best_idx = threshold_df["net_benefit"].idxmax()
        elif metric == "cost":
            best_idx = threshold_df["total_cost"].idxmin()
        elif metric == "roi":
            # ROI = value / cost
            threshold_df["roi"] = threshold_df["total_value"] / threshold_df["total_cost"].replace(0, 1)
            best_idx = threshold_df["roi"].idxmax()
        else:
            best_idx = threshold_df["net_benefit"].idxmax()

        return threshold_df.loc[best_idx, "threshold"]

    def optimize_targeting(
        self,
        y_proba: np.ndarray,
        customer_values: Optional[np.ndarray] = None,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Optimize customer targeting under budget constraints.

        Args:
            y_proba: Predicted churn probabilities.
            customer_values: Customer lifetime values (uses default if None).
            budget: Budget constraint (uses monthly_budget if None).

        Returns:
            Dictionary with targeting recommendations.
        """
        budget = budget or self.monthly_budget
        n_customers = len(y_proba)

        # Use default CLV if not provided
        if customer_values is None:
            customer_values = np.full(n_customers, self.cost_fn)

        # Sort by churn probability (highest first)
        sorted_indices = np.argsort(y_proba)[::-1]

        # Calculate cumulative costs
        targeted_customers = []
        total_cost = 0
        expected_value = 0

        for idx in sorted_indices:
            cost = self.cost_fp  # Cost of targeting
            if total_cost + cost > budget:
                break

            total_cost += cost
            # Expected value = retention_rate * value - cost
            exp_val = self.retention_rate * customer_values[idx] - self.cost_fp
            expected_value += exp_val

            targeted_customers.append({
                "index": int(idx),
                "churn_probability": y_proba[idx],
                "customer_value": customer_values[idx],
                "expected_value": exp_val,
            })

        return {
            "n_targeted": len(targeted_customers),
            "total_cost": total_cost,
            "expected_value": expected_value,
            "expected_roi": expected_value / total_cost if total_cost > 0 else 0,
            "budget_used_pct": total_cost / budget * 100,
            "targeted_customers": targeted_customers,
            "threshold_used": y_proba[sorted_indices[len(targeted_customers) - 1]] if targeted_customers else 1.0,
        }

    def calculate_expected_value(
        self,
        y_proba: np.ndarray,
        customer_values: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate expected business value of predictions.

        Args:
            y_proba: Predicted churn probabilities.
            customer_values: Customer lifetime values.
            threshold: Classification threshold.

        Returns:
            Dictionary with expected value metrics.
        """
        if customer_values is None:
            customer_values = np.full(len(y_proba), self.cost_fn)

        target_mask = y_proba >= threshold

        # Expected value of targeting each customer
        expected_benefit = (
            y_proba * self.retention_rate * customer_values -  # Probability of retention * value
            self.cost_fp  # Cost of targeting
        )

        # Only target customers above the operating threshold with positive expected value.
        should_target = target_mask & (expected_benefit > 0)

        return {
            "expected_value": expected_benefit[should_target].sum(),
            "n_should_target": should_target.sum(),
            "targeting_rate": should_target.mean(),
            "avg_expected_benefit": expected_benefit[should_target].mean() if should_target.any() else 0,
            "max_expected_benefit": expected_benefit.max(),
        }

    def plot_cost_vs_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot cost metrics vs threshold.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        threshold_df = self._analyze_thresholds(y_true, y_proba)
        optimal_threshold = self.find_optimal_threshold(y_true, y_proba)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        ax.plot(threshold_df["threshold"], threshold_df["total_cost"], "r-", label="Total Cost", linewidth=2)
        ax.plot(threshold_df["threshold"], threshold_df["total_value"], "g-", label="Total Value", linewidth=2)
        ax.plot(threshold_df["threshold"], threshold_df["net_benefit"], "b-", label="Net Benefit", linewidth=2)
        ax.axvline(x=optimal_threshold, color="red", linestyle="--", label=f"Optimal ({optimal_threshold:.2f})")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Amount ($)")
        ax.set_title("Cost Analysis vs Classification Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_cost_breakdown(
        self,
        result: CostResult,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot cost breakdown pie chart.

        Args:
            result: CostResult from analyze().
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        breakdown = result.cost_breakdown

        labels = [
            f"False Negatives\n({breakdown['false_negatives_count']} @ ${self.cost_fn})",
            f"False Positives\n({breakdown['false_positives_count']} @ ${self.cost_fp})",
            f"True Positives\n({breakdown['true_positives_count']} @ ${self.value_tp})",
        ]

        sizes = [
            breakdown["false_negatives_cost"],
            breakdown["false_positives_cost"],
            breakdown["true_positives_value"],
        ]

        colors = ["#ff6b6b", "#ffd93d", "#6bcb77"]
        explode = (0.05, 0.05, 0)

        # Only show non-zero values
        non_zero_mask = [s > 0 for s in sizes]
        labels = [l for l, m in zip(labels, non_zero_mask) if m]
        sizes = [s for s, m in zip(sizes, non_zero_mask) if m]
        colors = [c for c, m in zip(colors, non_zero_mask) if m]
        explode = tuple(e for e, m in zip(explode, non_zero_mask) if m)

        ax.pie(sizes, labels=labels, colors=colors, explode=explode,
               autopct="%1.1f%%", shadow=True, startangle=90)
        ax.set_title("Cost/Value Breakdown")

        return fig

    def plot_targeting_analysis(
        self,
        y_proba: np.ndarray,
        customer_values: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot targeting analysis.

        Args:
            y_proba: Predicted probabilities.
            customer_values: Customer values.
            ax: Matplotlib axes.

        Returns:
            Matplotlib figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if customer_values is None:
            customer_values = np.full(len(y_proba), self.cost_fn)

        # Sort by probability
        sorted_idx = np.argsort(y_proba)[::-1]
        sorted_proba = y_proba[sorted_idx]

        # Cumulative cost and value
        n_range = np.arange(1, len(y_proba) + 1)
        cumulative_cost = n_range * self.cost_fp

        # Expected retained value
        expected_retained = np.cumsum(
            sorted_proba * self.retention_rate * customer_values[sorted_idx]
        )
        net_value = expected_retained - cumulative_cost

        # Plot
        ax.plot(n_range, cumulative_cost, "r-", label="Cumulative Cost", linewidth=2)
        ax.plot(n_range, expected_retained, "g-", label="Expected Value", linewidth=2)
        ax.plot(n_range, net_value, "b-", label="Net Value", linewidth=2)

        # Mark optimal point
        optimal_n = np.argmax(net_value)
        ax.axvline(x=optimal_n, color="red", linestyle="--", alpha=0.5)
        ax.scatter([optimal_n], [net_value[optimal_n]], c="red", s=100, zorder=5)

        # Budget line
        ax.axhline(y=self.monthly_budget, color="orange", linestyle=":",
                   label=f"Budget (${self.monthly_budget:,.0f})")

        ax.set_xlabel("Number of Customers Targeted")
        ax.set_ylabel("Value ($)")
        ax.set_title(f"Targeting Analysis (Optimal: {optimal_n} customers)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def generate_report(
        self,
        result: CostResult,
    ) -> str:
        """Generate a text cost analysis report.

        Args:
            result: CostResult from analyze().

        Returns:
            Formatted report string.
        """
        report = []
        report.append("=" * 60)
        report.append("COST OF ERROR ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("COST PARAMETERS")
        report.append("-" * 40)
        report.append(f"Cost of False Negative:  ${self.cost_fn:,.2f}")
        report.append(f"Cost of False Positive: ${self.cost_fp:,.2f}")
        report.append(f"Value of True Positive:  ${self.value_tp:,.2f}")
        report.append(f"Retention Success Rate:  {self.retention_rate:.0%}")
        report.append("")

        report.append("RESULTS SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Cost:   ${result.total_cost:,.2f}")
        report.append(f"Total Value:  ${result.total_value:,.2f}")
        report.append(f"Net Benefit:  ${result.net_benefit:,.2f}")
        report.append(f"Per Customer: ${result.cost_per_customer:,.2f}")
        report.append("")

        report.append("CONFUSION MATRIX BREAKDOWN")
        report.append("-" * 40)
        cb = result.cost_breakdown
        report.append(f"True Positives:  {cb['true_positives_count']:>5} (${cb['true_positives_value']:>10,.2f})")
        report.append(f"True Negatives:  {cb['true_negatives_count']:>5} (${cb['true_negatives_value']:>10,.2f})")
        report.append(f"False Positives: {cb['false_positives_count']:>5} (${cb['false_positives_cost']:>10,.2f})")
        report.append(f"False Negatives: {cb['false_negatives_count']:>5} (${cb['false_negatives_cost']:>10,.2f})")
        report.append("")

        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append(f"Optimal Threshold: {result.optimal_threshold:.2f}")

        if result.net_benefit > 0:
            report.append("POSITIVE: Model provides net business value!")
            roi = (result.total_value / result.total_cost - 1) * 100 if result.total_cost > 0 else float('inf')
            report.append(f"ROI: {roi:.1f}%")
        else:
            report.append("WARNING: Current threshold results in net loss.")
            report.append("Consider adjusting threshold or cost parameters.")

        return "\n".join(report)


def analyze_costs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
) -> Tuple[CostResult, str]:
    """Convenience function to analyze prediction costs.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        cost_fn: Cost of false negative.
        cost_fp: Cost of false positive.

    Returns:
        Tuple of (result, report_string).
    """
    analyzer = CostAnalyzer(cost_fn=cost_fn, cost_fp=cost_fp)
    result = analyzer.analyze(y_true, y_pred, y_proba)
    report = analyzer.generate_report(result)

    return result, report


if __name__ == "__main__":
    # Test cost analyzer
    np.random.seed(42)

    # Generate synthetic predictions
    n = 1000
    y_true = np.random.binomial(1, 0.2, n)
    y_proba = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)
    y_pred = (y_proba >= 0.5).astype(int)

    # Analyze
    analyzer = CostAnalyzer(cost_fn=500, cost_fp=50, value_tp=450)
    result = analyzer.analyze(y_true, y_pred, y_proba)

    print(analyzer.generate_report(result))
    print(f"\nOptimal threshold: {result.optimal_threshold:.2f}")
