#!/usr/bin/env python3
"""Main entry point for the Churn Prediction pipeline.

This script demonstrates the complete workflow:
1. Generate/load data
2. Preprocess features
3. Train multiple models
4. Evaluate and compare models
5. Generate SHAP explanations
6. Analyze costs
7. Simulate retention campaigns
8. Generate reports and visualizations

Usage:
    python main.py [--n-samples N] [--output-dir DIR] [--tune] [--skip-shap]
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.generator import ChurnDataGenerator
from src.data.preprocessor import ChurnDataPreprocessor
from src.models.trainer import ChurnModelTrainer
from src.models.evaluator import ChurnModelEvaluator
from src.explainability.shap_analyzer import SHAPAnalyzer
from src.analysis.cost_analysis import CostAnalyzer
from src.analysis.retention_simulation import RetentionSimulator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run churn prediction pipeline"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP analysis (faster)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=500.0,
        help="Cost of false negative (default: 500)",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=50.0,
        help="Cost of false positive (default: 50)",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of training data reserved for threshold/model selection (default: 0.2)",
    )
    return parser.parse_args()


def setup_output_directories(base_dir: str) -> dict:
    """Create output directory structure."""
    dirs = {
        "base": base_dir,
        "figures": os.path.join(base_dir, "figures"),
        "models": os.path.join(base_dir, "models"),
        "reports": os.path.join(base_dir, "reports"),
        "data": os.path.join(base_dir, "data"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs


def generate_data(args, dirs: dict) -> pd.DataFrame:
    """Generate sample data."""
    print("\n" + "=" * 60)
    print("STEP 1: GENERATING DATA")
    print("=" * 60)

    generator = ChurnDataGenerator(random_state=args.random_state)
    df = generator.generate(n_samples=args.n_samples)

    # Save raw data
    data_path = os.path.join(dirs["data"], "churn_data.csv")
    df.to_csv(data_path, index=False)

    print(f"Generated {len(df)} samples")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"Data saved to: {data_path}")

    return df


def preprocess_data(df: pd.DataFrame, args) -> tuple:
    """Preprocess data for modeling."""
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 60)

    preprocessor = ChurnDataPreprocessor(
        target_column="churn",
        test_size=0.2,
        random_state=args.random_state,
    )

    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


def split_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    validation_size: float,
    random_state: int,
) -> tuple:
    """Create a validation split for model selection without touching the test set."""
    print("\n" + "=" * 60)
    print("STEP 3: SPLITTING TRAIN / VALIDATION")
    print("=" * 60)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    print(f"Model training samples: {len(X_fit)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Validation churn rate: {y_val.mean():.2%}")

    return X_fit, X_val, y_fit, y_val


def predict_with_threshold(model, X: pd.DataFrame, threshold: float) -> tuple:
    """Generate probabilities and class predictions with a custom threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_proba, y_pred


def build_comparison_thresholds(operating_threshold: float) -> list:
    """Build a compact comparison grid that always includes the active threshold."""
    default_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    thresholds = default_thresholds + [round(float(operating_threshold), 2)]
    return sorted(set(thresholds))


def train_models(X_train, X_val, y_train, y_val, args, dirs: dict) -> tuple:
    """Train and select the best model on validation data."""
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING MODELS")
    print("=" * 60)

    trainer = ChurnModelTrainer(
        random_state=args.random_state,
        cv_folds=5,
        optimize_threshold=True,
    )

    trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        tune_hyperparams=args.tune,
    )

    # Get results summary
    summary = trainer.get_results_summary()
    print("\nModel Comparison:")
    print(summary[["model", "cv_mean", "val_roc_auc", "val_f1", "optimal_threshold"]])

    # Save best model
    best_model, best_name = trainer.get_best_model()
    best_threshold = trainer.results_[best_name].optimal_threshold
    model_path = os.path.join(dirs["models"], f"best_model_{best_name}.joblib")
    trainer.save_model(best_name, model_path)
    print(f"\nBest model: {best_name}")
    print(f"Operating threshold: {best_threshold:.2f}")
    print(f"Model saved to: {model_path}")

    return trainer, best_model, best_name, best_threshold


def evaluate_model(model, X_test, y_test, threshold: float, dirs: dict) -> tuple:
    """Evaluate the best model."""
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATING MODEL")
    print("=" * 60)

    evaluator = ChurnModelEvaluator(threshold=threshold)
    result = evaluator.evaluate(model, X_test, y_test)

    # Generate report
    report = evaluator.generate_report(result, y_test.values)
    print(report)

    # Save report
    report_path = os.path.join(dirs["reports"], "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    evaluator.plot_confusion_matrix(result, ax=axes[0, 0])
    evaluator.plot_roc_curve(result, y_test.values, ax=axes[0, 1])
    evaluator.plot_precision_recall_curve(result, y_test.values, ax=axes[1, 0])
    evaluator.plot_probability_distribution(result, y_test.values, ax=axes[1, 1])

    plt.tight_layout()
    fig_path = os.path.join(dirs["figures"], "evaluation_plots.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nEvaluation plots saved to: {fig_path}")

    return result


def analyze_with_shap(model, X_train, X_test, dirs: dict, skip_shap: bool = False):
    """Perform SHAP analysis."""
    if skip_shap:
        print("\nSkipping SHAP analysis (--skip-shap)")
        return None

    print("\n" + "=" * 60)
    print("STEP 6: SHAP ANALYSIS")
    print("=" * 60)

    try:
        analyzer = SHAPAnalyzer(model, X_train.head(100))
        importance = analyzer.get_feature_importance(X_test)

        print("\nTop 10 Feature Importance (SHAP):")
        print(importance.head(10).to_string(index=False))

        # Generate SHAP plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot
        top_imp = importance.head(15)
        axes[0].barh(top_imp["feature"][::-1], top_imp["importance"][::-1])
        axes[0].set_xlabel("Mean |SHAP Value|")
        axes[0].set_title("Feature Importance (SHAP)")

        # Summary plot
        try:
            import shap
            shap_values = analyzer.compute_shap_values(X_test.head(200))
            shap.summary_plot(
                shap_values, X_test.head(200),
                max_display=15, show=False, ax=axes[1]
            )
            axes[1].set_title("SHAP Summary Plot")
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Summary plot unavailable:\n{e}",
                        ha="center", va="center", transform=axes[1].transAxes)

        plt.tight_layout()
        fig_path = os.path.join(dirs["figures"], "shap_analysis.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP plots saved to: {fig_path}")

        # Save importance to CSV
        importance_path = os.path.join(dirs["reports"], "feature_importance.csv")
        importance.to_csv(importance_path, index=False)

        return analyzer

    except ImportError:
        print("SHAP not installed. Skipping SHAP analysis.")
        print("Install with: pip install shap")
        return None


def analyze_costs(y_test, y_pred, y_proba, operating_threshold: float, args, dirs: dict):
    """Analyze cost of errors."""
    print("\n" + "=" * 60)
    print("STEP 7: COST ANALYSIS")
    print("=" * 60)
    print(f"Using operating threshold: {operating_threshold:.2f}")

    analyzer = CostAnalyzer(
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        value_tp=args.cost_fn - args.cost_fp,
    )

    result = analyzer.analyze(y_test.values, y_pred, y_proba)
    report = analyzer.generate_report(result)

    print(report)

    # Save report
    report_path = os.path.join(dirs["reports"], "cost_analysis.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Generate cost plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    analyzer.plot_cost_vs_threshold(y_test.values, y_proba, ax=axes[0])
    analyzer.plot_targeting_analysis(y_proba, ax=axes[1])

    plt.tight_layout()
    fig_path = os.path.join(dirs["figures"], "cost_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCost analysis plots saved to: {fig_path}")

    # Print optimal threshold
    optimal = analyzer.find_optimal_threshold(y_test.values, y_proba)
    print(f"\nOptimal threshold (cost-based): {optimal:.2f}")

    return result


def simulate_retention(y_proba, targeting_threshold: float, dirs: dict):
    """Simulate retention campaigns."""
    print("\n" + "=" * 60)
    print("STEP 8: RETENTION SIMULATION")
    print("=" * 60)
    print(f"Using targeting threshold: {targeting_threshold:.2f}")

    # Generate customer values
    np.random.seed(42)
    customer_values = np.random.lognormal(6, 0.5, len(y_proba))

    simulator = RetentionSimulator(
        offer_cost=50,
        success_rate=0.30,
    )

    # Run simulation
    result = simulator.simulate_campaign(
        y_proba, customer_values,
        strategy="target_high_risk",
        threshold=targeting_threshold,
    )

    report = simulator.generate_report(result)
    print(report)

    # Save report
    report_path = os.path.join(dirs["reports"], "retention_simulation.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Compare strategies
    print("\n" + "-" * 40)
    print("STRATEGY COMPARISON")
    print("-" * 40)

    comparison = simulator.compare_strategies(
        y_proba,
        customer_values,
        thresholds=build_comparison_thresholds(targeting_threshold),
    )
    print(comparison[["strategy", "threshold", "net_benefit", "roi"]].head())

    # Save comparison
    comparison_path = os.path.join(dirs["reports"], "strategy_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    # Generate plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    simulator.plot_strategy_comparison(comparison, ax=axes[0])

    # Time simulation
    time_results = simulator.simulate_time_horizon(
        y_proba,
        customer_values,
        n_months=12,
        threshold=targeting_threshold,
    )
    simulator.plot_time_simulation(time_results, ax=axes[1])

    plt.tight_layout()
    fig_path = os.path.join(dirs["figures"], "retention_simulation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSimulation plots saved to: {fig_path}")

    return result


def generate_final_report(
    args,
    dirs: dict,
    best_model_name: str,
    operating_threshold: float,
):
    """Generate final summary report."""
    print("\n" + "=" * 60)
    print("GENERATING FINAL REPORT")
    print("=" * 60)

    report_lines = [
        "=" * 60,
        "CHURN PREDICTION PIPELINE - FINAL REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "CONFIGURATION",
        "-" * 40,
        f"Samples: {args.n_samples}",
        f"Random State: {args.random_state}",
        f"Cost FN: ${args.cost_fn}",
        f"Cost FP: ${args.cost_fp}",
        f"Validation Size: {args.validation_size}",
        f"Hyperparameter Tuning: {args.tune}",
        f"Best Model: {best_model_name}",
        f"Operating Threshold: {operating_threshold:.2f}",
        "",
        "OUTPUT FILES",
        "-" * 40,
    ]

    # List all output files
    for subdir in ["reports", "figures", "models", "data"]:
        path = os.path.join(dirs["base"], subdir)
        if os.path.exists(path):
            files = os.listdir(path)
            for f in files:
                report_lines.append(f"  {subdir}/{f}")

    report_lines.extend([
        "",
        "NEXT STEPS",
        "-" * 40,
        f"1. Review evaluation reports in {dirs['reports']}/",
        f"2. Examine visualizations in {dirs['figures']}/",
        "3. Use saved model for predictions on new data",
        "4. Adjust cost parameters based on business context",
        "5. Implement retention campaigns based on simulation results",
        "",
        "=" * 60,
    ])

    report = "\n".join(report_lines)

    # Save final report
    report_path = os.path.join(dirs["reports"], "FINAL_REPORT.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nFinal report saved to: {report_path}")


def main():
    """Run the complete churn prediction pipeline."""
    args = parse_args()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("CHURN PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Samples: {args.n_samples}")
    print(f"  - Output: {args.output_dir}")
    print(f"  - Tuning: {args.tune}")
    print(f"  - SHAP: {not args.skip_shap}")

    # Setup directories
    dirs = setup_output_directories(args.output_dir)

    # Run pipeline
    df = generate_data(args, dirs)
    X_train, X_test, y_train, y_test = preprocess_data(df, args)
    X_fit, X_val, y_fit, y_val = split_training_data(
        X_train,
        y_train,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    trainer, best_model, best_name, operating_threshold = train_models(
        X_fit,
        X_val,
        y_fit,
        y_val,
        args,
        dirs,
    )
    evaluate_model(
        best_model,
        X_test,
        y_test,
        operating_threshold,
        dirs,
    )
    analyze_with_shap(best_model, X_fit, X_test, dirs, args.skip_shap)

    # Get predictions for downstream analysis using the chosen operating threshold.
    y_proba, y_pred = predict_with_threshold(best_model, X_test, operating_threshold)

    cost_result = analyze_costs(
        y_test,
        y_pred,
        y_proba,
        operating_threshold,
        args,
        dirs,
    )
    simulate_retention(
        y_proba,
        operating_threshold,
        dirs,
    )

    # Generate final report
    generate_final_report(
        args,
        dirs,
        best_name,
        operating_threshold,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
