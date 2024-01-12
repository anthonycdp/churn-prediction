# Customer Churn Prediction

A comprehensive machine learning system for predicting customer churn with model explainability (SHAP), cost-based error analysis, and retention campaign simulation.

## Overview

This project demonstrates a production-ready approach to customer churn prediction, going beyond simple model training to include:

- **Multi-model comparison** with hyperparameter tuning
- **SHAP-based explainability** for understanding model decisions
- **Cost of error analysis** for business-aligned threshold optimization
- **Retention campaign simulation** for strategic planning
- **Comprehensive evaluation** with multiple metrics and visualizations

## Features

### Data Generation & Preprocessing
- Synthetic data generator with realistic churn patterns
- Feature engineering (derived features, engagement scores)
- Automatic handling of numerical and categorical features
- Train/test splitting with stratification

### Model Training
- Multiple algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Cross-validation for robust evaluation
- Automatic threshold optimization
- Feature importance extraction

### Model Explainability (SHAP)
- Global feature importance visualization
- Individual prediction explanations
- SHAP summary plots (beeswarm)
- Dependence plots for feature interactions
- High-risk customer identification

### Cost Analysis
- Configurable costs for false positives/negatives
- Optimal threshold selection based on business costs
- Budget-constrained targeting optimization
- ROI calculation

### Retention Simulation
- Multiple targeting strategies
- Monte Carlo simulation for uncertainty
- Time-horizon forecasting
- Sensitivity analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run the Complete Pipeline

```bash
python main.py --n-samples 5000 --output-dir outputs
```

The CLI keeps a final test split fully held out and creates an additional validation split from the training data for model selection and threshold tuning.

### Basic Usage

```python
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.generator import ChurnDataGenerator
from src.data.preprocessor import ChurnDataPreprocessor
from src.models.trainer import ChurnModelTrainer
from src.models.evaluator import ChurnModelEvaluator
from src.explainability.shap_analyzer import SHAPAnalyzer
from src.analysis.cost_analysis import CostAnalyzer
from src.analysis.retention_simulation import RetentionSimulator

# 1. Generate data
generator = ChurnDataGenerator(random_state=42)
df = generator.generate(n_samples=5000)

# 2. Preprocess
preprocessor = ChurnDataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

# 3. Create a validation split from the training data
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# 4. Train models and tune the operating threshold on validation data
trainer = ChurnModelTrainer(random_state=42, optimize_threshold=True)
trainer.train_all_models(X_fit, y_fit, X_val, y_val)
best_model, best_name = trainer.get_best_model()
operating_threshold = trainer.results_[best_name].optimal_threshold

# 5. Evaluate on the held-out test split
evaluator = ChurnModelEvaluator(threshold=operating_threshold)
result = evaluator.evaluate(best_model, X_test, y_test)
print(evaluator.generate_report(result, y_test.values))

# 6. Explain with SHAP
analyzer = SHAPAnalyzer(best_model, X_fit.head(100))
importance = analyzer.get_feature_importance(X_test)
print(importance.head(10))

# 7. Analyze costs using the same operating threshold
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= operating_threshold).astype(int)

cost_analyzer = CostAnalyzer(cost_fn=500, cost_fp=50)
cost_result = cost_analyzer.analyze(y_test.values, y_pred, y_proba)
print(cost_analyzer.generate_report(cost_result))

# 8. Simulate retention using the same targeting threshold
customer_values = np.random.lognormal(6, 0.5, len(y_proba))
simulator = RetentionSimulator(offer_cost=50, success_rate=0.30)
sim_result = simulator.simulate_campaign(
    y_proba,
    customer_values,
    strategy="target_high_risk",
    threshold=operating_threshold,
)
print(simulator.generate_report(sim_result))
```

## Project Structure

```
churn-prediction/
├── README.md
├── requirements.txt
├── main.py                    # Main entry point
├── notebooks/                 # Optional exploratory notebooks
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py       # Synthetic data generation
│   │   └── preprocessor.py    # Feature engineering & preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Model training & comparison
│   │   └── evaluator.py       # Model evaluation & metrics
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_analyzer.py   # SHAP-based explainability
│   └── analysis/
│       ├── __init__.py
│       ├── cost_analysis.py   # Cost of error analysis
│       └── retention_simulation.py  # Campaign simulation
├── tests/
│   ├── __init__.py
│   ├── test_data.py           # Data module tests
│   ├── test_models.py         # Model module tests
│   ├── test_explainability.py # SHAP tests
│   ├── test_analysis.py       # Analysis tests
│   └── test_main.py           # Pipeline orchestration tests
└── outputs/                   # Default generated outputs directory
    ├── data/
    ├── figures/
    ├── models/
    └── reports/
```

## Key Components

### 1. Data Generator (`src/data/generator.py`)

Generates realistic synthetic customer data with configurable churn patterns.

```python
generator = ChurnDataGenerator(random_state=42)
df = generator.generate(
    n_samples=5000,
    base_churn_rate=0.20,
    include_customer_id=True
)
```

Features generated:
- **Numerical**: tenure, monthly charges, total charges, call duration, support tickets, complaints, data usage
- **Categorical**: contract type, payment method, internet service, security, support, streaming

### 2. Data Preprocessor (`src/data/preprocessor.py`)

Handles feature engineering and data transformation.

```python
preprocessor = ChurnDataPreprocessor(
    scaling_method="standard",  # or None
    encode_method="onehot",     # or "label"
    test_size=0.2
)
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
```

Engineered features:
- `avg_monthly_charge`: Average charge over tenure
- `charge_change_ratio`: Current vs average charge ratio
- `engagement_score`: Composite engagement metric
- `tenure_bucket`: Categorized tenure
- `high_risk_flag`: Risk indicator based on complaints/changes
- `service_count`: Number of subscribed services

### 3. Model Trainer (`src/models/trainer.py`)

Trains and compares multiple models.

```python
trainer = ChurnModelTrainer(
    cv_folds=5,
    optimize_threshold=True,
    threshold_metric="f1"
)

# Train specific model
result = trainer.train_model(
    "random_forest",
    X_train, y_train,
    X_val, y_val,
    tune_hyperparams=True
)

# Train all models
trainer.train_all_models(X_train, y_train, X_val, y_val)

# Get best model
best_model, best_name = trainer.get_best_model()
operating_threshold = trainer.results_[best_name].optimal_threshold
```

### 4. Model Evaluator (`src/models/evaluator.py`)

Comprehensive model evaluation with visualizations.

```python
evaluator = ChurnModelEvaluator(threshold=operating_threshold)
result = evaluator.evaluate(model, X_test, y_test)

# Generate report
print(evaluator.generate_report(result, y_test.values))

# Plot evaluation
evaluator.plot_confusion_matrix(result)
evaluator.plot_roc_curve(result, y_test.values)
evaluator.plot_precision_recall_curve(result, y_test.values)
evaluator.plot_threshold_analysis(y_test.values, result.y_proba)
```

### 5. SHAP Analyzer (`src/explainability/shap_analyzer.py`)

Model explainability using SHAP values.

```python
analyzer = SHAPAnalyzer(model, X_train.head(100))

# Global feature importance
importance = analyzer.get_feature_importance(X_test)

# Individual prediction explanation
explanation = analyzer.explain_prediction(X_test.iloc[0])
print(f"Churn probability: {explanation['predicted_probability']:.3f}")
for feature in explanation['top_features'][:5]:
    print(f"  {feature['feature']}: {feature['shap_value']:+.3f}")

# Visualizations
analyzer.plot_summary(X_test)
analyzer.plot_bar(X_test)
analyzer.plot_waterfall(X_test.iloc[0])
```

### 6. Cost Analyzer (`src/analysis/cost_analysis.py`)

Business-cost-aware analysis.

```python
analyzer = CostAnalyzer(
    cost_fn=500,    # Cost of missing a churner
    cost_fp=50,     # Cost of false alarm
    value_tp=450,   # Value of successful retention
    retention_rate=0.30
)

result = analyzer.analyze(y_true, y_pred, y_proba)

# Find cost-optimal threshold
optimal_threshold = analyzer.find_optimal_threshold(y_true, y_proba)

# Budget-constrained targeting
targeting = analyzer.optimize_targeting(y_proba, budget=1000)
```

### 7. Retention Simulator (`src/analysis/retention_simulation.py`)

Simulate retention campaigns.

```python
simulator = RetentionSimulator(
    offer_cost=50,
    success_rate=0.30
)

# Simulate campaign
result = simulator.simulate_campaign(
    y_proba, customer_values,
    strategy="target_high_risk",
    threshold=operating_threshold
)

# Compare strategies
comparison = simulator.compare_strategies(
    y_proba,
    customer_values,
    thresholds=[operating_threshold, 0.5]
)

# Sensitivity analysis
sensitivity = simulator.sensitivity_analysis(y_proba, customer_values)

# Time horizon simulation
time_results = simulator.simulate_time_horizon(
    y_proba,
    customer_values,
    n_months=12,
    threshold=operating_threshold
)
```

## Command Line Interface

```bash
# Basic run
python main.py

# Custom configuration
python main.py \
    --n-samples 10000 \
    --output-dir my_outputs \
    --validation-size 0.25 \
    --random-state 123 \
    --cost-fn 750 \
    --cost-fp 75 \
    --tune \
    --skip-shap
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-samples` | 5000 | Number of samples to generate |
| `--output-dir` | outputs | Output directory |
| `--validation-size` | 0.2 | Validation split taken from the training data for model and threshold selection |
| `--random-state` | 42 | Random seed |
| `--cost-fn` | 500.0 | Cost of false negative |
| `--cost-fp` | 50.0 | Cost of false positive |
| `--tune` | False | Enable hyperparameter tuning |
| `--skip-shap` | False | Skip SHAP analysis |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run the pipeline orchestration tests
pytest tests/test_main.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module tests
pytest tests/test_models.py -v
```

## Output Files

After running the pipeline, the following files are generated inside the configured output directory:

```
<output-dir>/
├── data/
│   └── churn_data.csv           # Generated dataset
├── figures/
│   ├── evaluation_plots.png     # Model evaluation visualizations
│   ├── shap_analysis.png        # SHAP feature importance
│   ├── cost_analysis.png        # Cost analysis plots
│   └── retention_simulation.png # Campaign simulation results
├── models/
│   └── best_model_*.joblib      # Saved best model
└── reports/
    ├── evaluation_report.txt    # Model evaluation report
    ├── feature_importance.csv   # SHAP feature importance
    ├── cost_analysis.txt        # Cost analysis report
    ├── retention_simulation.txt # Retention simulation report
    ├── strategy_comparison.csv  # Strategy comparison results
    └── FINAL_REPORT.txt         # Summary report
```

## Business Insights

### Cost of Errors

The cost analysis helps answer:
- What's the optimal classification threshold given business costs?
- How much budget should be allocated to retention?
- What's the expected ROI of retention campaigns?

### Feature Importance

SHAP analysis reveals which factors drive churn:
- Tenure and contract type typically have high impact
- Customer engagement metrics often predict churn
- Service quality indicators (support tickets, complaints) are key signals

### Retention Strategy

The simulator compares strategies:
- **Target High Risk**: Target customers above the chosen operating threshold
- **Target Top N**: Target the highest-risk customers regardless of threshold
- **Budget Constrained**: Maximize impact within budget
- **Threshold Optimized**: Target customers with positive expected campaign value

## Extending the Project

### Adding New Models

```python
# In src/models/trainer.py, add to model_configs:
"my_model": {
    "class": MyModelClass,
    "params": {...},
    "param_grid": {...},
}
```

### Custom Features

```python
# In src/data/preprocessor.py, modify _engineer_features():
def _engineer_features(self, df):
    df = df.copy()
    # Add custom features
    df['my_feature'] = df['col1'] * df['col2']
    return df
```

### Custom Cost Functions

```python
# Modify CostAnalyzer for custom cost logic
class CustomCostAnalyzer(CostAnalyzer):
    def calculate_cost_matrix(self):
        # Custom cost calculation
        ...
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms and metrics
- **shap**: Model explainability
- **xgboost**: Gradient boosting (optional)
- **matplotlib/seaborn**: Visualization
- **pytest**: Testing

## License

This project is for educational and portfolio purposes.

## Author

Portfolio Project - Customer Churn Prediction System
