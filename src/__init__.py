"""Churn Prediction Package.

A comprehensive customer churn prediction system with:
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- SHAP-based explainability
- Cost of error analysis
- Retention simulation
"""

import matplotlib

# Use a headless backend so plots work in tests, CI, and terminal-only runs.
matplotlib.use("Agg")

__version__ = "1.0.0"
__author__ = "Portfolio Project"
