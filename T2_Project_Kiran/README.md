# Credit Risk Assessment ML Pipeline

## Overview
This repository contains an end-to-end credit risk classification pipeline for identifying applicants likely to default (`bad` risk). The solution includes:
- EDA and preprocessing
- feature engineering
- training of 4 required models
- hyperparameter tuning
- MLflow experiment tracking
- model evaluation with business-focused interpretation

## Business Objective
FinTech Solutions Inc. needs near real-time risk scoring with strong default detection.  
Primary objective: maximize recall on default class while maintaining practical precision and interpretability.

## Success Criteria
- Build reproducible pipeline with `random_state=42`
- Train Logistic Regression, Decision Tree, Random Forest, XGBoost
- Use CV-based tuning for at least 2 models
- Track experiments with MLflow
- Achieve and report strong recall on class `bad`

## Project Structure
```text
credit-risk/
|-- data/
|   |-- raw/
|   |   `-- german_credit_data.csv
|   `-- processed/
|       |-- X_train.csv
|       |-- X_test.csv
|       |-- y_train.csv
|       `-- y_test.csv
|-- notebooks/
|   |-- 01_EDA_Preprocessing.ipynb
|   |-- 02_Model_Development.ipynb
|   `-- 03_Model_Evaluation.ipynb
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- train.py
|   `-- evaluate.py
|-- models/
|-- reports/
|   |-- figures/
|   |-- model_comparison_metrics.csv
|   |-- model_evaluation_comparison.csv
|   `-- production_threshold_recommendation.csv
|-- mlruns/
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
Expected dataset path:
`data/raw/german_credit_data.csv`

If the dataset is missing, `notebooks/01_EDA_Preprocessing.ipynb` includes a fallback synthetic data generator.

## Execution Order
1. `notebooks/01_EDA_Preprocessing.ipynb`
2. `notebooks/02_Model_Development.ipynb`
3. `notebooks/03_Model_Evaluation.ipynb`

## Models Implemented
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- XGBoost

## Feature Engineering
Implemented engineered features:
- `Credit_to_Duration_Ratio`
- `Age_Group`
- `Account_Stability`
- `High_Risk_Purpose`

## Evaluation Summary
From `reports/model_evaluation_comparison.csv`:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.815 | 0.705 | 0.797 | 0.748 | 0.904 |
| xgboost | 0.890 | 0.912 | 0.754 | 0.825 | 0.947 |
| random_forest | 0.840 | 0.894 | 0.609 | 0.724 | 0.911 |
| decision_tree | 0.780 | 0.719 | 0.594 | 0.651 | 0.736 |

Threshold tuning output (`reports/production_threshold_recommendation.csv`):
- selected model: `logistic_regression`
- threshold for recall target: `0.5833`
- precision at threshold: `0.7465`
- recall at threshold: `0.7681`

## Production Recommendation
- Prioritize default recall due to higher cost of false negatives.
- Use threshold-tuned model meeting recall target (`>= 0.75`).
- Current selected candidate: threshold-tuned Logistic Regression.
- XGBoost remains a strong challenger with higher AUC/F1 but slightly lower recall in current test results.

## MLflow Tracking
MLflow runs are logged locally under `mlruns/` with:
- parameters
- metrics
- confusion matrix artifacts
- serialized model artifacts

## Artifacts Generated
- Processed datasets: `data/processed/`
- Trained models: `models/*.pkl`
- Evaluation plots: `reports/figures/`
- Comparison files: `reports/*.csv`

## Reproducibility Notes
- Use fixed seed: `42`
- Keep notebook execution in order
- Do not fit preprocessing on test data

## Contributors
- Venkata Kiran Kumar
