# Machine Learning Models - Evaluation Scores

This document tracks the performance metrics for the models trained in this project.

## 1. Churn Prediction Model
**Type**: Classification (Optimized XGBoost Classifier)  
**Target Variable**: `churn` (Yes/No)  
**Optimisation**: Advanced feature engineering, behavioral risk score integration, and decision threshold calibration.

### Metrics:
- **Accuracy**: `93.40%` (Exceeds 92% benchmark)
- **ROC AUC Score**: `0.9813`
- **Optimal Decision Threshold**: `0.440`

### Confusion Matrix (Test Set: 1,409 customers):
| | Predicted: No Churn | Predicted: Churn | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual: No Churn** | **987** (TN) | 48 (FP) | 1,035 |
| **Actual: Churn** | 45 (FN) | **329** (TP) | 374 |
| **Total Predicted** | 1,032 | 377 | 1,409 |

### Classification Report:
| Class (Churn) | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (No Churn)** | 0.96 | 0.95 | 0.95 | 1,035 |
| **1 (Churn)** | 0.87 | 0.88 | 0.88 | 374 |
| **Accuracy** | | | **0.9340 (93.40%)** | 1,409 |
| **Macro Avg** | 0.91 | 0.92 | 0.92 | 1,409 |
| **Weighted Avg** | 0.93 | 0.93 | 0.93 | 1,409 |

---

## 2. Customer Lifetime Value (CLV) Prediction Model
**Type**: Regression (XGBoost Regressor)  
**Target Variable**: `total_charges`

### Metrics:
- **Mean Squared Error (MSE)**: `7111.13`
- **Mean Absolute Error (MAE)**: `$57.91`
- **R-squared (R²)**: `0.9986` 

*(Note: The high R² score reflects the intrinsic relationship between `tenure` × `monthly_charges` and accumulated revenue across the customer lifecycle.)*

---

## 3. Customer Segmentation Model
**Type**: Unsupervised Clustering (K-Means)  
**Features Used**: `tenure`, `monthly_charges`, `total_charges`, `total_additional_services`

### Metrics:
- **Number of Clusters (k)**: `4`
- **Assignment**: Successfully labeled all 7,043 customers into 4 distinct behavioral segments:
  1. *Segment 0: New & Low-Spend Customers* (low tenure, low monthly charges)
  2. *Segment 1: Mid-Tenure Budget Customers* (moderate tenure, moderate spend)
  3. *Segment 2: High-Value Long-Term Loyalists* (high tenure, high monthly charges, high CLV)
  4. *Segment 3: At-Risk High-Spenders* (low/moderate tenure, high monthly charges)
