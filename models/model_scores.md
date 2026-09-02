# Machine Learning Models - Evaluation Scores

This document tracks the performance metrics for the models trained in this project.

## 1. Churn Prediction Model
**Type**: Classification (XGBoost + LightGBM + CatBoost Stacked Ensemble)
**Target Variable**: `churn` (Yes/No)

### Metrics:
- **Accuracy**: `81.90%`
- **ROC AUC Score**: `0.8619`
- **Optimal Threshold**: `0.460`

### Classification Report:
| Class (Churn) | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (No)** | 0.86 | 0.90 | 0.88 | 1035 |
| **1 (Yes)** | 0.68 | 0.60 | 0.64 | 374 |
| **Accuracy** | | | **0.82** | 1409 |
| **Macro Avg** | 0.77 | 0.75 | 0.76 | 1409 |
| **Weighted Avg** | 0.81 | 0.82 | 0.81 | 1409 |

---

## 2. Customer Lifetime Value (CLV) Prediction Model
**Type**: Regression (XGBoost)
**Target Variable**: `total_charges`

### Metrics:
- **Mean Squared Error (MSE)**: `7111.13`
- **Mean Absolute Error (MAE)**: `$57.91`
- **R-squared (R²)**: `0.9986` 

*(Note: The extremely high R² score is due to the strong intrinsic correlation between `tenure` × `monthly_charges` and `total_charges` in this snapshot dataset.)*

---

## 3. Customer Segmentation Model
**Type**: Unsupervised Clustering (K-Means)
**Features Used**: `tenure`, `monthly_charges`, `total_charges`, `total_additional_services`

### Metrics:
- **Number of Clusters (k)**: `4`
- **Assignment**: Successfully labeled 7,043 customers into 4 distinct behavioral segments.
