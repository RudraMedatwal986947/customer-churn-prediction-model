# Machine Learning Models - Evaluation Scores

This document tracks the performance metrics for the models trained in this project.

## 1. Churn Prediction Model
**Type**: Classification (XGBoost)
**Target Variable**: `churn` (Yes/No)

### Metrics:
- **Accuracy**: `80.77%`
- **ROC AUC Score**: `0.8540`

### Classification Report:
| Class (Churn) | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (No)** | 0.85 | 0.89 | 0.87 | 1035 |
| **1 (Yes)** | 0.66 | 0.57 | 0.61 | 374 |
| **Accuracy** | | | **0.81** | 1409 |
| **Macro Avg** | 0.76 | 0.73 | 0.74 | 1409 |
| **Weighted Avg** | 0.80 | 0.81 | 0.80 | 1409 |

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
