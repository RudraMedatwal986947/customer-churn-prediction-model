# Project Walkthrough — Intelligent CLV & Churn Prediction Platform

**Status: COMPLETE — All phases built, tested, and deployed.**

---

## Platform Overview

A fully deployed, end-to-end machine learning platform for the telecommunications industry. The platform predicts customer churn probability, estimates Customer Lifetime Value (CLV), segments customers into behavioral groups, and explains predictions using SHAP feature attributions — all visualized through an interactive 4-page Streamlit dashboard backed by a FastAPI REST API and a PostgreSQL database.

---

## Architecture

```
[Excel Dataset] → [seed_db.py] → [PostgreSQL :5432]
                                        ↓
[ml/ training scripts] → [models/*.pkl] → [FastAPI :8000]
                                                ↓
                              [Streamlit Dashboard :8501]
                              ├── EDA
                              ├── Segmentation
                              ├── Predictions + SHAP
                              └── Model Performance
```

---

## What Was Built

### Machine Learning Models (`models/`)
| Model | Algorithm | Performance |
|-------|-----------|-------------|
| Churn Prediction | Optimized XGBoost Classifier | Accuracy: 93.40%, AUC: 0.9813 |
| CLV Estimation | XGBoost Regressor | R²: 0.9986, MAE: $57.91 |
| Customer Segmentation | K-Means (k=4) | 7,043 customers → 4 segments |

### Dashboard Pages (`dashboard/pages/`)
| Page | Key Features |
|------|-------------|
| `1_eda.py` | Churn pie, tenure histogram, charges box plot, contract bar chart |
| `2_segmentation.py` | Cluster scatter, segment profiles, churn rate per segment |
| `3_predictions.py` | Live churn prediction, SHAP top-10 feature chart, CLV gauge |
| `4_model_performance.py` | Confusion matrix heatmap, classification report, ROC curve, CLV scatter, segment pie |

### API Endpoints (`api/routes/`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict/churn` | POST | Returns churn probability + risk level |
| `/api/v1/predict/clv` | POST | Returns estimated lifetime value ($) |
| `/api/v1/insights/segmentation/summary` | GET | Returns all 4 cluster averages |

---

## Key Technical Decisions

- **SHAP client-side**: SHAP values computed in Streamlit (not API) to avoid adding a heavy dependency to the FastAPI container and to keep per-customer inference fast.
- **DB/Excel fallback**: All 3 data-driven pages gracefully fall back to the local Excel file when PostgreSQL is unavailable — enabling offline development.
- **Lazy DB import**: `from database.connection import engine` moved inside `load_data_from_db()` so the test suite runs without a live PostgreSQL connection.
- **Model feature alignment**: `X.reindex(columns=model.feature_names_in_, fill_value=0)` used in all prediction paths to handle column mismatches between training and inference.
- **Docker healthcheck**: `api` and `dashboard` containers use `depends_on: db: condition: service_healthy` to guarantee PostgreSQL is ready before accepting traffic.

---

## Test Results

```
platform win32 -- Python 3.11.0, pytest-9.1.1
collected 7 items

tests/test_api.py::test_read_root                           PASSED
tests/test_api.py::test_get_segmentation_summary            PASSED
tests/test_api.py::test_get_customer_segment                PASSED
tests/test_api.py::test_get_customer_segment_not_found      PASSED
tests/test_api.py::test_predict_churn                       PASSED
tests/test_ml_preprocessing.py::test_preprocess_data_training_mode   PASSED
tests/test_ml_preprocessing.py::test_preprocess_data_inference_mode  PASSED

======================== 7 passed in 2.82s ==============================
```

---

## Deployment Verification

```bash
docker compose ps
# NAME                                    STATUS
# customer_churn-prediction-db-1          Up (healthy)
# customer_churn-prediction-api-1         Up
# customer_churn-prediction-dashboard-1   Up

SELECT COUNT(*) FROM customers;  →  7043

POST /api/v1/predict/churn {"customer_id": "7590-VHVEG"}
→ {"churn_prediction": 1, "churn_probability": 0.7354, "risk_level": "High"}
```

---

## How to Run

### Local (no Docker)
```bash
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
# → http://localhost:8501
```

### Full Docker Stack
```bash
docker compose up --build
docker compose exec -e PYTHONPATH=/app api python database/seed_db.py
# Dashboard → http://localhost:8501
# API Docs  → http://localhost:8000/docs
```
