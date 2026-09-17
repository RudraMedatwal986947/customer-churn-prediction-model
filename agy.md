# Customer Churn & CLV Prediction Platform — Project Memory

## Project Goals
An End-to-End Machine Learning Platform for Customer Segmentation, Customer Lifetime Value Prediction, Churn Prediction, and Business Intelligence.

## Tech Stack
- **Database**: PostgreSQL 15 (Dockerized)
- **Backend API**: FastAPI + Uvicorn
- **Frontend**: Streamlit + Plotly
- **Machine Learning**: Scikit-Learn, XGBoost, SHAP
- **Deployment**: Docker, Docker Compose
- **Testing**: Pytest

## Accomplished Tasks

### Phase 1 — Infrastructure
- [x] Initialized project directory structure (`data/`, `ml/`, `api/`, `dashboard/`, `database/`).
- [x] Created `requirements.txt` (includes `shap`, `joblib`) and `.gitignore`.
- [x] Set up `docker-compose.yml` (PostgreSQL, FastAPI, Streamlit) with healthchecks and volume mounts.
- [x] Created `.dockerignore` to optimize build context.
- [x] Configured database ORM models in `database/models.py`.
- [x] Implemented `database/seed_db.py` to populate PostgreSQL with 7,043 records.

### Phase 2 — Machine Learning Pipeline
- [x] Robust data preprocessing (`ml/data_preprocessing.py`): missing value handling, feature engineering (`tenure_group`, `avg_monthly_charge`, `total_additional_services`), OHE, StandardScaler.
- [x] **Churn Prediction**: XGBoost Classifier — Accuracy 80.77%, ROC AUC 0.854.
- [x] **CLV Prediction**: XGBoost Regressor — R² 0.9986, MAE $57.91.
- [x] **Segmentation**: K-Means Clustering (k=4) — 7,043 customers in 4 segments.
- [x] All models serialized to `models/*.pkl`.
- [x] Visualizations generated (`roc_curve.png`, `feature_importance.png`, `confusion_matrix.png`, `segmentation_scatter.png`).

### Phase 3 — Backend API
- [x] FastAPI routes: `/api/v1/predict/churn`, `/api/v1/predict/clv`, `/api/v1/insights/segmentation/summary`.
- [x] Models loaded from `.pkl` at request time (lazy loading pattern).
- [x] SQLAlchemy 2.0 compatible (`declarative_base` from `sqlalchemy.orm`).

### Phase 4 — Frontend Dashboard (4 pages)
- [x] `1_eda.py` — Churn distribution, tenure histogram, monthly charges box plot, contract type bar chart.
- [x] `2_segmentation.py` — K-Means cluster scatter, box plots, churn rate per segment.
- [x] `3_predictions.py` — Live churn prediction + **SHAP explainability bar chart** + CLV gauge chart.
- [x] `4_model_performance.py` — Confusion matrix heatmap, classification report, ROC curve, CLV scatter, segment profiles.

### Phase 5 — Testing & Quality
- [x] **7/7 pytest tests passing, 0 warnings**.
- [x] `test_api.py`: root, segmentation summary, customer segment, 404 handling, churn prediction.
- [x] `test_ml_preprocessing.py`: training mode pipeline, inference mode pipeline.
- [x] Lazy DB import fix in `ml/data_preprocessing.py` (tests run without live PostgreSQL).

### Phase 6 — Deployment
- [x] `docker-compose up --build` — all 3 containers built and started.
- [x] Database seeded: **7,043 customers** in `churn_db.customers`.
- [x] **End-to-end verified**: Customer `7590-VHVEG` → Churn 73.5% (High Risk) via live API.
- [x] All 3 API endpoints returning `200 OK` from live PostgreSQL data.

## Current Status — PROJECT COMPLETE - Docker stack (`db`, `api`, `dashboard`) fully running.
- Dashboard at **http://localhost:8501** (4 pages, PostgreSQL-connected).
- API Swagger at **http://localhost:8000/docs** (3 endpoints live).
- Database: 7,043 customer records.

## Future Scope (Post-Submission)
- SHAP summary plots across the full dataset (not just per-customer).
- MLOps: automated retraining pipeline triggered by data drift.
- Real-time streaming via Kafka for live customer event ingestion.
- Deep learning models (e.g., LSTM) for time-series CLV.
- CRM integration (Salesforce / HubSpot API).
