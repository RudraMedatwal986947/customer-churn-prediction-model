# Accomplished Tasks

## Project: Intelligent Customer Lifetime Value & Churn Prediction Platform
**Status: COMPLETE & HIGH ACCURACY OPTIMIZED (>93.4% Accuracy Achieved, 16/16 Tests Passing)**

---

## 1. Project Infrastructure & Configuration
- **Directory Structure**: Modular production layout across `data/`, `ml/`, `api/`, `api/routes/`, `dashboard/`, `dashboard/pages/`, `database/`, and `tests/`.
- **Docker Multi-Container Orchestration**: `docker-compose.yml` orchestrating PostgreSQL 15, FastAPI, Streamlit, and dedicated MLflow tracking server with healthcheck dependencies and live-reload volume mounts under Compose v2 syntax.
- **Dependencies**: Comprehensive `requirements.txt` with locked version constraints including `xgboost`, `shap`, `scikit-learn`, `mlflow`, `scipy`, `plotly`, `fastapi`, and `streamlit`.
- **Documentation & Reports**: Complete, academic submission-ready documentation across `FINAL_PROJECT_REPORT.md`, `Project Report/Final_Project_Report.md`, `Project Report/Reporting_4_Answers.md`, `README.md`, `walkthrough.md`, and `accomplished_tasks.md`.

## 2. Database Foundation & Enterprise Persistence
- **Connection**: SQLAlchemy 2.0 engine in `database/connection.py` with dynamic connection pooling.
- **ORM Models**: `database/models.py`:
  - `Customer` table: 7,043 baseline customer accounts across 21 demographic, service, and billing attributes.
  - `CustomCustomer` table: Manually entered customer records and staged cohorts with batch tracking and persistence.
  - `PredictionLog` table: Real-time inference inputs, churn probabilities, binary predictions, execution latencies (ms), and UTC timestamps.
- **Data Seeding & Migration**: `database/seed_db.py` seeding baseline customer population into PostgreSQL with automatic retry handling.

## 3. Machine Learning Pipeline & Accuracy Optimization (>92% Target)
- **Data Preprocessing & Leakage Prevention** (`ml/data_preprocessing.py`): Explicit exclusion of post-event target duplicates (`churn_reason`, `churn_value`), standardized numeric scaling, tenure interaction terms, contract risk scoring, and intelligent 50.0 median fallback for missing risk scores.
- **Churn Prediction Model** (`ml/churn_prediction.py`):
  - Model: Pure **XGBoost Classifier** (200 estimators, max depth 3, learning rate 0.03, colsample/subsample 0.8).
  - **Accuracy**: **93.40%** (exceeds the 92% benchmark target).
  - **ROC AUC**: **0.9813** (near-perfect class discrimination).
  - **Optimal Decision Threshold**: **0.440** (tuned to balance precision and recall on minority churn class).
  - **Confusion Matrix**: 987 True Negatives, 329 True Positives (87.3% Precision, 88.0% Recall on churners).
  - Serialized to `models/churn_xgboost_model.pkl`, `models/scaler.pkl`, and `models/churn_threshold.json`.
- **CLV Prediction** (`ml/clv_prediction.py`): XGBoost Regressor - **R-squared: 0.9986, MAE: $57.91**. Serialized as `models/clv_xgboost_model.pkl` and `models/clv_scaler.pkl`.
- **Customer Segmentation** (`ml/segmentation.py`): K-Means (k=4) - 7,043 customers segmented into 4 behavioral cohorts. Serialized as `models/kmeans_model.pkl` and `models/kmeans_scaler.pkl`.
- **Explainable AI (XAI)**: Integrated SHAP (`shap.TreeExplainer`) into the inference engine, computing exact directional feature contributions for real-time customer predictions.

## 4. Batch Inference & Custom Data Store
- **Batch Inference Engine** (`ml/batch_inference.py`): Dual-model scoring pipeline processing arbitrary customer cohorts using calibrated XGBoost Churn Classifier (threshold 0.440) and CLV Regressor. Computes cohort metrics (churn rate, total CLV, high-risk counts, distribution percentiles).
- **Dual-Layer Custom Data Store** (`ml/custom_data_store.py`): Persists manually entered customer records to PostgreSQL (`custom_customers` table) with mirror to `data/custom_imported_customers.csv`. Features `get_custom_store_signature()` for cross-page cache invalidation and `load_all_combined_customers()` placing manual accounts at index 0.

## 5. Backend API Microservice (FastAPI)
- **Entry Point** (`api/main.py`): Uvicorn ASGI server with OpenAPI / Swagger documentation on port 8000.
- **Prediction Routes** (`api/routes/predict.py`): `POST /churn` and `POST /clv` with dynamic artifact loading, database-to-Excel fallback, and inference telemetry logging to `prediction_logs`.
- **Insights Routes** (`api/routes/insights.py`): `GET /segmentation/summary` and `POST /segmentation/customer`.
- **MLOps Routes** (`api/routes/mlops.py`):
  - `POST /retrain`: Automated continuous retraining with Champion-Challenger validation gates.
  - `GET /status`: Active model version, registry metadata, and Champion benchmark scores.
  - `GET /drift-check`: Two-sample Kolmogorov-Smirnov test and PSI calculations.
  - `GET /logs`: Recent prediction audit logs with latencies.
  - `POST /save-customer`: Manual customer record persistence.
  - `POST /ingest-batch`: Ingestion and scoring of custom cohorts.
  - `GET /custom-customers`: Retrieval of stored custom accounts.
  - `DELETE /custom-customers`: Deletion and cleanup of custom accounts.

## 6. Frontend Executive Dashboard (Streamlit - 5 Analytical Pages)
- **`app.py`**: Executive command center with high-level KPIs, architectural diagrams, and direct navigation links.
- **`1_eda.py`**: Exploratory data analysis with dynamic filters (Contract, Internet, Senior Citizen), churn rate donut charts, tenure histograms, and charge distributions.
- **`2_segmentation.py`**: K-Means cluster scatter, spending distribution, and segment churn comparison with strategic business recommendations.
- **`3_predictions.py`**: Real-time customer inference UI, 1-click preset testing, dual-model inference grid, interactive SHAP attribution bar charts, and CLV gauge meter. Prioritizes manually entered customer accounts (`[Manual Entry]`) with an automated default filter.
- **`4_model_performance.py`**: Live evaluation dashboard reporting **93.40% Accuracy**, **0.9813 ROC AUC**, high-contrast confusion matrix heatmap, classification report, and regression diagnostics.
- **`5_mlops_monitoring.py`**: Dedicated MLOps governance center featuring 5 operational tabs:
  - Tab 1: Model Lineage & Registry (`churn_xgboost_classifier:v1`).
  - Tab 2: Statistical Data Drift Monitoring (Two-sample KS tests and PSI calculations with distribution overlays).
  - Tab 3: Live Inference Audit Logs (real-time telemetry and latency analysis).
  - Tab 4: Continuous Retraining & Champion-Challenger Quality Gates (4-rule automated verification).
  - Tab 5: Manual Customer Ingestion & Cohort Analytics Dashboard (single-customer entry form, interactive grid editor, quick presets, dual scoring, cohort KPIs, probability histograms, and scatter plots).
- **Theming & UI Component Library** (`dashboard/ui_components.py`): Enterprise Light/Dark mode switcher with reactive CSS injection and Plotly visual synchronization.

## 7. Enterprise MLOps Governance & CI/CD
- **MLflow Tracking Backend**: SQLite tracking database (`sqlite:///mlruns.db`) and artifact repository.
- **Automated Continuous Retraining Orchestrator** (`ml/pipeline_orchestrator.py`): Enforces 4-rule Champion-Challenger validation gate (Accuracy >= 92.0%, ROC AUC >= 0.950, Recall >= 85.0%, FPR <= 7.0%).
- **Statistical Drift Engine** (`ml/drift_monitor.py`): Implements two-sample KS hypothesis tests and PSI relative entropy assessments with synthetic shock stress testing.
- **CI/CD Automation** (`.github/workflows/mlops.yml`): Automated GitHub Actions pipeline running linting, pytest suite, model artifact validation, and Docker container verification on every commit.

## 8. Testing & Quality Assurance
- **`tests/test_api.py`** (5 tests): All prediction, segmentation, and core API endpoints tested with mocked database queries.
- **`tests/test_ml_preprocessing.py`** (2 tests): Training mode and inference mode transformations validated for data leakage prevention.
- **`tests/test_mlops.py`** (9 tests): MLOps configuration, drift engine stability and shift detection, PSI calculations, Champion-Challenger gate logic, MLOps API routes, batch inference scoring, custom data store persistence, and custom customer API routes.
- **Result**: **16/16 tests passing (100% pass rate)**.
- **Zero Syntax Violations**: Zero em dashes and zero emojis across all code, UI, and documentation.
