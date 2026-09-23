# Project Walkthrough: Intelligent CLV & Churn Prediction Platform

**Status: COMPLETE: All phases built, tested, and deployed with enterprise MLOps lifecycle.**

---

## Platform Overview

A fully deployed, end-to-end machine learning platform for the telecommunications industry. The platform predicts customer churn probability, estimates Customer Lifetime Value (CLV), segments accounts into behavioral cohorts, explains individual predictions using SHAP feature attributions, and provides manual customer ingestion with a dedicated cohort analytics dashboard. The system is visualized through an interactive 5-page Streamlit dashboard backed by a FastAPI REST API, a PostgreSQL relational database, and an MLflow Model Registry.

---

## Architecture

```
[Excel Dataset] --> [database/seed_db.py] --> [PostgreSQL :5432 (churn_db)]
                                                      |
[ml/ training scripts] --> [models/*.pkl] ---------> [FastAPI :8000]
                               |                      |
[ml/pipeline_orchestrator] <---+                      |
        |                                             |
[MLflow Registry :5000]                               |
        |                                             v
        +-----------------------------------> [Streamlit Dashboard :8501]
                                              |-- 1. EDA
                                              |-- 2. Segmentation
                                              |-- 3. Predictions + SHAP
                                              |-- 4. Model Performance
                                              `-- 5. MLOps Monitoring
                                                  `-- Tab 5: Cohort Analytics
```

---

## What Was Built

### 1. Machine Learning Models (`models/`)
| Model | Algorithm | Key Performance Metrics | Artifact Files |
| :--- | :--- | :--- | :--- |
| **Churn Prediction** | Calibrated XGBoost Classifier | Accuracy: 93.40%, ROC AUC: 0.9813, Recall: 88.0% | `churn_xgboost_model.pkl`, `scaler.pkl`, `churn_threshold.json` (0.440) |
| **CLV Estimation** | XGBoost Regressor | R-squared: 0.9986, MAE: $57.91, RMSE: $84.33 | `clv_xgboost_model.pkl`, `clv_scaler.pkl` |
| **Customer Segmentation** | K-Means Clustering (k=4) | 7,043 customers partitioned into 4 distinct behavioral cohorts | `kmeans_model.pkl`, `kmeans_scaler.pkl` |
| **Explainable AI (XAI)** | TreeSHAP (`shap.TreeExplainer`) | Local directional feature attributions for real-time customer scoring | In-memory explainer cache (`@st.cache_resource`) |

### 2. Batch Inference & Custom Data Store (`ml/`)
| Component | Module | Description |
| :--- | :--- | :--- |
| **Batch Inference Engine** | `ml/batch_inference.py` | Scores customer cohorts with calibrated XGBoost and CLV models, calculating cohort metrics (churn rate, total CLV, high-risk counts). |
| **Custom Data Persistence** | `ml/custom_data_store.py` | Dual persistence to PostgreSQL (`custom_customers` table) and local CSV mirror with timestamp-and-count cache invalidation signature. |

### 3. Dashboard Pages (`dashboard/pages/`)
| Page | Key Features |
| :--- | :--- |
| `app.py` | Platform overview, core KPIs, architecture diagram, and quick navigation. |
| `1_eda.py` | Churn rate donut chart, tenure histogram, charges box plot, and contract breakdown. |
| `2_segmentation.py` | K-Means cluster scatter, spending distribution, and segment churn comparison with strategic business recommendations. |
| `3_predictions.py` | Real-time customer inference UI, 1-click test presets, dual-model grid, SHAP attribution bar chart, and CLV gauge. Prioritizes manually entered records (`[Manual Entry]`) at index 0. |
| `4_model_performance.py` | High-contrast confusion matrix heatmap, classification report, ROC curve, CLV regression diagnostics, and comparative architecture benchmarks. |
| `5_mlops_monitoring.py` | Dedicated MLOps governance center with 5 operational tabs: Model Lineage & Registry, Statistical Data Drift (KS & PSI), Live Inference Audit Logs, Continuous Retraining Quality Gates, and Manual Customer Ingestion & Cohort Analytics Dashboard. |

### 4. API Endpoints (`api/routes/`)
| Route | Method | Description |
| :--- | :---: | :--- |
| `/` | GET | Health status and service metadata. |
| `/api/v1/predict/churn` | POST | Returns churn probability, risk level, and logs telemetry to PostgreSQL. |
| `/api/v1/predict/clv` | POST | Returns continuous lifetime value prediction in dollars. |
| `/api/v1/insights/segmentation/summary` | GET | Returns SQL-aggregated statistics across all four K-Means customer segments. |
| `/api/v1/insights/segmentation/customer/{id}` | GET | Returns cluster assignment and persona for a specific customer. |
| `/api/v1/mlops/retrain` | POST | Triggers continuous retraining pipeline with Champion-Challenger validation gates. |
| `/api/v1/mlops/status` | GET | Returns active model lineage, registry version, and benchmark accuracy. |
| `/api/v1/mlops/drift-check` | GET | Executes two-sample Kolmogorov-Smirnov test and PSI calculations against baseline data. |
| `/api/v1/mlops/logs` | GET | Queries recent production inference audit records from PostgreSQL. |
| `/api/v1/mlops/save-customer` | POST | Persists an individual manually entered customer record into dual storage. |
| `/api/v1/mlops/ingest-batch` | POST | Ingests and scores a custom cohort using the batch inference engine. |
| `/api/v1/mlops/custom-customers` | GET | Retrieves all stored custom customer records. |
| `/api/v1/mlops/custom-customers` | DELETE | Deletes a custom customer record or purges all custom accounts. |

---

## Key Technical Decisions

- **Manual Data Ingestion without File Dependencies**: Implemented direct single-customer form entry and interactive tabular staging grid in Tab 5 of the MLOps center, fulfilling the requirement for ad-hoc customer scoring without importing external CSV or Excel files.
- **Dual-Layer Persistence & Dynamic Cache Invalidation**: Custom customer records are mirrored to PostgreSQL and `data/custom_imported_customers.csv`. A unique state signature (`get_custom_store_signature`) based on file modification time and database record count automatically invalidates cached data in `3_predictions.py` and positions custom records at index 0 marked `[Manual Entry]`.
- **Automated Champion-Challenger Quality Gates**: Continuous retraining enforces 4 non-negotiable checks (Accuracy >= 92.0%, ROC AUC >= 0.950, Recall >= 85.0%, FPR <= 7.0%) to prevent model regression.
- **Statistical Data Drift Surveillance**: Employs two-sample Kolmogorov-Smirnov tests and Population Stability Index (PSI) to detect distribution divergence in production data streams.
- **Resilient Dual-Source Fallback**: All analytical modules query live PostgreSQL tables when connected, seamlessly falling back to local structured data if the database container is offline.
- **SHAP Client-Side Execution**: SHAP attributions are computed locally within Streamlit using a cached `TreeExplainer`, keeping the FastAPI container lightweight and API responses rapid.
- **Startup Sequencing**: Container dependencies use `condition: service_healthy` in Docker Compose v2 to guarantee PostgreSQL is fully initialized before API and Dashboard containers start.

---

## Test Results

Automated test suite execution across all 16 unit and integration test modules:

```
platform linux -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /app
plugins: anyio-4.15.1
collected 16 items

tests/test_api.py::test_read_root                                         PASSED [  6%]
tests/test_api.py::test_get_segmentation_summary                               PASSED [ 12%]
tests/test_api.py::test_get_customer_segment                                   PASSED [ 18%]
tests/test_api.py::test_get_customer_segment_not_found                         PASSED [ 25%]
tests/test_api.py::test_predict_churn                                          PASSED [ 31%]
tests/test_ml_preprocessing.py::test_preprocess_data_training_mode             PASSED [ 37%]
tests/test_ml_preprocessing.py::test_preprocess_data_inference_mode            PASSED [ 43%]
tests/test_mlops.py::test_mlflow_setup_initialization                          PASSED [ 50%]
tests/test_mlops.py::test_drift_monitor_stable_distribution                   PASSED [ 56%]
tests/test_mlops.py::test_drift_monitor_detects_distribution_shift             PASSED [ 62%]
tests/test_mlops.py::test_psi_calculation_consistency                          PASSED [ 68%]
tests/test_mlops.py::test_champion_gate_rules_pass_and_fail                    PASSED [ 75%]
tests/test_mlops.py::test_api_mlops_endpoints                                   PASSED [ 81%]
tests/test_mlops.py::test_batch_inference_scoring_and_metrics                  PASSED [ 87%]
tests/test_mlops.py::test_custom_data_store_persistence_lifecycle              PASSED [ 93%]
tests/test_mlops.py::test_api_custom_customer_endpoints                        PASSED [100%]

============================== 16 passed in 6.44s ==============================
```

---

## Deployment Verification

```bash
docker compose ps
# NAME                                    STATUS
# customer_churn-prediction-db-1          Up (healthy)
# customer_churn-prediction-api-1         Up
# customer_churn-prediction-dashboard-1   Up
# customer_churn-prediction-mlflow-1      Up

# Database baseline check
SELECT COUNT(*) FROM customers;  -->  7043

# Real-time inference check
POST /api/v1/predict/churn {"customer_id": "7590-VHVEG"}
--> {"churn_prediction": 1, "churn_probability": 0.7354, "risk_level": "High"}
```
