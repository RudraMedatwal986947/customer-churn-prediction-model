# Skills & Technologies Acquired

Through the complete design, implementation, testing, and deployment of the **Intelligent Customer Lifetime Value & Churn Prediction Platform**, the following skills, tools, and methodologies have been acquired and practically applied:

---

## 1. Machine Learning & Data Science
- **Predictive Modeling**: XGBoost Classifier for binary churn prediction (80.77% accuracy, AUC 0.854) and XGBoost Regressor for CLV estimation (R² 0.9986, MAE $57.91).
- **Customer Segmentation**: K-Means clustering (k=4) with elbow-method selection; RFM-inspired feature set.
- **Feature Engineering**: Derived `tenure_group`, `avg_monthly_charge`, `charge_difference`, `total_additional_services` from raw telco data.
- **Data Preprocessing Pipeline**: Handling missing values (`TotalCharges` coercion), One-Hot Encoding for categorical variables, StandardScaler for numerical features.
- **Explainable AI (XAI)**: Integrating **SHAP (SHapley Additive exPlanations)** with `shap.TreeExplainer` to produce per-customer feature attribution charts.
- **Libraries**: `scikit-learn`, `xgboost`, `shap`, `pandas`, `numpy`, `joblib`.

---

## 2. Backend & API Development
- **RESTful APIs**: Designed and implemented endpoints for real-time ML inference (`/predict/churn`, `/predict/clv`) and data aggregation (`/insights/segmentation`).
- **FastAPI**: High-performance async Python API framework with automatic Swagger UI documentation at `/docs`.
- **Lazy Model Loading**: Global model variables loaded on first request to avoid startup overhead.
- **Pydantic**: Request body validation using `BaseModel`.
- **Uvicorn**: ASGI server with `--reload` for development hot-reloading.

---

## 3. Database Management & ORM
- **PostgreSQL 15**: Relational database hosting 7,043 structured customer records.
- **SQLAlchemy 2.0**: ORM with `declarative_base` (updated from deprecated `sqlalchemy.ext.declarative`), `create_engine`, `SessionLocal`.
- **Data Seeding**: Scripted ETL from Excel to PostgreSQL (`database/seed_db.py`) with duplicate-safe `merge` logic.
- **SQL Queries**: Direct `pd.read_sql()` for aggregation queries from Python.

---

## 4. Frontend & Data Visualization
- **Streamlit**: Built a 4-page interactive web application entirely in Python — EDA, Segmentation, Predictions + SHAP, Model Performance.
- **Plotly**: Interactive charts — pie, histogram, box, scatter, bar, heatmap (`go.Heatmap`), gauge (`go.Indicator`), horizontal bar for SHAP values.
- **Multi-page App Architecture**: Streamlit `pages/` layout with shared cached data loaders (`@st.cache_data`, `@st.cache_resource`).
- **DB/File Fallback Pattern**: Pages gracefully degrade from PostgreSQL → local Excel when the database is unavailable.

---

## 5. DevOps & Containerization
- **Docker**: Wrote `Dockerfile.api` and `Dockerfile.dashboard` using `python:3.11-slim` base image with `--no-cache-dir` pip installs.
- **`.dockerignore`**: Created to exclude caches, IDE files, and test artifacts from build context.
- **Docker Compose v2**: Multi-container orchestration with service `healthcheck` for PostgreSQL, `depends_on: condition: service_healthy` ordering, environment variable injection, and bind-mount volumes.
- **Container Exec**: Running seed scripts and ad-hoc commands inside containers with `docker compose exec`.
- **End-to-End Deployment**: Full stack (`db` → `api` → `dashboard`) deployed and verified with live API calls returning real ML predictions.

---

## 6. Software Testing
- **Pytest**: Wrote 7 unit tests (5 API + 2 ML preprocessing), all passing with 0 warnings.
- **Mocking**: Used `unittest.mock.patch` and `MagicMock` to isolate units — mocking `joblib.load`, `pd.read_sql`, and `get_customer_features`.
- **Test Design**: Fixed brittle tests (pre-scale value assertions replaced with structural checks; DB-coupled imports replaced with lazy imports).
- **Deprecation Enforcement**: Ran tests with `-W error::DeprecationWarning` to guarantee a clean bill of health.

---

## 7. Software Architecture & Project Management
- **End-to-End System Design**: Connected raw data ingestion → preprocessing → model training → API serving → interactive dashboard in a single cohesive platform.
- **Modular Project Structure**: Clean separation of concerns (`data/`, `ml/`, `api/`, `dashboard/`, `database/`, `tests/`, `models/`, `visualizations/`).
- **Technical Documentation**: Maintained `README.md`, `agy.md`, `implementation-plan.md`, `accomplished_tasks.md`, `task-in-progress.md`, `walkthrough.md`, `project_report_answers.md`, and `models/model_scores.md` throughout the project lifecycle.
- **Iterative Development**: Used planning artifacts and approval gates to manage incremental feature delivery across multiple sessions.
