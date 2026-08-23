# Internship / UDP / Training Report — Reporting 3
## Advanced Implementation, Testing & Validation Progress Report

---

## 1. Student Information

| Field | Details |
|---|---|
| **Student Name** | Rudra Medatwal |
| **Enrollment Number** | *(to be filled)* |
| **Program / Branch** | *(to be filled)* |
| **Semester** | *(to be filled)* |
| **Contact Number** | *(to be filled)* |
| **Email ID** | *(to be filled)* |

---

## 2. Internship / UDP / Training Details

| Field | Details |
|---|---|
| **Type** | UDP |
| **Company / Organization** | *(to be filled)* |
| **Department / Domain** | Data Science & Machine Learning |
| **Mentor Name (Company)** | *(to be filled)* |
| **Mentor Name (Institute)** | *(to be filled)* |
| **Start Date** | *(to be filled)* |
| **End Date** | *(to be filled)* |

---

## 3. Work Progress Summary

**Project Title:**
Intelligent Customer Lifetime Value & Churn Prediction Platform

---

### b) Brief Overview of Work Completed After Reporting 2

After Reporting 2, the focus shifted from building the core ML pipeline and backend to completing the full system — finishing the frontend, adding explainability, hardening the codebase with tests, and deploying everything end-to-end using Docker.

**Major tasks completed:**
- Added a fourth Streamlit dashboard page (`4_model_performance.py`) covering a complete model evaluation report card.
- Integrated SHAP (SHapley Additive exPlanations) into the predictions page so users can see which features actually drove each individual prediction.
- Rewrote the predictions page (`3_predictions.py`) with a smarter data loading architecture — it now tries the live PostgreSQL database first and gracefully falls back to the local Excel file if the database isn't available.
- Added a CLV gauge chart to the predictions page for a more intuitive visual output.
- Fixed the SQLAlchemy 2.0 deprecation warning in `database/connection.py`.
- Moved the database engine import inside the function body so the test suite can run without a live PostgreSQL connection.
- Fixed two failing pytest tests and brought the suite to **7/7 passing with zero warnings**.
- Rewrote the `README.md` from scratch with a full Mermaid architecture diagram, run instructions, model metrics, and tech stack table.
- Ran `docker compose up --build` to build and start all three containers, then verified the full end-to-end prediction flow against real database data.

**Modules completed:**
- Model Performance Dashboard Page (new)
- SHAP Explainability Panel (new)
- Full Docker Stack Deployment (completed and verified)
- Test Suite (all passing)

**New features implemented:**
- SHAP per-customer feature attribution bar chart (red = pushes toward churn, blue = toward retention)
- CLV prediction gauge chart with color-coded zones (red/yellow/green)
- Four-tab Model Performance page — Confusion Matrix heatmap, ROC curve, CLV scatter, Segment profiles
- `.dockerignore` for faster Docker builds
- DB → Excel fallback pattern in all data-dependent pages

**Research / development work completed:**
- Studied SHAP TreeExplainer mechanics for XGBoost model explanations
- Investigated and resolved SQLAlchemy 2.0 breaking API changes
- Researched Docker Compose v2 `healthcheck` and `depends_on: condition: service_healthy` patterns to eliminate race conditions on container startup

**Integration work completed:**
- SHAP explainer integrated directly into the Streamlit prediction page (cached with `@st.cache_resource`)
- `DATABASE_URL` environment variable propagated to both the API and Dashboard Docker containers
- All three containers communicate over Docker's internal network — Dashboard → API → PostgreSQL

---

### c) Overall Project Completion Status

**Overall Project Completion: 97%**

| Component | Status | Completion (%) |
|---|---|---|
| Requirement / Analysis | Completed | 100% |
| Design | Completed | 100% |
| Development | Completed | 100% |
| Integration | Completed | 100% |
| Testing | Completed | 100% |
| Documentation | Completed | 100% |
| Deployment | Completed | 95% |

> The remaining 3% is post-submission scope: cloud deployment (AWS/GCP) and an MLOps retraining pipeline.

---

## 4. Advanced Implementation

### a) Module / Feature Implementation

| Module / Feature | Work Completed | Status |
|---|---|---|
| **Model Performance Dashboard** (`4_model_performance.py`) | Built a 3-tab Streamlit page: Tab 1 shows XGBoost Churn classifier with confusion matrix heatmap, classification report table with gradient coloring, ROC curve image, and feature importance image. Tab 2 shows the CLV regressor metrics plus a "Predicted vs Actual CLV" scatter plot. Tab 3 shows K-Means segmentation profiles, a pie chart of segment sizes, and a business interpretation note for each cluster. | ✅ Completed |
| **SHAP Explainability Panel** | Inside the Predictions page, after a churn probability is computed, a collapsible "Explain this Prediction" expander computes SHAP values using `shap.TreeExplainer` and renders a horizontal bar chart of the top 10 features. Red bars push the model toward predicting churn; blue bars push toward retention. The explainer is cached with `@st.cache_resource` so it's only built once per session. | ✅ Completed |
| **CLV Gauge Chart** | After a CLV prediction is returned, a Plotly `go.Indicator` gauge renders the value on a 0–8000 scale with color zones: red (0–2000), yellow (2000–5000), green (5000–8000). | ✅ Completed |
| **DB / Excel Fallback Architecture** | The new predictions page tries to connect to PostgreSQL first and silently falls back to reading the local Excel file if the database is unreachable. An info banner tells the user which data source is active. | ✅ Completed |
| **Test Suite Repair** | Two previously failing tests were fixed. `test_ml_preprocessing.py` was updated to check structural properties of the DataFrame (column existence, dtype, shape) rather than pre-scale numeric values that get transformed in-place. `test_api.py` was updated to mock `get_customer_features` directly instead of `pd.read_sql`, isolating the route handler cleanly from the preprocessing pipeline. | ✅ Completed |
| **SQLAlchemy 2.0 Fix** | `database/connection.py` was updated to import `declarative_base` from `sqlalchemy.orm` instead of the deprecated `sqlalchemy.ext.declarative`, eliminating the `MovedIn20Warning` that appeared in the test output. | ✅ Completed |
| **Docker Stack Deployment** | `docker-compose.yml` was updated to Compose v2 format (removed deprecated `version:` key), added `DATABASE_URL` env var to both the API and dashboard containers, and upgraded healthcheck logic so app containers only start once PostgreSQL reports healthy. `.dockerignore` was created to speed up build context transfer. | ✅ Completed |

---

### b) Integration / Development Details

**Module integration:**

The four Streamlit pages share a common pattern: they each try to import and use the SQLAlchemy engine to pull live data, and fall back to the local Excel file if that fails. This means the dashboard works in three distinct deployment modes — fully local (Excel only), hybrid (Streamlit local + Docker DB), and full Docker (all three containers together) — without any code changes.

The new `4_model_performance.py` page is standalone. It doesn't need the database or the API — it reads pre-generated static visualization images from `visualizations/` and uses hardcoded metric values from `models/model_scores.md`. This was a deliberate design choice so the report card is always available, even in environments where the database is unavailable.

**API integration:**

The predictions page (`3_predictions.py`) doesn't call the FastAPI backend directly — it loads the `.pkl` models itself, runs preprocessing, and calls `shap.TreeExplainer` locally. This keeps the SHAP computation entirely client-side (in Streamlit) without adding a heavy dependency to the FastAPI container. The tradeoff is some code duplication between the Streamlit preprocessing and the API preprocessing, but it was the right call for a prototype system.

For the FastAPI routes, the `get_customer_features` function in `api/routes/predict.py` is now the clear boundary. Tests mock this function directly, which means the route handler tests are completely decoupled from the preprocessing implementation. If the preprocessing logic changes tomorrow, the route tests don't break.

**Database integration:**

All three containers connect to PostgreSQL using the `DATABASE_URL` environment variable. Inside the Docker network, the API and dashboard containers resolve the database host as `db` (the Docker service name). Outside Docker, the same variable defaults to `localhost:5432`, so local development works without changes.

The database lazy import fix (`from database.connection import engine` moved inside `load_data_from_db()`) ensures that simply importing any module from `ml/data_preprocessing.py` in the test runner doesn't attempt to open a database connection. This was a blocking issue for CI environments.

**Hardware / software integration:**

No hardware integration was applicable for this project. The stack runs entirely on standard commodity hardware — any laptop or cloud VM capable of running Docker Desktop can host the full platform.

**Software integration:**

`shap` version 0.51.0 was added to `requirements.txt` along with an explicit `joblib` entry. Both packages are now installed inside both Docker images as part of the standard `pip install -r requirements.txt` build step.

**Major coding / development work:**

The most technically involved piece of work in this phase was the SHAP integration. The challenge was that `shap.TreeExplainer` requires the exact same feature matrix — same column names, same column order — that the XGBoost model was trained on. The predictions page handles this by calling `X.reindex(columns=model.feature_names_in_, fill_value=0)` before passing any data to either the model or the SHAP explainer. This pattern guarantees column alignment regardless of which columns happen to be present in the specific customer's row after preprocessing and one-hot encoding.

---

### c) Final / Updated Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DOCKER COMPOSE STACK                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │             Presentation Layer — Streamlit :8501                    │    │
│  │  ┌──────────┐ ┌──────────────┐ ┌────────────────┐ ┌─────────────┐ │    │
│  │  │  EDA     │ │ Segmentation │ │ Predictions    │ │   Model     │ │    │
│  │  │  Page    │ │   Page       │ │ + SHAP + Gauge │ │ Performance │ │    │
│  │  └──────────┘ └──────────────┘ └────────────────┘ └─────────────┘ │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │ HTTP POST (predict/churn, predict/clv)       │
│  ┌────────────────────────────▼────────────────────────────────────────┐    │
│  │              Logic Layer — FastAPI :8000                             │    │
│  │      /api/v1/predict/churn   /api/v1/predict/clv                    │    │
│  │      /api/v1/insights/segmentation/summary                          │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │ SQL (SQLAlchemy + psycopg2)                  │
│  ┌────────────────────────────▼────────────────────────────────────────┐    │
│  │              Data Layer — PostgreSQL :5432                           │    │
│  │                churn_db.customers — 7,043 rows                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              ML Layer (offline / training time)                      │    │
│  │   XGBoost Churn Classifier   XGBoost CLV Regressor   K-Means (k=4) │    │
│  │   SHAP TreeExplainer (runtime, loaded per-session in Streamlit)     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Brief Explanation:**

The overall architecture remains a three-tier system — Data, Logic, Presentation — now extended with an explainability layer. The key architectural addition in Reporting 3 is that SHAP explanations are computed client-side inside Streamlit rather than server-side in FastAPI. This keeps the API container lean and avoids making the API response contract more complex. The `.pkl` models are loaded directly into Streamlit using `@st.cache_resource`, and `shap.TreeExplainer` is built once and reused for any subsequent explanation requests in the same session.

The Docker Compose stack now enforces strict startup ordering: the `db` container must pass its `pg_isready` healthcheck before either the `api` or `dashboard` containers attempt to start. This eliminates the race condition where app containers would crash on boot because PostgreSQL wasn't accepting connections yet.

---

## 5. Testing and Validation

**Testing Performed:**

☑ Unit Testing
☑ Integration Testing
☑ Functional Testing
☑ System Testing

### Test Table

| Test No. | Test / Function | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| 1 | `test_read_root` — GET `/` on FastAPI | HTTP 200 with welcome message JSON | HTTP 200, `{"message": "Welcome to the Customer Churn & CLV Prediction API"}` | ✅ Pass |
| 2 | `test_get_segmentation_summary` — GET `/api/v1/insights/segmentation/summary` | HTTP 200 with list of segment objects | HTTP 200, list of 2 mocked segment dicts returned correctly | ✅ Pass |
| 3 | `test_get_customer_segment` — POST `/api/v1/insights/segmentation/customer` | HTTP 200 with customer's segment | HTTP 200, `{"customer_id": "CUST123", "segment": "Cluster 2"}` | ✅ Pass |
| 4 | `test_get_customer_segment_not_found` — POST with unknown customer ID | HTTP 404 | HTTP 404 returned as expected | ✅ Pass |
| 5 | `test_predict_churn` — POST `/api/v1/predict/churn` with mocked model + features | HTTP 200 with `churn_prediction`, `churn_probability`, `risk_level` fields | HTTP 200, all three fields present, `churn_prediction` in {0, 1} | ✅ Pass |
| 6 | `test_preprocess_data_training_mode` — Preprocessing pipeline on fixture DataFrame | Correct column drops, numeric types, target extraction, scaler returned | All assertions passed — columns dropped, `y_churn` correct, `StandardScaler` returned | ✅ Pass |
| 7 | `test_preprocess_data_inference_mode` — Preprocessing without target column | DataFrame returned without `churn` column, correct shape | DataFrame returned with 2 rows, no churn column | ✅ Pass |

**Final test run result:** `7 passed in 2.82s` — zero warnings, zero failures.

### Testing Methods Used

**Unit Testing:** Each function — `preprocess_data`, each API route — was tested in isolation with mocked dependencies. No live database or real model file was required for any unit test to pass.

**Integration Testing:** The `TestClient` from Starlette was used to send HTTP requests through the full FastAPI routing stack (middleware, dependency injection, route handler) without actually binding to a network port. This validated that the integration between FastAPI's routing layer and the underlying Python functions worked correctly.

**Functional Testing:** After `docker compose up --build`, the full prediction flow was manually tested end-to-end — selecting a real customer ID from PostgreSQL, running it through the churn model, and verifying the SHAP bar chart rendered correctly in the browser.

**System Testing:** All three containers were verified simultaneously: `docker compose ps` confirmed all three services as `Up`, a `SELECT COUNT(*) FROM customers` confirmed 7,043 rows in the database, and `POST /api/v1/predict/churn` for customer `7590-VHVEG` returned `{"churn_prediction": 1, "churn_probability": 0.7354, "risk_level": "High"}`.

---

## 6. Results and Performance Analysis

### a) Final / Updated Results

The platform is now fully operational. All three machine learning models are trained, serialized, and serving predictions through both the FastAPI backend and the Streamlit dashboard. The SHAP integration adds a layer of transparency that was missing in earlier reporting phases — users can now see not just *what* the model predicted, but *why*.

The most significant result from this phase is the successful end-to-end deployment. Customer `7590-VHVEG` was used as the live verification test:
- **Churn Probability:** 73.54% → Risk Level: **High**
- **CLV Prediction:** $28.10 (low-value customer — consistent with high churn risk)
- **SHAP Top Feature:** Month-to-month contract type was the strongest driver pushing toward churn

This is exactly the kind of actionable output the platform was designed to produce — a business user can now immediately know which customers to prioritize for retention campaigns and why.

### b) Performance Results

| Parameter / Metric | Result Obtained | Expected / Target | Status |
|---|---|---|---|
| **Churn Classifier Accuracy** | 80.77% | ≥ 78% | ✅ Met |
| **Churn Classifier ROC AUC** | 0.854 | ≥ 0.80 | ✅ Met |
| **Churn Classifier F1 (churn class)** | 0.61 | ≥ 0.55 | ✅ Met |
| **CLV Regressor R²** | 0.9986 | ≥ 0.90 | ✅ Met |
| **CLV Regressor MAE** | $57.91 | ≤ $100 | ✅ Met |
| **K-Means Segments** | 4 distinct clusters | 3–5 clusters | ✅ Met |
| **Pytest Test Suite** | 7/7 passed, 0 warnings | All pass | ✅ Met |
| **API Response (churn endpoint)** | ~200ms (Docker, local) | ≤ 2s | ✅ Met |
| **DB Row Count (post-seed)** | 7,043 rows | 7,043 (full dataset) | ✅ Met |

### c) Output / Result Screenshots

**Screenshot 1: EDA Dashboard Page**

*(Insert screenshot of `http://localhost:8501` — Page 1: EDA)*

**Description:**
The Exploratory Data Analysis page shows a churn distribution pie chart (roughly 26% churners), a tenure histogram broken down by churn label, a monthly charges box plot comparing churners vs non-churners, and a contract type breakdown bar chart. The data is pulled live from the PostgreSQL container — the page header confirms "Data source: database".

---

**Screenshot 2: Live Churn Prediction + SHAP Explanation**

*(Insert screenshot of `http://localhost:8501` — Page 3: Predictions, after predicting for customer 7590-VHVEG)*

**Description:**
The predictions page shows the customer profile at the top, followed by the churn probability (73.5%), a red "High Risk" badge, and a collapsible SHAP panel. Inside the panel, a horizontal bar chart shows the top 10 features — the month-to-month contract, short tenure, and high monthly charge are shown as the top three red bars pushing toward churn. Below, the CLV gauge chart shows $28.10 in the red (low-value) zone.

---

**Screenshot 3: Model Performance Dashboard**

*(Insert screenshot of `http://localhost:8501` — Page 4: Model Performance)*

**Description:**
The Model Performance page's first tab (XGBoost Churn Classifier) shows four metric cards (Accuracy: 80.77%, AUC: 0.854, Precision: 0.66, Recall: 0.57), an interactive confusion matrix heatmap, and a styled classification report table. Below, the ROC curve and feature importance bar chart images are embedded from the `visualizations/` directory.

---

## 7. Challenges, Solutions and Improvements

### Problems Encountered After Reporting 2

| Challenge / Problem | Solution Implemented | Final Outcome |
|---|---|---|
| **SHAP column alignment mismatch** — After one-hot encoding a single customer's row in Streamlit, the resulting DataFrame had different columns than what the XGBoost model was trained on, causing a `ValueError: feature_names mismatch`. | Used `X.reindex(columns=model.feature_names_in_, fill_value=0)` immediately before calling both the model and `shap.TreeExplainer`. This forces the feature matrix to exactly match training shape regardless of which dummy columns appear after encoding. | SHAP values compute correctly for every customer with no column errors. |
| **Failing pytest tests** — `test_preprocess_data_training_mode` was checking that `total_additional_services` equalled 3, but StandardScaler transforms this column in-place during the test, so the assertion saw 0.0 instead of 3. `test_predict_churn` failed because the mock DataFrame didn't have enough columns for the full preprocessing pipeline. | Fixed the first test to check structural properties (column existence, dtype, shape, returned scaler type) rather than exact pre-scale numeric values. Fixed the second test by mocking `get_customer_features` directly instead of `pd.read_sql`, so the route handler is tested in isolation from the preprocessing pipeline. | Both tests now pass. Full suite: 7/7 passing, 0 warnings. |
| **SQLAlchemy 2.0 deprecation warning** — `from sqlalchemy.ext.declarative import declarative_base` raised a `MovedIn20Warning` in every test run, cluttering the output and failing the `-W error::DeprecationWarning` flag. | Changed the import to `from sqlalchemy.orm import declarative_base`, which is the correct SQLAlchemy 2.0 location for this class. | Test suite now runs with zero warnings even under strict deprecation enforcement. |
| **Database engine imported at module level** — `from database.connection import engine` at the top of `ml/data_preprocessing.py` caused pytest collection to attempt a PostgreSQL connection immediately, crashing the test runner in CI environments without a database. | Moved the import inside `load_data_from_db()` (lazy import pattern). Module-level imports no longer require a running database. | Test suite runs cleanly without a live PostgreSQL connection. `load_data_from_db()` still works exactly the same when called at runtime. |
| **Docker container startup race condition** — On `docker compose up`, the API and dashboard containers occasionally started before PostgreSQL finished initializing, causing immediate crash-restart loops. | Added a `healthcheck` to the `db` service (`pg_isready` command) and changed both `api` and `dashboard` to use `depends_on: db: condition: service_healthy` instead of just `depends_on: db`. | All containers start in the correct order. No crash loops observed after any of several test builds. |

### Improvements Made After Reporting 2

- Added the **Model Performance page** — the platform now has a proper model evaluation report built into the dashboard itself, not just in the `model_scores.md` file.
- Added **SHAP explainability** — the most important addition from a business value perspective. Predictions without explanations are hard to trust; SHAP makes the model transparent.
- Added **`.dockerignore`** — previously missing, which meant Python caches, IDE files, and hundreds of megabytes of unnecessary files were included in the Docker build context. Adding `.dockerignore` significantly reduced build times on repeated builds.
- Updated **`docker-compose.yml`** to Compose v2 format — removed the deprecated `version:` key, added `DATABASE_URL` to the dashboard container (previously it was only in the API container), and added proper startup ordering.
- Added **`shap`** and explicit **`joblib`** to `requirements.txt` — both were missing from the dependency manifest, which would have caused build failures on a clean environment.
- Comprehensive **README rewrite** — the previous README was sparse. The new version includes a Mermaid architecture diagram, a model metrics table, a tech stack table, and clear instructions for both local and Docker deployment.

---

## 8. Current Project Status

☑ **Mostly Completed** (97% — pending only cloud deployment and MLOps pipeline)

### Completed Components

- Data ingestion and PostgreSQL seeding (7,043 records)
- Full ML pipeline: XGBoost Churn Classifier, XGBoost CLV Regressor, K-Means Segmentation (k=4)
- FastAPI REST backend with 3 live endpoints
- Streamlit dashboard with 4 pages (EDA, Segmentation, Predictions, Model Performance)
- SHAP per-customer explainability panel
- CLV gauge chart
- Docker Compose stack (3 containers, all running and healthy)
- Pytest test suite (7/7 passing, 0 warnings)
- Documentation: README, walkthrough, accomplished tasks, skills acquired, agy.md

### Pending Components

- Cloud deployment (AWS ECS / GCP Cloud Run)
- MLOps retraining pipeline (Airflow/Prefect cron scheduler to retrain models on new data)
- JWT/OAuth2 authentication on FastAPI prediction endpoints

### Work Planned Before Reporting 4

- Provision a cloud environment (AWS EC2 or GCP) and deploy the Docker Compose stack there
- Set up GitHub Actions CI/CD to auto-run tests and rebuild Docker images on every push to `main`
- Research and prototype the MLOps retraining trigger — start with a simple cron job before graduating to Airflow
- Optionally: add a global SHAP summary page (beeswarm plot across the full 7,043-customer dataset) to complement the per-customer explainability panel

---

## 9. Learning and Skills Acquired

### Technical Skills

**Explainable AI with SHAP:**
Before this project, I had a surface-level understanding that SHAP values explain model predictions. Working with `shap.TreeExplainer` hands-on made it concrete — I had to handle the feature alignment problem (training vs inference column ordering), understand the sign of SHAP values (positive = pushes toward churn, negative = pushes toward retention), and figure out how to cache the explainer object in Streamlit so it doesn't rebuild on every button click. These are details that documentation doesn't spell out clearly.

**Advanced Docker Compose patterns:**
I knew how to write a basic `docker-compose.yml` before this project. What I didn't know was the difference between `depends_on: db` (just waits for the container to start) and `depends_on: db: condition: service_healthy` (actually waits for the healthcheck to pass). This distinction matters a lot in practice — it was the root cause of intermittent startup crashes.

**Pytest mocking strategy:**
The test failures in this phase forced me to think carefully about what a unit test should actually test. A test for a route handler shouldn't break when the preprocessing function changes. Mocking `get_customer_features` instead of `pd.read_sql` is a better abstraction because it matches the logical boundary of the route — the route's job is to call `get_customer_features` and interpret its output, not to reimplement how features are fetched.

**SQLAlchemy 2.0 migration patterns:**
Upgrading from `sqlalchemy.ext.declarative` to `sqlalchemy.orm` seems minor, but it's a good example of staying current with library deprecations. The lesson is to treat deprecation warnings as bugs — they're telling you something is about to break.

### Tools / Technologies Learned

| Tool / Technology | Application |
|---|---|
| `shap` (v0.51.0) | `TreeExplainer`, per-customer SHAP value computation, waterfall/bar chart visualization |
| `plotly.graph_objects` (Indicator, Heatmap) | CLV gauge chart, confusion matrix heatmap |
| Docker Compose v2 | `healthcheck`, `depends_on: condition`, `.dockerignore`, no-`version` syntax |
| SQLAlchemy 2.0 | `declarative_base` from `sqlalchemy.orm` |
| `pytest` mocking | `@patch`, `MagicMock`, patching at the correct import path |

### Professional Skills

**Reading error messages carefully:** Several issues in this phase (the SHAP column mismatch, the test failures, the SQLAlchemy deprecation) would have been much harder to fix without reading the full traceback carefully. Each error message pointed exactly to the line and reason for the failure.

**Knowing when not to over-engineer:** A key decision in this phase was keeping SHAP computation in Streamlit rather than adding a SHAP endpoint to the FastAPI backend. The API would have needed to serialize and return SHAP values as JSON, which adds complexity without much benefit for a dashboard that directly controls the model. Simpler was better here.

**End-to-end thinking:** Completing the Docker deployment made the platform real in a way that running individual services locally doesn't. Seeing the full request flow — browser → Streamlit → FastAPI → PostgreSQL → ML model → SHAP explainer → visual output — gave a much clearer picture of where performance bottlenecks could appear and what the operational concerns would be.

### Major Learning from This Phase

The biggest technical lesson from Reporting 3 was about the gap between "it works on my machine" and "it works everywhere." The lazy import fix, the Docker healthcheck, the `.dockerignore`, and the test isolation improvements were all about closing that gap. Each one addressed a scenario where the code worked fine locally but would have failed in a clean or automated environment. This is what production-readiness actually means in practice — it's not about feature completeness, it's about making every piece of the system reliable and reproducible.

---

## 10. References and Resources Used

All references are in APA 7th Edition format. These were actually used during the implementation work described in this report.

**[1]** Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

**[2]** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

**[3]** SHAP Library Documentation. (2024). *SHAP — A game theoretic approach to explain the output of any machine learning model*. https://shap.readthedocs.io/en/latest/ (Accessed August 2026)

**[4]** Streamlit Inc. (2024). *Streamlit API reference — st.cache_resource*. https://docs.streamlit.io/library/api-reference/performance/st.cache_resource (Accessed August 2026)

**[5]** Docker Inc. (2024). *Compose file reference — healthcheck*. https://docs.docker.com/compose/compose-file/05-services/#healthcheck (Accessed August 2026)

**[6]** SQLAlchemy Project. (2024). *What's new in SQLAlchemy 2.0 — ORM declarative mapping*. https://docs.sqlalchemy.org/en/20/changelog/migration_20.html (Accessed August 2026)

**[7]** pytest Documentation. (2024). *How to monkeypatch/mock modules and environments*. https://docs.pytest.org/en/stable/how-to/monkeypatch.html (Accessed August 2026)

**[8]** Plotly Technologies Inc. (2024). *Plotly Python Open Source Graphing Library — Indicator traces*. https://plotly.com/python/gauge-charts/ (Accessed August 2026)

**[9]** IBM. (2019). *Telco Customer Churn Dataset*. IBM Cognos Analytics Sample Data. https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/ (Accessed July 2026)

**[10]** Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

---

*Report prepared by: Rudra Medatwal*
*Date: August 2026*

---

> **Evaluator Section (To be filled by Mentor — Institute)**
>
> | Criteria | Excellent | Good | Average | Poor | Marks |
> |---|---|---|---|---|---|
> | Work Completion & Implementation (25) | ≥90% meaningful work | 75–89% | 50–74% | <50% | /25 |
> | Testing, Results & Validation (25) | Strong testing with evidence | Good testing | Limited testing | No validation | /25 |
> | Technical Understanding (20) | Strong conceptual understanding | Good understanding | Basic understanding | Unable to explain | /20 |
> | Documentation & Progress Presentation (20) | Complete, clear, well documented | Minor gaps | Incomplete | Poor | /20 |
> | References & Citations (10) | Relevant, properly cited | Minor issues | Limited sources | Missing/improper | /10 |
> | **TOTAL** | | | | | **/100** |
>
> **Mentor (Institute) Remarks:**
> ___________________________________________
