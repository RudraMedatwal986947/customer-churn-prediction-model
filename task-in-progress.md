# Tasks In Progress

## Project: Intelligent Customer Lifetime Value & Churn Prediction Platform

> **All core tasks are complete.** The platform is fully built, tested, and deployed.
> See [`accomplished_tasks.md`](accomplished_tasks.md) for the full record.

---

## Current Status: PROJECT COMPLETE

| Phase | Status |
|-------|--------|
| 1. Project Infrastructure & Docker | Complete |
| 2. Database (PostgreSQL + ORM + Seeding) | Complete |
| 3. Machine Learning Pipeline (3 models) | Complete |
| 4. Backend API (FastAPI — 3 endpoints) | Complete |
| 5. Frontend Dashboard (Streamlit — 4 pages) | Complete |
| 6. Testing (7/7 pytest tests passing) | Complete |
| 7. Docker Deployment (full stack live) | Complete |

---

## Running Services
- **Dashboard** → http://localhost:8501
- **API Swagger** → http://localhost:8000/docs
- **PostgreSQL** → localhost:5432 / `churn_db` / 7,043 rows

---

## Future Scope (Post-Submission)
- [ ] **SHAP Global Summary**: Aggregate SHAP values across the full dataset (beeswarm plot).
- [ ] **MLOps / Retraining Pipeline**: Trigger model retraining on data drift using Airflow or Prefect.
- [ ] **Real-Time Streaming**: Kafka-based live customer event ingestion.
- [ ] **Deep Learning CLV**: LSTM / Transformer model for time-series CLV prediction.
- [ ] **CRM Integration**: Connect to Salesforce / HubSpot API for automated retention actions.
- [ ] **Cloud Deployment**: Deploy Docker stack to AWS ECS, GCP Cloud Run, or Azure ACI.
- [ ] **Explainable AI Page**: Full SHAP dashboard page with force plots, dependency plots, and interaction values.
