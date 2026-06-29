# Customer Churn & CLV Prediction Platform - Project Memory

## 🎯 Project Goals
An End-to-End Machine Learning Platform for Customer Segmentation, Customer Lifetime Value Prediction, Churn Prediction, and Business Intelligence.

## 🛠 Tech Stack
- **Database**: PostgreSQL (Dockerized)
- **Backend API**: FastAPI
- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas
- **Deployment**: Docker, Docker Compose

## ✅ Accomplished Tasks
- [x] Initialized project directory structure (`data/`, `ml/`, `api/`, `dashboard/`, `database/`).
- [x] Created standard `requirements.txt` and python `.gitignore` (ignoring data and `.pkl` models).
- [x] Set up `docker-compose.yml` (PostgreSQL, FastAPI, Streamlit) with local bind mounts for live-reload.
- [x] Configured database ORM models in `database/models.py`.
- [x] Created `database/seed_db.py` to seed PostgreSQL with the Kaggle Telco Churn dataset (`Telco_customer_churn.xlsx`).
- [x] Bootstrapped FastAPI routes and Streamlit dashboard pages.
- [x] Implemented robust data preprocessing (`ml/data_preprocessing.py`):
    - Cleaned missing values (e.g., empty `total_charges`).
    - Feature Engineered new attributes (`tenure_group`, `total_additional_services`, `avg_monthly_charge`, `charge_difference`).
    - Applied One-Hot Encoding and Standard Scaling.
- [x] Exported the processed data to `data/cleaned_customer_data.csv`.
- [x] **ML Modeling (Churn)**: Implemented XGBoost classifier (80.77% accuracy, 0.85 AUC).
- [x] **ML Modeling (CLV)**: Implemented XGBoost regressor (R² 0.99, MAE $57).
- [x] **Segmentation**: Implemented K-Means clustering (4 segments) and mapped to database.
- [x] **Visualization**: Generated ML plots (Feature Importance, ROC, Confusion Matrix, Clusters) into `visualizations/`.
- [x] **API Integration**: Loaded the `.pkl` models into FastAPI routes (`predict.py`, `insights.py`) to serve live predictions.
- [x] Committed and pushed codebase to GitHub remote repository.

## 🚀 Current Status
- Docker stack (`db`, `api`, `dashboard`) is running perfectly.
- Database contains 7,043 mapped and segmented records.
- The FastAPI backend (`http://localhost:8000`) is actively serving ML predictions and data aggregations.

## 📝 Next Steps
- [ ] **Dashboard Integration**: Write the Streamlit UI code in `dashboard/pages/` to consume the FastAPI endpoints and display interactive metrics, predictions, and the generated visualizations.
