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
- [x] Created standard `requirements.txt` and python `.gitignore`.
- [x] Set up `docker-compose.yml` (PostgreSQL, FastAPI, Streamlit) with local bind mounts for live-reload.
- [x] Configured database ORM models in `database/models.py`.
- [x] Created `database/seed_db.py` to seed PostgreSQL with the Kaggle Telco Churn dataset (`Telco_customer_churn.xlsx`).
- [x] Bootstrapped FastAPI routes (`/predict/churn`, `/predict/clv`, `/insights/segmentation`).
- [x] Bootstrapped Streamlit dashboard pages (EDA, Segmentation, Predictions).
- [x] Implemented robust data preprocessing (`ml/data_preprocessing.py`):
    - Cleaned missing values (e.g., empty `total_charges`).
    - Feature Engineered new attributes (`tenure_group`, `total_additional_services`, `avg_monthly_charge`, `charge_difference`).
    - Applied One-Hot Encoding and Standard Scaling.
- [x] Wrote `ml/export_cleaned_data.py` and exported the processed data to `data/cleaned_customer_data.csv`.
- [x] Initialized Git repository and made the initial push to GitHub.

## 🚀 Current Status
- Docker containers (`db`, `api`, `dashboard`) are running.
- PostgreSQL database is seeded with 7,043 customer records.
- Feature engineering pipeline is complete and verified.

## 📝 Next Steps
- [x] **ML Modeling (Churn)**: Implement the XGBoost training script (`ml/churn_prediction.py`) to train on the cleaned dataset.
- [x] **ML Modeling (CLV)**: Implement the CLV prediction pipeline in `ml/clv_prediction.py`.
- [x] **Segmentation**: Implement K-Means clustering & RFM analysis in `ml/segmentation.py`.
- [x] **API Integration**: Load the trained models into the FastAPI routes so they can serve live predictions.
- [ ] **Dashboard Integration**: Connect the Streamlit dashboard to the live API endpoints to display predictions and visualizations.
