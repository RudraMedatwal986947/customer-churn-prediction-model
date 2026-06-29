# Intelligent Customer Lifetime Value & Churn Prediction Platform

An End-to-End Machine Learning Platform for Customer Segmentation, Customer Lifetime Value Prediction, Churn Prediction, and Business Intelligence based on the defined project scope.

## Platform Architecture
The system consists of three Dockerized containers:
1. **PostgreSQL Database**: Stores customer attributes, transaction logs, and dynamically updated model predictions/segments.
2. **FastAPI Backend**: Hosts the trained XGBoost and K-Means models and provides RESTful endpoints for real-time predictions and data aggregation.
3. **Streamlit Frontend**: An interactive dashboard for business stakeholders to explore EDA, segmentation clusters, and run live churn/CLV predictions.

## Implementation Status

### Infrastructure & Configuration
- `docker-compose.yml` **[IMPLEMENTED]**: Orchestrates the DB, API, and Dashboard with live-reload volume mounts.
- `requirements.txt` **[IMPLEMENTED]**: Contains all python dependencies (pandas, scikit-learn, fastapi, streamlit, xgboost, etc.).
- `.gitignore` **[IMPLEMENTED]**: Configured to ignore `data/`, `__pycache__`, and `*.pkl` models.

### Database & Models
- `database/connection.py` **[IMPLEMENTED]**: PostgreSQL connection setup using SQLAlchemy.
- `database/models.py` **[IMPLEMENTED]**: SQLAlchemy ORM mapping for the `Customer` table.
- `database/seed_db.py` **[IMPLEMENTED]**: Ingests and formats the Kaggle Telco Churn dataset (`Telco_customer_churn.xlsx`).

### Machine Learning Pipeline
- `ml/data_preprocessing.py` **[IMPLEMENTED]**: Robust pipeline that handles missing values, engineers new features (`tenure_group`, `avg_monthly_charge`), and performs One-Hot Encoding and Scaling.
- `ml/segmentation.py` **[IMPLEMENTED]**: Extracts features, trains a K-Means clustering model (4 clusters), and pushes segments back to the DB.
- `ml/churn_prediction.py` **[IMPLEMENTED]**: Trains an XGBoost Classifier (80% Accuracy). Serializes the `.pkl` model and scaler.
- `ml/clv_prediction.py` **[IMPLEMENTED]**: Trains an XGBoost Regressor (0.99 R²) for lifetime value estimation.
- `ml/generate_plots.py` **[IMPLEMENTED]**: Generates static PNGs for ROC, Feature Importance, and Segmentation.

### Backend API (FastAPI)
- `api/main.py` **[IMPLEMENTED]**: Application entry point and router aggregator.
- `api/routes/predict.py` **[IMPLEMENTED]**: REST API endpoints (`/churn`, `/clv`) that load `.pkl` models and run live inference for a given `customer_id`.
- `api/routes/insights.py` **[IMPLEMENTED]**: REST API endpoints returning segmentation aggregate stats directly from SQL.

### Frontend Dashboard (Streamlit)
- `dashboard/app.py` **[PENDING]**: Streamlit application entry point and sidebar navigation.
- `dashboard/pages/1_eda.py` **[PENDING]**: Page for Exploratory Data Analysis visualizations.
- `dashboard/pages/2_segmentation.py` **[PENDING]**: Page displaying K-Means customer segments.
- `dashboard/pages/3_predictions.py` **[PENDING]**: Interactive page for querying the churn and CLV prediction APIs.

## Next Phase: Frontend Integration
The immediate next step is to build out the Streamlit dashboard components to make HTTP calls to the FastAPI endpoints (`http://api:8000`) and visually present the insights and predictions to the end-user.
