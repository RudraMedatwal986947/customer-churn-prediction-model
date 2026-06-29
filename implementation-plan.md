# Intelligent Customer Lifetime Value & Churn Prediction Platform

An End-to-End Machine Learning Platform for Customer Segmentation, Customer Lifetime Value Prediction, Churn Prediction, and Business Intelligence based on the defined project scope.

## User Review Required

> [!IMPORTANT]
> The initial implementation phase involves setting up the data infrastructure (PostgreSQL) and starting the ML pipeline. Please review the proposed directory structure and technology stack (FastAPI, Streamlit, PostgreSQL) for alignment with your exact requirements.

## Open Questions

> [!NOTE]
> 1. Do you already have a dataset ready for this project, or should we use a standard mock dataset (like the Telco Customer Churn dataset from Kaggle or similar) to start development?
> 2. Are there any specific machine learning frameworks (e.g., scikit-learn, XGBoost, PyTorch) you prefer we prioritize?
> 3. Do you have a preferred cloud provider (AWS, GCP, Azure) for the deployment phase using Docker?

## Proposed Changes

We will construct a new project directory structure and set up the foundation for the analytics platform, encompassing data ingestion, ML pipelines, API endpoints, and a front-end dashboard.

### Infrastructure & Configuration

#### [NEW] docker-compose.yml
- Sets up the PostgreSQL database, FastAPI backend, and Streamlit frontend.

#### [NEW] requirements.txt
- Python dependencies (pandas, scikit-learn, fastapi, uvicorn, streamlit, psycopg2-binary, sqlalchemy, etc.).

#### [NEW] README.md
- Project setup instructions and overview.

### Database & Models

#### [NEW] database/connection.py
- PostgreSQL connection setup using SQLAlchemy.

#### [NEW] database/models.py
- SQLAlchemy ORM models for storing customer demographics, transactions, and behavior data.

### Machine Learning Pipeline

#### [NEW] ml/data_preprocessing.py
- Scripts for handling missing values, encoding, and feature engineering.

#### [NEW] ml/segmentation.py
- Implementation of RFM analysis and K-Means clustering.

#### [NEW] ml/churn_prediction.py
- Classification model training and evaluation script.

#### [NEW] ml/clv_prediction.py
- Regression model for Customer Lifetime Value estimation.

### Backend API (FastAPI)

#### [NEW] api/main.py
- FastAPI application entry point.

#### [NEW] api/routes/predict.py
- REST API endpoints to serve churn and CLV predictions.

#### [NEW] api/routes/insights.py
- REST API endpoints to serve segmentation and EDA insights.

### Frontend Dashboard (Streamlit)

#### [NEW] dashboard/app.py
- Streamlit application entry point and sidebar navigation.

#### [NEW] dashboard/pages/1_eda.py
- Page for Exploratory Data Analysis visualizations.

#### [NEW] dashboard/pages/2_segmentation.py
- Page displaying RFM and K-Means customer segments.

#### [NEW] dashboard/pages/3_predictions.py
- Page for interacting with the churn and CLV prediction API.

## Verification Plan

### Automated Tests
- Run `pytest` on the ML pipeline to ensure data preprocessing and model training complete without errors.
- Test FastAPI endpoints using `pytest` and `httpx` to verify responses.

### Manual Verification
- Launch the `docker-compose` stack locally.
- Verify PostgreSQL database initialization.
- Open the Streamlit dashboard on `http://localhost:8501` to ensure all pages render correctly.
- Test API endpoints via the FastAPI Swagger UI on `http://localhost:8000/docs`.
