# Tasks In Progress

This document tracks the active development phases and pending tasks for the **Intelligent Customer Lifetime Value & Churn Prediction Platform**.

## 1. Data Ingestion & Preprocessing
* **Status**: ⏳ Pending Dataset
* **Tasks**:
  * Acquire and place the raw dataset (e.g., Telco Customer Churn dataset) into the `data/` directory.
  * Implement data cleaning logic in `ml/data_preprocessing.py` (handling missing values, encoding categorical variables, scaling numerical features).
  * Build the logic to ingest the cleaned data and populate the PostgreSQL database.

## 2. Machine Learning Model Development
* **Status**: 🛠️ In Development (Stubs created)
* **Tasks**:
  * **Segmentation (`ml/segmentation.py`)**: Implement RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to segment the customer base.
  * **Churn Prediction (`ml/churn_prediction.py`)**: Train and evaluate classification models (e.g., XGBoost, Random Forest) to predict the likelihood of customer churn.
  * **CLV Prediction (`ml/clv_prediction.py`)**: Train regression models to estimate the Customer Lifetime Value.
  * **Model Serialization**: Save trained models (e.g., as `.pkl` or `.joblib` files) so they can be loaded by the FastAPI backend.

## 3. Backend API Integration
* **Status**: 🛠️ In Development (Routes initialized)
* **Tasks**:
  * Load the serialized ML models into the FastAPI application state.
  * Complete the logic in `api/routes/predict.py` to accept customer data payloads, run inferences through the models, and return predictions.
  * Complete `api/routes/insights.py` to fetch aggregated data from the PostgreSQL database for the frontend dashboard.

## 4. Frontend Dashboard Completion
* **Status**: ✅ Completed
* **Tasks**:
  * **EDA Page (`dashboard/pages/1_eda.py`)**: Implemented data hooks to fetch and display interactive Plotly charts based on the database.
  * **Segmentation Page (`dashboard/pages/2_segmentation.py`)**: Visualized the K-Means clusters and RFM segments with scatter plots and box plots.
  * **Predictions Page (`dashboard/pages/3_predictions.py`)**: Created forms to input new customer details and wired them up to the FastAPI `/predict` endpoints to display real-time Churn and CLV results.

## 5. Testing
* **Status**: ✅ Completed
* **Tasks**:
  * Wrote Pytest unit tests for the ML pipeline and FastAPI routes using `pytest`.

## 6. Deployment
* **Status**: ⏳ Not Started
* **Tasks**:
  * Spin up the entire stack using `docker-compose up` to verify end-to-end integration (Database -> FastAPI -> Streamlit).
