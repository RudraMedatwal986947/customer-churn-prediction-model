# Customer Churn and CLV Prediction Platform

This is an end-to-end Machine Learning platform for customer segmentation, Customer Lifetime Value (CLV) prediction, churn prediction, and business intelligence.

## Tech Stack
- **Database**: PostgreSQL
- **Backend API**: FastAPI
- **Frontend Dashboard**: Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas
- **Deployment**: Docker, Docker Compose

## Project Structure
- `data/`: Contains the raw and processed datasets (e.g., from Kaggle).
- `database/`: SQLAlchemy ORM models and connection settings.
- `ml/`: Machine learning pipeline scripts (preprocessing, training, evaluation).
- `api/`: FastAPI application and route definitions.
- `dashboard/`: Streamlit dashboard pages.

## Setup Instructions

### 1. Prerequisites
- Docker and Docker Compose installed on your system.

### 2. Dataset
Place your Kaggle dataset in the `data/` directory (e.g., `data/customer_churn.csv`).

### 3. Running the Platform
You can start the entire platform using Docker Compose:

```bash
docker-compose up --build
```

- **Dashboard**: `http://localhost:8501`
- **API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **Database**: `localhost:5432`
