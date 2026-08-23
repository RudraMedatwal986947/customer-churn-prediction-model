import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Customer Churn & CLV Prediction API"}

@patch("api.routes.insights.pd.read_sql")
def test_get_segmentation_summary(mock_read_sql):
    import pandas as pd
    # Mock database return value
    mock_df = pd.DataFrame({
        "segment": ["Cluster 0", "Cluster 1"],
        "count": [10, 20],
        "avg_tenure": [15.5, 30.2],
        "avg_monthly_charges": [55.0, 90.0]
    })
    mock_read_sql.return_value = mock_df
    
    response = client.get("/api/v1/insights/segmentation/summary")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["segment"] == "Cluster 0"
    assert data[0]["count"] == 10

@patch("api.routes.insights.pd.read_sql")
def test_get_customer_segment(mock_read_sql):
    import pandas as pd
    mock_df = pd.DataFrame({"segment": ["Cluster 2"]})
    mock_read_sql.return_value = mock_df
    
    response = client.post("/api/v1/insights/segmentation/customer", json={"customer_id": "CUST123"})
    assert response.status_code == 200
    assert response.json() == {"customer_id": "CUST123", "segment": "Cluster 2"}
    
@patch("api.routes.insights.pd.read_sql")
def test_get_customer_segment_not_found(mock_read_sql):
    import pandas as pd
    mock_df = pd.DataFrame()
    mock_read_sql.return_value = mock_df
    
    response = client.post("/api/v1/insights/segmentation/customer", json={"customer_id": "UNKNOWN"})
    assert response.status_code == 404

@patch("api.routes.predict.get_customer_features")
@patch("api.routes.predict.joblib.load")
def test_predict_churn(mock_joblib_load, mock_get_features):
    """
    Tests the /predict/churn route by mocking both model loading and feature
    extraction, so we validate the route handler in isolation.
    """
    import pandas as pd

    # Build a mock model that returns a churn prediction
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.1, 0.9]]

    # Build a mock scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x  # identity transform

    def joblib_side_effect(path):
        if 'scaler' in path:
            return mock_scaler
        return mock_model

    mock_joblib_load.side_effect = joblib_side_effect

    # Mock get_customer_features to return a minimal 1-row DataFrame
    feature_cols = ["tenure", "monthly_charges", "total_charges",
                    "total_additional_services", "avg_monthly_charge", "charge_difference"]
    mock_feature_df = pd.DataFrame([[0.5] * len(feature_cols)], columns=feature_cols)
    mock_get_features.return_value = mock_feature_df

    response = client.post("/api/v1/predict/churn", json={"customer_id": "CUST1"})

    # The route should return 200 with a valid churn_prediction field
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in [0, 1]
    assert "churn_probability" in data
    assert "risk_level" in data
