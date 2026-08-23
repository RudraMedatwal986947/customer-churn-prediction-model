import sys
import os
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data_preprocessing import preprocess_data

@pytest.fixture
def sample_customer_data():
    """Returns a dummy DataFrame resembling raw database output."""
    return pd.DataFrame({
        "id": [1, 2],
        "customer_id": ["CUST1", "CUST2"],
        "gender": ["Female", "Male"],
        "tenure": [12, 24],
        "partner": ["Yes", "No"],
        "dependents": ["No", "Yes"],
        "phone_service": ["Yes", "Yes"],
        "paperless_billing": ["Yes", "No"],
        "monthly_charges": [50.0, 100.0],
        "total_charges": ["600.0", "2400.0"],
        "online_security": ["Yes", "No"],
        "online_backup": ["No", "Yes"],
        "device_protection": ["Yes", "No"],
        "tech_support": ["No", "Yes"],
        "streaming_tv": ["Yes", "No"],
        "streaming_movies": ["No", "Yes"],
        "contract": ["Month-to-month", "One year"],
        "churn": ["Yes", "No"],
        "predicted_churn": [None, None],
        "predicted_clv": [None, None],
        "segment": [None, None]
    })

def test_preprocess_data_training_mode(sample_customer_data):
    df_processed, y_churn, scaler = preprocess_data(sample_customer_data, is_training=True)

    # Assert columns to drop are gone
    assert "id" not in df_processed.columns
    assert "customer_id" not in df_processed.columns
    assert "predicted_churn" not in df_processed.columns

    # Assert total charges is numeric
    assert pd.api.types.is_numeric_dtype(df_processed["total_charges"])

    # Assert feature engineering columns exist
    assert "total_additional_services" in df_processed.columns
    assert "avg_monthly_charge" in df_processed.columns
    assert "charge_difference" in df_processed.columns

    # Assert target extraction
    assert list(y_churn) == [1, 0]  # Yes -> 1, No -> 0
    assert "churn" not in df_processed.columns

    # Assert encoding (gender: Female->1, Male->0)
    assert list(df_processed["gender"]) == [1, 0]

    # Assert scaling (scaler is returned and is a StandardScaler)
    assert isinstance(scaler, StandardScaler)

    # Assert the DataFrame has the expected shape (more cols than raw after OHE)
    assert df_processed.shape[0] == 2  # 2 rows preserved

def test_preprocess_data_inference_mode(sample_customer_data):
    # First get a scaler by running training mode
    _, _, scaler = preprocess_data(sample_customer_data.copy(), is_training=True)

    # Now run inference — should not raise
    df_processed = preprocess_data(sample_customer_data, is_training=False, scaler=scaler)

    assert "churn" not in df_processed.columns
    assert isinstance(df_processed, pd.DataFrame)
    assert df_processed.shape[0] == 2
