"""
tests/test_mlops.py
Automated Quality Assurance for MLOps Architecture.
Validates MLflow configuration, statistical drift detection routines,
Champion-Challenger promotion gate rules, and FastAPI MLOps routes.
"""

import pytest
import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from ml.mlops_config import (
    setup_mlflow,
    CHURN_EXPERIMENT_NAME,
    get_champion_gate_rules,
)
from ml.drift_monitor import (
    calculate_psi,
    evaluate_numerical_feature,
    evaluate_categorical_feature,
    run_drift_analysis,
    generate_synthetic_drift_data,
)
from ml.pipeline_orchestrator import evaluate_champion_gate, get_latest_pipeline_status
from api.main import app


def test_mlflow_setup_initialization():
    """Verify MLflow setup correctly initializes the SQLite tracking backend."""
    cfg = setup_mlflow()
    assert "tracking_uri" in cfg
    assert "sqlite:///" in cfg["tracking_uri"]
    assert cfg["churn_experiment"] == CHURN_EXPERIMENT_NAME


def test_drift_monitor_stable_distribution():
    """Verify statistical drift tests confirm stability when comparing matching distributions."""
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(50, 10, 500), name="monthly_charges")
    s2 = pd.Series(rng.normal(50, 10, 500), name="monthly_charges")

    diag = evaluate_numerical_feature(s1, s2, "monthly_charges")
    assert diag["drift_detected"] is False
    assert diag["p_value"] > 0.05
    assert diag["severity"] in ["None", "Low"]


def test_drift_monitor_detects_distribution_shift():
    """Verify statistical drift tests correctly identify severe distribution shifts."""
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(30, 5, 500), name="tenure")
    s2 = pd.Series(rng.normal(70, 5, 500), name="tenure")  # Major shift in mean

    diag = evaluate_numerical_feature(s1, s2, "tenure")
    assert diag["drift_detected"] is True
    assert diag["p_value"] < 0.01
    assert diag["severity"] in ["Moderate", "High"]


def test_psi_calculation_consistency():
    """Verify Population Stability Index produces 0.0 for identical series and >0.2 for shifted."""
    series = pd.Series([10, 20, 30, 40, 50] * 100)
    identical_psi = calculate_psi(series, series)
    assert identical_psi == pytest.approx(0.0, abs=1e-3)

    shifted = pd.Series([80, 90, 100, 110, 120] * 100)
    shifted_psi = calculate_psi(series, shifted)
    assert shifted_psi > 0.20


def test_champion_gate_rules_pass_and_fail():
    """Verify the Champion-Challenger gate correctly admits or rejects candidate models."""
    rules = get_champion_gate_rules()

    # Case 1: Candidate exceeds all thresholds
    passing_metrics = {
        "accuracy": 0.935,
        "roc_auc": 0.982,
        "recall": 0.885,
        "fpr": 0.035
    }
    passed, checks = evaluate_champion_gate(passing_metrics)
    assert passed is True
    assert all(checks.values())

    # Case 2: Candidate with degraded recall fails the gate
    failing_metrics = {
        "accuracy": 0.925,
        "roc_auc": 0.960,
        "recall": 0.720,  # Below 0.850 minimum requirement
        "fpr": 0.040
    }
    passed_fail, checks_fail = evaluate_champion_gate(failing_metrics)
    assert passed_fail is False
    assert checks_fail["recall_check"] is False


def test_api_mlops_endpoints():
    """Verify FastAPI MLOps routes respond with valid payloads."""
    client = TestClient(app)

    # Test status endpoint
    res_status = client.get("/api/v1/mlops/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["status"] == "operational"
    assert "active_registered_model" in data

    # Test drift-check endpoint
    res_drift = client.post("/api/v1/mlops/drift-check", json={"simulate_drift": False, "sample_size": 200})
    assert res_drift.status_code == 200
    drift_data = res_drift.json()
    assert "overall_drift_detected" in drift_data
    assert "features" in drift_data


def test_batch_inference_scoring_and_metrics():
    """Verify batch inference engine computes calibrated probabilities, CLV, and cohort KPIs."""
    from ml.batch_inference import score_customer_dataframe, generate_sample_customer_record

    rec_high = generate_sample_customer_record("high_risk")
    rec_low = generate_sample_customer_record("low_risk")
    df = pd.DataFrame([rec_high, rec_low])

    scored_df, metrics = score_customer_dataframe(df)

    assert len(scored_df) == 2
    assert "churn_probability" in scored_df.columns
    assert "predicted_churn" in scored_df.columns
    assert "predicted_clv" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    assert "clv_tier" in scored_df.columns

    # High risk profile assertions
    assert scored_df.iloc[0]["risk_tier"] in ["High", "Medium"]
    assert scored_df.iloc[0]["churn_probability"] >= 0.30
    assert scored_df.iloc[0]["predicted_clv"] >= 0.0

    # Low risk profile assertions
    assert scored_df.iloc[1]["risk_tier"] in ["Low", "Medium"]
    assert scored_df.iloc[1]["churn_probability"] < 0.60

    # Metrics assertions
    assert metrics["total_customers"] == 2
    assert 0.0 <= metrics["churn_rate_pct"] <= 100.0
    assert metrics["total_projected_clv"] >= 0.0
    assert "decision_threshold" in metrics


def test_custom_data_store_persistence_lifecycle():
    """Verify customer records are persisted, loaded in combined cohorts, and cleared correctly."""
    from ml.batch_inference import generate_sample_customer_record, score_customer_dataframe
    from ml.custom_data_store import (
        save_manual_cohort,
        load_manual_customers,
        load_all_combined_customers,
        clear_manual_cohort
    )

    # 1. Clear any leftover custom records
    clear_manual_cohort()

    # 2. Score and save a new custom customer
    sample = generate_sample_customer_record("high_risk")
    sample["customer_id"] = "TEST-PERSIST-001"
    raw_df = pd.DataFrame([sample])
    scored_df, _ = score_customer_dataframe(raw_df)

    count_saved = save_manual_cohort(scored_df, batch_id="unit_test")
    assert count_saved == 1

    # 3. Retrieve through load_manual_customers
    loaded = load_manual_customers()
    assert not loaded.empty
    assert "TEST-PERSIST-001" in loaded["customer_id"].astype(str).values

    # 4. Retrieve through load_all_combined_customers
    combined, src = load_all_combined_customers()
    assert not combined.empty
    assert "is_custom" in combined.columns
    custom_rows = combined[combined["is_custom"]]
    assert len(custom_rows) >= 1
    assert "TEST-PERSIST-001" in custom_rows["customer_id"].astype(str).values

    # 5. Clean up
    cleared = clear_manual_cohort()
    assert cleared >= 1
    after_clear = load_manual_customers()
    assert after_clear.empty or "TEST-PERSIST-001" not in after_clear["customer_id"].astype(str).values


def test_api_custom_customer_endpoints():
    """Verify REST API routes for saving, batch ingesting, fetching, and clearing custom customers."""
    from ml.batch_inference import generate_sample_customer_record
    client = TestClient(app)

    # 1. Save single customer endpoint
    sample = generate_sample_customer_record("high_risk")
    sample["customer_id"] = "API-TEST-999"
    res_save = client.post("/api/v1/mlops/save-customer", json={"customer": sample, "batch_id": "test_api"})
    assert res_save.status_code == 200
    save_data = res_save.json()
    assert save_data["status"] == "success"
    assert save_data["customer_id"] == "API-TEST-999"
    assert "record" in save_data
    assert "metrics" in save_data

    # 2. Get custom customers endpoint
    res_get = client.get("/api/v1/mlops/custom-customers")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["count"] >= 1
    cust_ids = [c["customer_id"] for c in get_data["customers"]]
    assert "API-TEST-999" in cust_ids

    # 3. Batch ingest endpoint
    rec2 = generate_sample_customer_record("low_risk")
    rec2["customer_id"] = "API-TEST-888"
    res_batch = client.post("/api/v1/mlops/ingest-batch", json={"records": [rec2], "batch_id": "test_batch"})
    assert res_batch.status_code == 200
    batch_data = res_batch.json()
    assert batch_data["count"] == 1
    assert "metrics" in batch_data

    # 4. Clear custom customers endpoint
    res_del = client.delete("/api/v1/mlops/custom-customers")
    assert res_del.status_code == 200
    del_data = res_del.json()
    assert del_data["status"] == "success"
