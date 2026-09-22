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
