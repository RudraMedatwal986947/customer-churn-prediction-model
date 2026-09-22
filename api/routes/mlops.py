"""
api/routes/mlops.py
REST Endpoints for MLOps Lifecycle Operations.
Provides endpoints for triggering automated retraining pipelines, inspecting
model governance and registry status, running statistical drift checks, and
retrieving real-time prediction audit logs.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.pipeline_orchestrator import run_retraining_pipeline, get_latest_pipeline_status
from ml.drift_monitor import run_drift_analysis, load_baseline_data, generate_synthetic_drift_data
from ml.mlops_config import setup_mlflow, CHURN_MODEL_REGISTRY_NAME, CLV_MODEL_REGISTRY_NAME

router = APIRouter(prefix="/api/v1/mlops", tags=["MLOps"])


class RetrainResponse(BaseModel):
    status: str
    gate_decision: str
    run_id: Optional[str] = None
    timestamp: str
    metrics: Dict[str, Any]
    gate_checks: Dict[str, bool]
    notes: str


class DriftCheckRequest(BaseModel):
    simulate_drift: bool = Field(default=False, description="Simulate a sudden inflation shock for demo purposes")
    sample_size: int = Field(default=500, description="Sample size to compare against baseline")


@router.post("/retrain", response_model=RetrainResponse)
def trigger_retraining():
    """
    Triggers the continuous training pipeline.
    Ingests latest customer data, trains a candidate Challenger model,
    evaluates it on a holdout benchmark, and gates promotion to Production.
    """
    try:
        report = run_retraining_pipeline()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining pipeline failed: {str(e)}")


@router.get("/status")
def get_mlops_status():
    """Returns active model governance metadata, latest pipeline run, and registry tags."""
    try:
        report = get_latest_pipeline_status()
        cfg = setup_mlflow()
        return {
            "mlflow_tracking_uri": cfg["tracking_uri"],
            "active_registered_model": CHURN_MODEL_REGISTRY_NAME,
            "latest_run": report,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve MLOps status: {str(e)}")


@router.post("/drift-check")
def check_drift(req: DriftCheckRequest = DriftCheckRequest()):
    """
    Executes Kolmogorov-Smirnov and PSI drift detection across core customer features.
    If simulate_drift is true, applies a synthetic shift to demonstrate alert behavior.
    """
    try:
        baseline = load_baseline_data()
        target = baseline.sample(min(req.sample_size, len(baseline)), random_state=42)

        if req.simulate_drift:
            target = generate_synthetic_drift_data(target, severity=0.35)

        drift_report = run_drift_analysis(baseline, target)
        return drift_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")


@router.get("/logs")
def get_prediction_logs(limit: int = 50):
    """Retrieves the most recent inference request audit logs from the database."""
    try:
        from database.connection import engine
        import pandas as pd
        query = f"SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql(query, engine)
        records = df.to_dict(orient="records")
        for rec in records:
            if isinstance(rec.get("timestamp"), (datetime, pd.Timestamp)):
                rec["timestamp"] = rec["timestamp"].isoformat()
        return {"count": len(records), "logs": records}
    except Exception as e:
        # Fallback if table is empty or database is in Excel-only fallback mode
        return {"count": 0, "logs": [], "note": f"Database logging offline or empty: {str(e)}"}
