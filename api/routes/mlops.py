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
from ml.batch_inference import score_customer_dataframe, normalize_customer_input
from ml.custom_data_store import save_manual_cohort, load_manual_customers, clear_manual_cohort
import pandas as pd

router = APIRouter(prefix="/api/v1/mlops", tags=["MLOps"])


class BatchIngestRequest(BaseModel):
    records: List[Dict[str, Any]]
    batch_id: Optional[str] = None


class SingleCustomerRequest(BaseModel):
    customer: Dict[str, Any]
    batch_id: Optional[str] = "manual"


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


@router.post("/save-customer")
def save_single_customer(req: SingleCustomerRequest):
    """
    Ingests and scores a single customer record entered manually.
    Persists the scored record into PostgreSQL and local storage.
    """
    try:
        raw_df = pd.DataFrame([req.customer])
        scored_df, metrics = score_customer_dataframe(raw_df)
        if scored_df.empty:
            raise HTTPException(status_code=400, detail="Invalid customer data provided.")
        save_manual_cohort(scored_df, batch_id=req.batch_id or "manual")
        record = scored_df.to_dict(orient="records")[0]
        for k, v in record.items():
            if hasattr(v, "item"):
                record[k] = v.item()
        return {
            "status": "success",
            "message": "Customer record scored and persisted successfully.",
            "customer_id": record.get("customer_id"),
            "record": record,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to score and save customer: {str(e)}")


@router.post("/ingest-batch")
def ingest_batch_cohort(req: BatchIngestRequest):
    """
    Ingests, scores, and persists a batch of customer records entered manually.
    Generates cohort analytics metrics including churn rate, CLV, and risk distributions.
    """
    try:
        if not req.records:
            raise HTTPException(status_code=400, detail="No customer records provided.")
        raw_df = pd.DataFrame(req.records)
        scored_df, metrics = score_customer_dataframe(raw_df)
        save_manual_cohort(scored_df, batch_id=req.batch_id or "batch")
        records = scored_df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if hasattr(v, "item"):
                    rec[k] = v.item()
        return {
            "status": "success",
            "count": len(records),
            "metrics": metrics,
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest cohort batch: {str(e)}")


@router.get("/custom-customers")
def get_custom_customers():
    """
    Retrieves all manually entered customer records stored in PostgreSQL or CSV fallback.
    """
    try:
        df = load_manual_customers()
        records = df.to_dict(orient="records")
        for rec in records:
            if isinstance(rec.get("created_at"), (datetime, pd.Timestamp)):
                rec["created_at"] = rec["created_at"].isoformat()
            for k, v in rec.items():
                if hasattr(v, "item"):
                    rec[k] = v.item()
        return {
            "count": len(records),
            "customers": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load custom customers: {str(e)}")


@router.delete("/custom-customers")
def clear_custom_cohort():
    """
    Clears all manually entered customer records from PostgreSQL and CSV fallback.
    """
    try:
        cleared_count = clear_manual_cohort()
        return {
            "status": "success",
            "message": f"Successfully cleared {cleared_count} custom customer records.",
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear custom cohort: {str(e)}")
