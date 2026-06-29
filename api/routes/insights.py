import os
import sys
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.connection import engine

router = APIRouter()

@router.get("/segmentation/summary")
def get_segmentation_summary():
    """Returns average stats per segment to understand the clusters"""
    query = """
    SELECT segment, 
           COUNT(*) as count, 
           AVG(tenure) as avg_tenure, 
           AVG(CAST(monthly_charges AS FLOAT)) as avg_monthly_charges
    FROM customers
    WHERE segment IS NOT NULL
    GROUP BY segment
    """
    try:
        df = pd.read_sql(query, engine)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CustomerRequest(BaseModel):
    customer_id: str

@router.post("/segmentation/customer")
def get_customer_segment(req: CustomerRequest):
    """Returns the segment for a specific customer"""
    query = f"SELECT segment FROM customers WHERE customer_id = '{req.customer_id}'"
    df = pd.read_sql(query, engine)
    
    if df.empty or pd.isna(df.iloc[0]['segment']):
        raise HTTPException(status_code=404, detail="Customer or segment not found")
        
    return {
        "customer_id": req.customer_id,
        "segment": df.iloc[0]['segment']
    }
