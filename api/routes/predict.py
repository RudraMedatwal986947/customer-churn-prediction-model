import os
import sys
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.connection import SessionLocal, engine
from ml.data_preprocessing import preprocess_data

router = APIRouter()

# Global variables to hold models (lazy loading)
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
churn_model = None
churn_scaler = None
clv_model = None
clv_scaler = None

def load_models():
    global churn_model, churn_scaler, clv_model, clv_scaler
    try:
        if churn_model is None:
            churn_model = joblib.load(os.path.join(MODELS_DIR, 'churn_xgboost_model.pkl'))
            churn_scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        if clv_model is None:
            clv_model = joblib.load(os.path.join(MODELS_DIR, 'clv_xgboost_model.pkl'))
            clv_scaler = joblib.load(os.path.join(MODELS_DIR, 'clv_scaler.pkl'))
    except Exception as e:
        print(f"Warning: Models not loaded. Train models first. Error: {e}")

class CustomerRequest(BaseModel):
    customer_id: str

def get_customer_features(customer_id: str):
    query = f"SELECT * FROM customers WHERE customer_id = '{customer_id}'"
    df = pd.read_sql(query, engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="Customer not found in database")
        
    # We need to preprocess this single row
    # The trick is that one-hot encoding requires all columns.
    # To fix this, we load the whole dataset, but wait, loading 7k rows for 1 prediction is slow.
    # A better way for production is to save the expected columns during training.
    # For now, since it's 7k rows, loading it takes ~50ms, which is acceptable for this prototype.
    all_customers = pd.read_sql("SELECT * FROM customers", engine)
    
    # We apply preprocessing to all to ensure all dummy columns are present, then extract our user
    X_all = preprocess_data(all_customers, is_training=False, scaler=churn_scaler)
    
    # Get the index of our customer
    customer_idx = all_customers.index[all_customers['customer_id'] == customer_id].tolist()
    if not customer_idx:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    # Extract just that row
    X_single = X_all.iloc[[customer_idx[0]]]
    return X_single

@router.post("/churn")
def predict_churn(req: CustomerRequest):
    load_models()
    if churn_model is None:
        raise HTTPException(status_code=500, detail="Churn model not trained yet")
        
    X_single = get_customer_features(req.customer_id)
    
    prediction = churn_model.predict(X_single)[0]
    probability = churn_model.predict_proba(X_single)[0][1]
    
    return {
        "customer_id": req.customer_id,
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_level": "High" if probability > 0.6 else ("Medium" if probability > 0.3 else "Low")
    }

@router.post("/clv")
def predict_clv(req: CustomerRequest):
    load_models()
    if clv_model is None:
        raise HTTPException(status_code=500, detail="CLV model not trained yet")
        
    # Re-use the same preprocessing logic but we must use the CLV scaler
    # CLV uses the same exact feature engineering steps as churn but a different scaler.
    all_customers = pd.read_sql("SELECT * FROM customers", engine)
    
    # CLV target removal and specific scaling
    all_customers['total_charges'] = pd.to_numeric(all_customers['total_charges'], errors='coerce').fillna(0)
    drop_cols = ['id', 'customer_id', 'created_at', 'churn', 'predicted_churn', 'predicted_clv', 'segment', 'total_charges']
    X_clv = all_customers.drop(columns=[col for col in drop_cols if col in all_customers.columns], errors='ignore')
    
    # Apply tenure mapping and services
    def map_tenure(tenure):
        if tenure <= 12: return '0_1_year'
        elif tenure <= 24: return '1_2_years'
        elif tenure <= 36: return '2_3_years'
        elif tenure <= 48: return '3_4_years'
        elif tenure <= 60: return '4_5_years'
        else: return '5_plus_years'
    X_clv['tenure_group'] = X_clv['tenure'].apply(map_tenure)
    
    services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    X_clv['total_additional_services'] = 0
    for service in services:
        if service in X_clv.columns:
            X_clv['total_additional_services'] += (X_clv[service] == 'Yes').astype(int)
            
    # Binary
    for col in ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']:
        if col in X_clv.columns:
            if col == 'gender': X_clv[col] = (X_clv[col] == 'Female').astype(int)
            else: X_clv[col] = (X_clv[col] == 'Yes').astype(int)
            
    # Dummies
    categorical = X_clv.select_dtypes(include=['object']).columns
    X_clv = pd.get_dummies(X_clv, columns=categorical, drop_first=True)
    
    # Scaling
    numerical = [c for c in ['tenure', 'monthly_charges', 'total_additional_services'] if c in X_clv.columns]
    X_clv[numerical] = clv_scaler.transform(X_clv[numerical])
    
    customer_idx = all_customers.index[all_customers['customer_id'] == req.customer_id].tolist()
    if not customer_idx:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    X_single = X_clv.iloc[[customer_idx[0]]]
    
    predicted_clv = clv_model.predict(X_single)[0]
    
    return {
        "customer_id": req.customer_id,
        "predicted_clv": float(predicted_clv)
    }
