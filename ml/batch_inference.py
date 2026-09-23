"""
ml/batch_inference.py
Batch and Manual Customer Inference Engine.
Processes individual or cohort DataFrames through the trained XGBoost Churn Classifier
and XGBoost CLV Regressor, generating predictions, probability distributions,
risk tiers, and cohort-level summary metrics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import uuid

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

DEFAULT_CHURN_THRESHOLD = 0.440


def get_model_artifacts():
    """Loads trained churn model, clv model, scalers, and decision threshold."""
    churn_model = joblib.load(os.path.join(MODELS_DIR, 'churn_xgboost_model.pkl'))
    churn_scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    clv_model = joblib.load(os.path.join(MODELS_DIR, 'clv_xgboost_model.pkl'))
    clv_scaler = joblib.load(os.path.join(MODELS_DIR, 'clv_scaler.pkl'))

    threshold = DEFAULT_CHURN_THRESHOLD
    thresh_path = os.path.join(MODELS_DIR, 'churn_threshold.json')
    if os.path.exists(thresh_path):
        try:
            with open(thresh_path, 'r') as f:
                threshold = float(json.load(f).get('threshold', DEFAULT_CHURN_THRESHOLD))
        except Exception:
            threshold = DEFAULT_CHURN_THRESHOLD

    return churn_model, churn_scaler, clv_model, clv_scaler, threshold


def normalize_customer_input(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names, applies standard defaults, and ensures required fields exist."""
    df = df.copy()

    # Normalize column names: strip, lower, replace spaces/hyphens with underscores
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('-', '_', regex=False)
    )

    alias_map = {
        'customerid': 'customer_id',
        'cust_id': 'customer_id',
        'tenure_months': 'tenure',
        'churn_label': 'churn',
        'monthly_charge': 'monthly_charges',
        'total_charge': 'total_charges',
    }
    df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)

    # Ensure customer_id exists
    if 'customer_id' not in df.columns or df['customer_id'].isnull().all():
        df['customer_id'] = [f"MANUAL-{uuid.uuid4().hex[:6].upper()}" for _ in range(len(df))]
    else:
        df['customer_id'] = df['customer_id'].fillna("").apply(
            lambda cid: f"MANUAL-{uuid.uuid4().hex[:6].upper()}" if not str(cid).strip() else str(cid).strip()
        )

    # Defaults for categorical and numerical features
    defaults = {
        'gender': 'Male',
        'senior_citizen': 0,
        'partner': 'No',
        'dependents': 'No',
        'tenure': 1,
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'DSL',
        'online_security': 'No',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'contract': 'Month-to-month',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check',
        'monthly_charges': 65.0,
        'churn_score': 50.0,
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    # Total charges default
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(1).astype(int)
    df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce').fillna(65.0).astype(float)
    if 'total_charges' not in df.columns:
        df['total_charges'] = df['tenure'] * df['monthly_charges']
    else:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(
            df['tenure'] * df['monthly_charges']
        ).astype(float)

    if 'churn_score' in df.columns:
        df['churn_score'] = pd.to_numeric(df['churn_score'], errors='coerce').fillna(50.0).astype(float)

    return df


def prepare_churn_features(df: pd.DataFrame, churn_scaler, expected_features: List[str]) -> pd.DataFrame:
    """Performs feature engineering and scaling for the churn model."""
    df_feat = df.copy()

    # Tenure mapping
    def map_tenure(t):
        if t <= 12: return '0_1_year'
        elif t <= 24: return '1_2_years'
        elif t <= 36: return '2_3_years'
        elif t <= 48: return '3_4_years'
        elif t <= 60: return '4_5_years'
        else: return '5_plus_years'
    df_feat['tenure_group'] = df_feat['tenure'].apply(map_tenure)

    # Additional services count
    services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    df_feat['total_additional_services'] = sum((df_feat[s] == 'Yes').astype(int) for s in services if s in df_feat.columns)

    # Financial interaction features
    df_feat['avg_monthly_charge'] = df_feat['total_charges'] / (df_feat['tenure'] + 1)
    df_feat['charge_difference'] = df_feat['monthly_charges'] - df_feat['avg_monthly_charge']

    # Contract risk score
    contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
    df_feat['contract_risk_score'] = df_feat['contract'].map(contract_risk).fillna(1).astype(int)

    # Senior with no contract
    is_senior = df_feat['senior_citizen'].astype(str).isin(['1', 'Yes', 'True', '1.0'])
    is_mtm = df_feat['contract'] == 'Month-to-month'
    df_feat['senior_no_contract'] = (is_senior & is_mtm).astype(int)

    # High charge with no support
    high_threshold = df_feat['monthly_charges'].median() if len(df_feat) > 1 else 65.0
    no_security = df_feat.get('online_security', pd.Series('No', index=df_feat.index)) == 'No'
    no_support = df_feat.get('tech_support', pd.Series('No', index=df_feat.index)) == 'No'
    df_feat['high_charge_no_support'] = ((df_feat['monthly_charges'] > high_threshold) & no_security & no_support).astype(int)

    # Payment risk score
    payment_risk = {
        'Electronic check': 2,
        'Mailed check': 1,
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 0,
    }
    df_feat['payment_risk_score'] = df_feat['payment_method'].map(payment_risk).fillna(1).astype(int)

    # Tenure x charges interaction
    df_feat['tenure_x_charges'] = df_feat['tenure'] * df_feat['monthly_charges']

    # Drop non-feature columns
    drop_cols = [
        'customer_id', 'id', 'churn', 'predicted_churn', 'predicted_clv', 'segment',
        'created_at', 'churn_probability', 'risk_tier', 'clv_tier', 'batch_id'
    ]
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors='ignore')

    # Binary mappings
    binary_map = {
        'gender': lambda s: (s == 'Female').astype(int),
        'senior_citizen': lambda s: s.map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int),
        'partner': lambda s: (s == 'Yes').astype(int),
        'dependents': lambda s: (s == 'Yes').astype(int),
        'phone_service': lambda s: (s == 'Yes').astype(int),
        'paperless_billing': lambda s: (s == 'Yes').astype(int),
    }
    for col, fn in binary_map.items():
        if col in df_feat.columns:
            df_feat[col] = fn(df_feat[col])

    # One-hot encoding for multi-class variables
    cat_cols = df_feat.select_dtypes(include='object').columns.tolist()
    df_feat = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)
    df_feat.columns = [c.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for c in df_feat.columns]

    # Scaling numerical features
    num_cols = [
        'tenure', 'monthly_charges', 'total_charges',
        'total_additional_services', 'avg_monthly_charge', 'charge_difference',
        'contract_risk_score', 'tenure_x_charges', 'churn_score',
    ]
    scale_cols = [c for c in num_cols if c in df_feat.columns]
    if hasattr(churn_scaler, 'transform') and scale_cols:
        try:
            df_feat[scale_cols] = churn_scaler.transform(df_feat[scale_cols])
        except Exception:
            pass

    # Exact column alignment against trained model
    df_feat = df_feat.reindex(columns=expected_features, fill_value=0)
    return df_feat


def prepare_clv_features(df: pd.DataFrame, clv_scaler, expected_features: List[str]) -> pd.DataFrame:
    """Performs feature engineering and scaling for the CLV regression model."""
    df_clv = df.copy()

    # Tenure mapping
    def map_tenure(t):
        if t <= 12: return '0_1_year'
        elif t <= 24: return '1_2_years'
        elif t <= 36: return '2_3_years'
        elif t <= 48: return '3_4_years'
        elif t <= 60: return '4_5_years'
        else: return '5_plus_years'
    df_clv['tenure_group'] = df_clv['tenure'].apply(map_tenure)

    # Additional services
    services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    df_clv['total_additional_services'] = sum((df_clv[s] == 'Yes').astype(int) for s in services if s in df_clv.columns)

    # Drop non-feature columns
    drop_cols = [
        'id', 'customer_id', 'created_at', 'churn', 'predicted_churn', 'predicted_clv',
        'segment', 'total_charges', 'churn_score', 'churn_probability', 'risk_tier',
        'clv_tier', 'batch_id'
    ]
    df_clv = df_clv.drop(columns=[c for c in drop_cols if c in df_clv.columns], errors='ignore')

    # Binary mappings
    for col in ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']:
        if col in df_clv.columns:
            if col == 'gender':
                df_clv[col] = (df_clv[col] == 'Female').astype(int)
            else:
                df_clv[col] = (df_clv[col] == 'Yes').astype(int)

    # One-hot encoding
    cat_cols = df_clv.select_dtypes(include='object').columns.tolist()
    df_clv = pd.get_dummies(df_clv, columns=cat_cols, drop_first=True)

    # Scaling
    numerical = [c for c in ['tenure', 'monthly_charges', 'total_additional_services'] if c in df_clv.columns]
    if hasattr(clv_scaler, 'transform') and numerical:
        try:
            df_clv[numerical] = clv_scaler.transform(df_clv[numerical])
        except Exception:
            pass

    # Exact column alignment against trained model
    df_clv = df_clv.reindex(columns=expected_features, fill_value=0)
    return df_clv


def score_customer_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes full batch scoring on the provided customer DataFrame.
    Returns:
        scored_df: DataFrame with original attributes plus churn probability, prediction, CLV, and tiers.
        metrics: Summary dictionary of cohort-level KPIs.
    """
    if df.empty:
        return df, {}

    churn_model, churn_scaler, clv_model, clv_scaler, threshold = get_model_artifacts()

    # Step 1: Normalize input
    clean_df = normalize_customer_input(df)

    # Step 2: Churn Inference
    expected_churn_cols = list(churn_model.feature_names_in_)
    X_churn = prepare_churn_features(clean_df, churn_scaler, expected_churn_cols)
    churn_probabilities = churn_model.predict_proba(X_churn)[:, 1]
    predicted_churn = (churn_probabilities >= threshold).astype(int)

    # Step 3: CLV Inference
    expected_clv_cols = list(clv_model.feature_names_in_)
    X_clv = prepare_clv_features(clean_df, clv_scaler, expected_clv_cols)
    predicted_clv = clv_model.predict(X_clv)
    predicted_clv = np.maximum(0.0, predicted_clv)

    # Step 4: Tier assignments
    risk_tiers = []
    for prob in churn_probabilities:
        if prob > 0.60:
            risk_tiers.append("High")
        elif prob >= 0.30:
            risk_tiers.append("Medium")
        else:
            risk_tiers.append("Low")

    clv_tiers = []
    for clv in predicted_clv:
        if clv >= 3000.0:
            clv_tiers.append("High Yield")
        elif clv >= 1000.0:
            clv_tiers.append("Core Value")
        else:
            clv_tiers.append("Low Yield")

    # Step 5: Construct output DataFrame
    scored_df = clean_df.copy()
    scored_df['churn_probability'] = np.round(churn_probabilities, 4)
    scored_df['predicted_churn'] = predicted_churn
    scored_df['predicted_clv'] = np.round(predicted_clv, 2)
    scored_df['risk_tier'] = risk_tiers
    scored_df['clv_tier'] = clv_tiers

    # Step 6: Summary Metrics
    total_count = len(scored_df)
    churn_count = int(predicted_churn.sum())
    churn_rate = round((churn_count / total_count) * 100.0, 2) if total_count > 0 else 0.0
    avg_prob = round(float(np.mean(churn_probabilities)) * 100.0, 2) if total_count > 0 else 0.0
    total_clv = round(float(np.sum(predicted_clv)), 2)
    avg_clv = round(float(np.mean(predicted_clv)), 2) if total_count > 0 else 0.0

    high_risk_count = int((scored_df['risk_tier'] == 'High').sum())
    medium_risk_count = int((scored_df['risk_tier'] == 'Medium').sum())
    low_risk_count = int((scored_df['risk_tier'] == 'Low').sum())

    # High value at risk (CLV >= 1000 and High Risk)
    high_val_at_risk = int(((scored_df['predicted_clv'] >= 1000.0) & (scored_df['risk_tier'] == 'High')).sum())
    at_risk_revenue = round(float(scored_df.loc[scored_df['risk_tier'] == 'High', 'predicted_clv'].sum()), 2)

    cohort_metrics = {
        'total_customers': total_count,
        'churn_count': churn_count,
        'churn_rate_pct': churn_rate,
        'avg_churn_prob_pct': avg_prob,
        'total_projected_clv': total_clv,
        'avg_projected_clv': avg_clv,
        'high_risk_count': high_risk_count,
        'medium_risk_count': medium_risk_count,
        'low_risk_count': low_risk_count,
        'high_value_at_risk_count': high_val_at_risk,
        'at_risk_revenue': at_risk_revenue,
        'decision_threshold': threshold,
    }

    return scored_df, cohort_metrics


def generate_sample_customer_record(profile_type: str = "high_risk") -> Dict[str, Any]:
    """Provides realistic sample customer parameters for quick manual testing."""
    if profile_type == "high_risk":
        return {
            'customer_id': f"MANUAL-{uuid.uuid4().hex[:6].upper()}",
            'gender': 'Female',
            'senior_citizen': 1,
            'partner': 'No',
            'dependents': 'No',
            'tenure': 2,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'Fiber optic',
            'online_security': 'No',
            'online_backup': 'No',
            'device_protection': 'No',
            'tech_support': 'No',
            'streaming_tv': 'Yes',
            'streaming_movies': 'Yes',
            'contract': 'Month-to-month',
            'paperless_billing': 'Yes',
            'payment_method': 'Electronic check',
            'monthly_charges': 89.50,
            'total_charges': 179.00,
            'churn_score': 85.0,
        }
    elif profile_type == "low_risk":
        return {
            'customer_id': f"MANUAL-{uuid.uuid4().hex[:6].upper()}",
            'gender': 'Male',
            'senior_citizen': 0,
            'partner': 'Yes',
            'dependents': 'Yes',
            'tenure': 60,
            'phone_service': 'Yes',
            'multiple_lines': 'Yes',
            'internet_service': 'DSL',
            'online_security': 'Yes',
            'online_backup': 'Yes',
            'device_protection': 'Yes',
            'tech_support': 'Yes',
            'streaming_tv': 'No',
            'streaming_movies': 'No',
            'contract': 'Two year',
            'paperless_billing': 'No',
            'payment_method': 'Credit card (automatic)',
            'monthly_charges': 64.20,
            'total_charges': 3852.00,
            'churn_score': 15.0,
        }
    else:  # moderate
        return {
            'customer_id': f"MANUAL-{uuid.uuid4().hex[:6].upper()}",
            'gender': 'Female',
            'senior_citizen': 0,
            'partner': 'Yes',
            'dependents': 'No',
            'tenure': 24,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'DSL',
            'online_security': 'Yes',
            'online_backup': 'No',
            'device_protection': 'Yes',
            'tech_support': 'No',
            'streaming_tv': 'No',
            'streaming_movies': 'Yes',
            'contract': 'One year',
            'paperless_billing': 'Yes',
            'payment_method': 'Bank transfer (automatic)',
            'monthly_charges': 55.40,
            'total_charges': 1329.60,
            'churn_score': 45.0,
        }


def generate_sample_cohort(size: int = 5) -> pd.DataFrame:
    """Generates a small cohort of diverse customer records for instant testing."""
    records = []
    types = ['high_risk', 'low_risk', 'moderate']
    for i in range(size):
        ptype = types[i % len(types)]
        rec = generate_sample_customer_record(ptype)
        rec['customer_id'] = f"MANUAL-{1001 + i}"
        records.append(rec)
    return pd.DataFrame(records)
