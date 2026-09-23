"""
ml/custom_data_store.py
Backend Persistence Layer for Manually Entered and Imported Customer Cohorts.
Guarantees resilient persistence across PostgreSQL and local structured storage (CSV),
enabling custom customer records to be inspected in analytics dashboards and
fetched on demand in the Predictions and Inference modules.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CUSTOM_CSV_PATH = os.path.join(DATA_DIR, 'custom_imported_customers.csv')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _init_db_tables():
    """Initializes tables in PostgreSQL if available."""
    try:
        from database.connection import engine, Base
        from database.models import CustomCustomer
        Base.metadata.create_all(bind=engine)
    except Exception:
        pass


def save_manual_cohort(df: pd.DataFrame, batch_id: Optional[str] = None) -> int:
    """
    Persists a DataFrame of customer records to PostgreSQL and mirrors to a local CSV store.
    Returns the count of successfully persisted records.
    """
    if df.empty:
        return 0

    _init_db_tables()

    records_df = df.copy()
    if 'created_at' not in records_df.columns:
        records_df['created_at'] = datetime.datetime.utcnow().isoformat()
    if batch_id and 'batch_id' not in records_df.columns:
        records_df['batch_id'] = batch_id

    # 1. Local CSV store persistence
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CUSTOM_CSV_PATH):
        try:
            existing_csv = pd.read_csv(CUSTOM_CSV_PATH)
            # Remove any overlapping customer_ids to update them
            existing_ids = set(records_df['customer_id'].astype(str))
            existing_csv = existing_csv[~existing_csv['customer_id'].astype(str).isin(existing_ids)]
            combined_csv = pd.concat([existing_csv, records_df], ignore_index=True)
            combined_csv.to_csv(CUSTOM_CSV_PATH, index=False)
        except Exception:
            records_df.to_csv(CUSTOM_CSV_PATH, index=False)
    else:
        records_df.to_csv(CUSTOM_CSV_PATH, index=False)

    # 2. PostgreSQL persistence
    saved_count = len(records_df)
    try:
        from database.connection import SessionLocal
        from database.models import CustomCustomer, Customer

        db = SessionLocal()
        for _, row in records_df.iterrows():
            cid = str(row['customer_id'])

            # Query existing in CustomCustomer
            existing = db.query(CustomCustomer).filter_by(customer_id=cid).first()
            if not existing:
                custom_entry = CustomCustomer(
                    customer_id=cid,
                    gender=str(row.get('gender', 'Male')),
                    senior_citizen=int(row.get('senior_citizen', 0)),
                    partner=str(row.get('partner', 'No')),
                    dependents=str(row.get('dependents', 'No')),
                    tenure=int(row.get('tenure', 1)),
                    contract=str(row.get('contract', 'Month-to-month')),
                    paperless_billing=str(row.get('paperless_billing', 'Yes')),
                    payment_method=str(row.get('payment_method', 'Electronic check')),
                    phone_service=str(row.get('phone_service', 'Yes')),
                    multiple_lines=str(row.get('multiple_lines', 'No')),
                    internet_service=str(row.get('internet_service', 'DSL')),
                    online_security=str(row.get('online_security', 'No')),
                    online_backup=str(row.get('online_backup', 'No')),
                    device_protection=str(row.get('device_protection', 'No')),
                    tech_support=str(row.get('tech_support', 'No')),
                    streaming_tv=str(row.get('streaming_tv', 'No')),
                    streaming_movies=str(row.get('streaming_movies', 'No')),
                    monthly_charges=float(row.get('monthly_charges', 65.0)),
                    total_charges=float(row.get('total_charges', 65.0)),
                    churn_score=float(row.get('churn_score', 50.0)),
                    churn_probability=float(row.get('churn_probability', 0.0)) if pd.notnull(row.get('churn_probability')) else None,
                    predicted_churn=int(row.get('predicted_churn', 0)) if pd.notnull(row.get('predicted_churn')) else None,
                    predicted_clv=float(row.get('predicted_clv', 0.0)) if pd.notnull(row.get('predicted_clv')) else None,
                    risk_tier=str(row.get('risk_tier', 'Low')),
                    clv_tier=str(row.get('clv_tier', 'Low Yield')),
                    batch_id=str(row.get('batch_id', batch_id or 'manual')),
                    created_at=datetime.datetime.utcnow()
                )
                db.add(custom_entry)
            else:
                # Update existing
                existing.monthly_charges = float(row.get('monthly_charges', 65.0))
                existing.total_charges = float(row.get('total_charges', 65.0))
                existing.tenure = int(row.get('tenure', 1))
                existing.contract = str(row.get('contract', 'Month-to-month'))
                existing.churn_score = float(row.get('churn_score', 50.0))
                existing.churn_probability = float(row.get('churn_probability', 0.0)) if pd.notnull(row.get('churn_probability')) else None
                existing.predicted_churn = int(row.get('predicted_churn', 0)) if pd.notnull(row.get('predicted_churn')) else None
                existing.predicted_clv = float(row.get('predicted_clv', 0.0)) if pd.notnull(row.get('predicted_clv')) else None
                existing.risk_tier = str(row.get('risk_tier', 'Low'))
                existing.clv_tier = str(row.get('clv_tier', 'Low Yield'))

            # Also mirror into primary customers table for general queries if desired
            base_cust = db.query(Customer).filter_by(customer_id=cid).first()
            if not base_cust:
                new_base = Customer(
                    customer_id=cid,
                    gender=str(row.get('gender', 'Male')),
                    senior_citizen=int(row.get('senior_citizen', 0)),
                    partner=str(row.get('partner', 'No')),
                    dependents=str(row.get('dependents', 'No')),
                    tenure=int(row.get('tenure', 1)),
                    contract=str(row.get('contract', 'Month-to-month')),
                    paperless_billing=str(row.get('paperless_billing', 'Yes')),
                    payment_method=str(row.get('payment_method', 'Electronic check')),
                    phone_service=str(row.get('phone_service', 'Yes')),
                    multiple_lines=str(row.get('multiple_lines', 'No')),
                    internet_service=str(row.get('internet_service', 'DSL')),
                    online_security=str(row.get('online_security', 'No')),
                    online_backup=str(row.get('online_backup', 'No')),
                    device_protection=str(row.get('device_protection', 'No')),
                    tech_support=str(row.get('tech_support', 'No')),
                    streaming_tv=str(row.get('streaming_tv', 'No')),
                    streaming_movies=str(row.get('streaming_movies', 'No')),
                    monthly_charges=float(row.get('monthly_charges', 65.0)),
                    total_charges=float(row.get('total_charges', 65.0)),
                    churn=str(row.get('churn', 'No')),
                    churn_score=float(row.get('churn_score', 50.0)),
                    predicted_churn=float(row.get('predicted_churn', 0)),
                    predicted_clv=float(row.get('predicted_clv', 0)),
                    segment=str(row.get('risk_tier', 'Custom')),
                    created_at=datetime.datetime.utcnow()
                )
                db.add(new_base)

        db.commit()
        db.close()
    except Exception:
        pass

    return saved_count


def load_manual_customers() -> pd.DataFrame:
    """
    Loads all custom/manually entered customer records from PostgreSQL or local CSV store.
    """
    _init_db_tables()

    # Attempt PostgreSQL load first
    try:
        from database.connection import engine
        query = "SELECT * FROM custom_customers ORDER BY created_at DESC"
        df = pd.read_sql(query, engine)
        if not df.empty:
            return df
    except Exception:
        pass

    # Local CSV store fallback
    if os.path.exists(CUSTOM_CSV_PATH):
        try:
            df = pd.read_csv(CUSTOM_CSV_PATH)
            return df
        except Exception:
            pass

    return pd.DataFrame()


def get_custom_store_signature() -> str:
    """
    Returns a unique cache-invalidation signature string based on current
    custom customer count and CSV file modification timestamp.
    """
    mtime = 0
    if os.path.exists(CUSTOM_CSV_PATH):
        try:
            mtime = os.path.getmtime(CUSTOM_CSV_PATH)
        except Exception:
            pass

    db_count = 0
    try:
        from database.connection import SessionLocal
        from database.models import CustomCustomer
        db = SessionLocal()
        db_count = db.query(CustomCustomer).count()
        db.close()
    except Exception:
        pass

    return f"{mtime}_{db_count}"


def load_all_combined_customers() -> Tuple[pd.DataFrame, str]:
    """
    Retrieves the complete customer population: baseline customers (7,043)
    plus any manually entered records, tagging custom accounts with `is_custom=True`.
    Custom accounts are positioned at the beginning of the DataFrame so they are
    immediately selectable and prominent in the UI.
    """
    from ml.data_preprocessing import load_data_from_db

    base_df = load_data_from_db()
    base_df['is_custom'] = False

    custom_df = load_manual_customers()
    if not custom_df.empty:
        custom_ids = set(custom_df['customer_id'].astype(str))
        base_df.loc[base_df['customer_id'].astype(str).isin(custom_ids), 'is_custom'] = True
        base_df.loc[base_df['customer_id'].astype(str).str.startswith("MANUAL-"), 'is_custom'] = True

        existing_ids = set(base_df['customer_id'].astype(str))
        custom_unique = custom_df[~custom_df['customer_id'].astype(str).isin(existing_ids)].copy()
        if not custom_unique.empty:
            custom_unique['is_custom'] = True

        # Separate custom accounts from baseline accounts
        custom_in_base = base_df[base_df['is_custom']].copy()
        regular_base = base_df[~base_df['is_custom']].copy()

        all_custom = pd.concat([custom_in_base, custom_unique], ignore_index=True) if not custom_unique.empty else custom_in_base
        # Place custom accounts at the front of the dataset
        combined = pd.concat([all_custom, regular_base], ignore_index=True)

        custom_count = len(all_custom)
        base_count = len(regular_base)
        return combined, f"Combined Dataset: {base_count} Baseline + {custom_count} Custom"

    return base_df, f"Baseline Dataset: {len(base_df)} Records"


def delete_manual_customer(customer_id: str) -> bool:
    """Removes a single manually entered customer from both PostgreSQL and local CSV."""
    success = False

    # Remove from local CSV
    if os.path.exists(CUSTOM_CSV_PATH):
        try:
            df = pd.read_csv(CUSTOM_CSV_PATH)
            df = df[df['customer_id'].astype(str) != str(customer_id)]
            df.to_csv(CUSTOM_CSV_PATH, index=False)
            success = True
        except Exception:
            pass

    # Remove from PostgreSQL
    try:
        from database.connection import SessionLocal
        from database.models import CustomCustomer, Customer
        db = SessionLocal()
        cust = db.query(CustomCustomer).filter_by(customer_id=str(customer_id)).first()
        if cust:
            db.delete(cust)
        base = db.query(Customer).filter_by(customer_id=str(customer_id)).first()
        if base and getattr(base, 'segment', '') in ['High', 'Medium', 'Low', 'Custom']:
            db.delete(base)
        db.commit()
        db.close()
        success = True
    except Exception:
        pass

    return success


def clear_manual_cohort() -> int:
    """Clears all custom customer records from both storage layers."""
    count = 0

    if os.path.exists(CUSTOM_CSV_PATH):
        try:
            df = pd.read_csv(CUSTOM_CSV_PATH)
            count = len(df)
            os.remove(CUSTOM_CSV_PATH)
        except Exception:
            pass

    try:
        from database.connection import SessionLocal
        from database.models import CustomCustomer, Customer
        db = SessionLocal()
        c_count = db.query(CustomCustomer).count()
        custom_cids = [c.customer_id for c in db.query(CustomCustomer.customer_id).all()]
        db.query(CustomCustomer).delete()
        if custom_cids:
            db.query(Customer).filter(Customer.customer_id.in_(custom_cids)).delete(synchronize_session=False)
        db.query(Customer).filter(Customer.customer_id.like("MANUAL-%")).delete(synchronize_session=False)
        db.query(Customer).filter(Customer.id > 7043).delete(synchronize_session=False)
        db.commit()
        db.close()
        count = max(count, c_count)
    except Exception:
        pass

    return count
