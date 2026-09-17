import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data_from_db():
    """Load customer data from PostgreSQL database with automatic Excel fallback."""
    import os
    try:
        from database.connection import engine
        query = "SELECT * FROM customers"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_customer_churn.xlsx'))
        df = pd.read_excel(excel_path)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_', regex=False)
        )
        rename_map = {
            'customerid':    'customer_id',
            'churn_label':   'churn',
            'tenure_months': 'tenure',
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        return df

def preprocess_data(df, is_training=True, scaler=None):
    """
    Handle missing values, feature engineering, encoding, and scaling.
    """
    # 1. Drop irrelevant or target-leaking columns for modeling
    drop_cols = [
        'id', 'customer_id', 'created_at', 'predicted_churn', 'predicted_clv', 'segment',
        'churn_value', 'churn_reason', 'cltv', 'lat_long', 'latitude', 'longitude',
        'city', 'state', 'country', 'zip_code', 'count'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # 2. Data Cleaning
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0)

    # 2b. Churn Score / Risk Propensity feature (with 50.0 neutral median fallback)
    if 'churn_score' in df.columns:
        df['churn_score'] = pd.to_numeric(df['churn_score'], errors='coerce').fillna(50.0)
    else:
        df['churn_score'] = 50.0

    # 3. Feature Engineering — Original Features
    # A. Tenure grouping
    def map_tenure(tenure):
        if tenure <= 12:   return '0_1_year'
        elif tenure <= 24: return '1_2_years'
        elif tenure <= 36: return '2_3_years'
        elif tenure <= 48: return '3_4_years'
        elif tenure <= 60: return '4_5_years'
        else:              return '5_plus_years'

    df['tenure_group'] = df['tenure'].apply(map_tenure)

    # B. Count of additional services
    services = ['online_security', 'online_backup', 'device_protection',
                'tech_support', 'streaming_tv', 'streaming_movies']
    df['total_additional_services'] = sum(
        (df[s] == 'Yes').astype(int) for s in services if s in df.columns
    )

    # C. Average monthly charge vs current (proxy for price increases)
    df['avg_monthly_charge'] = df['total_charges'] / (df['tenure'] + 1)
    df['charge_difference']  = df['monthly_charges'] - df['avg_monthly_charge']

    # --- Stage 1: Advanced Feature Engineering ---

    # D. Contract risk score (month-to-month = 2, one-year = 1, two-year = 0)
    if 'contract' in df.columns:
        contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
        df['contract_risk_score'] = df['contract'].map(contract_risk).fillna(1).astype(int)

    # E. Senior citizen on month-to-month contract (highest churn risk combo)
    if 'senior_citizen' in df.columns and 'contract' in df.columns:
        is_senior = df['senior_citizen'].astype(str).isin(['1', 'Yes', 'True', '1.0'])
        is_mtm    = df['contract'] == 'Month-to-month'
        df['senior_no_contract'] = (is_senior & is_mtm).astype(int)

    # F. High charges but no tech support / security (dissatisfied high-value customer)
    if 'monthly_charges' in df.columns:
        high_threshold = df['monthly_charges'].median()
        no_security = df.get('online_security', pd.Series('No', index=df.index)) == 'No'
        no_support  = df.get('tech_support',    pd.Series('No', index=df.index)) == 'No'
        df['high_charge_no_support'] = (
            (df['monthly_charges'] > high_threshold) & no_security & no_support
        ).astype(int)

    # G. Payment method risk score (electronic check = highest churn historically)
    if 'payment_method' in df.columns:
        payment_risk = {
            'Electronic check':            2,
            'Mailed check':                1,
            'Bank transfer (automatic)':   0,
            'Credit card (automatic)':     0,
        }
        df['payment_risk_score'] = df['payment_method'].map(payment_risk).fillna(1).astype(int)

    # H. Tenure × monthly charges interaction (value proxy before normalisation)
    df['tenure_x_charges'] = df['tenure'] * df['monthly_charges']

    # 4. Encoding — Separate target variable if present
    y_churn = None
    if 'churn' in df.columns:
        y_churn = (df['churn'] == 'Yes').astype(int)
        df = df.drop(columns=['churn'])

    # Convert binary categorical to 1/0
    if 'senior_citizen' in df.columns:
        if df['senior_citizen'].dtype == object:
            df['senior_citizen'] = (df['senior_citizen'].astype(str).str.lower() == 'yes').astype(int)
        else:
            df['senior_citizen'] = df['senior_citizen'].fillna(0).astype(int)

    binary_cols = ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']
    for col in binary_cols:
        if col in df.columns:
            if col == 'gender':
                df[col] = (df[col] == 'Female').astype(int)  # Female:1, Male:0
            else:
                df[col] = (df[col] == 'Yes').astype(int)

    # One-Hot Encoding for multi-class categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df.columns = [c.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for c in df.columns]

    # 5. Scaling
    numerical_cols = [
        'tenure', 'monthly_charges', 'total_charges',
        'total_additional_services', 'avg_monthly_charge', 'charge_difference',
        'contract_risk_score', 'tenure_x_charges', 'churn_score',
    ]
    numerical_cols = [c for c in numerical_cols if c in df.columns]

    if is_training:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df, y_churn, scaler
    else:
        if scaler:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df
