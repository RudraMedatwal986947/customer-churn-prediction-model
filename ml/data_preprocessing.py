import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from database.connection import engine

def load_data_from_db():
    """Load customer data from PostgreSQL database"""
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, engine)
    return df

def preprocess_data(df, is_training=True, scaler=None):
    """
    Handle missing values, feature engineering, encoding, and scaling.
    """
    # 1. Drop irrelevant columns for modeling
    drop_cols = ['id', 'customer_id', 'created_at', 'predicted_churn', 'predicted_clv', 'segment']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # 2. Data Cleaning
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0)
    
    # 3. Feature Engineering
    # A. Tenure grouping
    def map_tenure(tenure):
        if tenure <= 12: return '0_1_year'
        elif tenure <= 24: return '1_2_years'
        elif tenure <= 36: return '2_3_years'
        elif tenure <= 48: return '3_4_years'
        elif tenure <= 60: return '4_5_years'
        else: return '5_plus_years'
        
    df['tenure_group'] = df['tenure'].apply(map_tenure)
    
    # B. Count of additional services
    services = ['online_security', 'online_backup', 'device_protection', 
                'tech_support', 'streaming_tv', 'streaming_movies']
    
    # Count services that are 'Yes'
    df['total_additional_services'] = 0
    for service in services:
        if service in df.columns:
            df['total_additional_services'] += (df[service] == 'Yes').astype(int)
            
    # C. Average monthly charge vs current (proxy for price increases)
    df['avg_monthly_charge'] = df['total_charges'] / (df['tenure'] + 1)
    df['charge_difference'] = df['monthly_charges'] - df['avg_monthly_charge']
    
    # 4. Encoding
    # Separate target variable if present
    y_churn = None
    if 'churn' in df.columns:
        y_churn = (df['churn'] == 'Yes').astype(int)
        df = df.drop(columns=['churn'])
        
    # Convert binary categorical to 1/0
    binary_cols = ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']
    for col in binary_cols:
        if col in df.columns:
            if col == 'gender':
                df[col] = (df[col] == 'Female').astype(int) # Female: 1, Male: 0
            else:
                df[col] = (df[col] == 'Yes').astype(int)
                
    # One-Hot Encoding for multi-class categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 5. Scaling
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges', 
                      'total_additional_services', 'avg_monthly_charge', 'charge_difference']
    
    # Ensure numerical cols exist
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    
    if is_training:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df, y_churn, scaler
    else:
        if scaler:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df
