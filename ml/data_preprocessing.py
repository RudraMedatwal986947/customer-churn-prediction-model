import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from database.connection import engine

def load_data_from_db():
    """Load customer data from PostgreSQL database"""
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, engine)
    return df

def preprocess_data(df):
    """
    Handle missing values, encoding, and scaling.
    """
    # Drop irrelevant columns for modeling
    drop_cols = ['id', 'customer_id', 'created_at', 'predicted_churn', 'predicted_clv', 'segment']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Handle missing values
    # Total charges is already handled in seeding, but just in case
    df['total_charges'] = df['total_charges'].fillna(0)
    
    # Encoding categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

