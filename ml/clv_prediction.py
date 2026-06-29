import os
import sys
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Ensure we can import from siblings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import engine

def load_data_from_db():
    query = "SELECT * FROM customers"
    return pd.read_sql(query, engine)

def train_clv_model():
    print("Loading data from database...")
    df = load_data_from_db()
    
    # Data Cleaning
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0)
    
    # Target variable for CLV is total_charges
    y = df['total_charges']
    
    # Features (Drop identifiers, labels, and target)
    drop_cols = ['id', 'customer_id', 'created_at', 'churn', 'predicted_churn', 'predicted_clv', 'segment', 'total_charges']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Feature Engineering (replicated from preprocessing for consistency)
    def map_tenure(tenure):
        if tenure <= 12: return '0_1_year'
        elif tenure <= 24: return '1_2_years'
        elif tenure <= 36: return '2_3_years'
        elif tenure <= 48: return '3_4_years'
        elif tenure <= 60: return '4_5_years'
        else: return '5_plus_years'
        
    X['tenure_group'] = X['tenure'].apply(map_tenure)
    
    services = ['online_security', 'online_backup', 'device_protection', 
                'tech_support', 'streaming_tv', 'streaming_movies']
    X['total_additional_services'] = 0
    for service in services:
        if service in X.columns:
            X['total_additional_services'] += (X[service] == 'Yes').astype(int)
            
    # Encoding
    binary_cols = ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']
    for col in binary_cols:
        if col in X.columns:
            if col == 'gender':
                X[col] = (X[col] == 'Female').astype(int)
            else:
                X[col] = (X[col] == 'Yes').astype(int)
                
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'monthly_charges', 'total_additional_services']
    numerical_cols = [c for c in numerical_cols if c in X.columns]
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print(f"Features shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost Regressor
    print("Training XGBoost Regressor for CLV...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n--- CLV Model Evaluation ---")
    evaluate_clv_model(model, X_test, y_test)
    
    # Save model and scaler
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'clv_xgboost_model.pkl')
    scaler_path = os.path.join(model_dir, 'clv_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nCLV Model saved to: {model_path}")
    print(f"CLV Scaler saved to: {scaler_path}")

def evaluate_clv_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE):  {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2):            {r2:.4f}")

if __name__ == "__main__":
    train_clv_model()
