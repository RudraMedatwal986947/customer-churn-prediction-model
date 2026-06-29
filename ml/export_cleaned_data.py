import pandas as pd
import os
import sys

# Add the parent directory to sys.path so we can import from ml and database modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data_preprocessing import load_data_from_db, preprocess_data

def export_data():
    print("Loading data from database...")
    df = load_data_from_db()
    
    print(f"Original dataset shape: {df.shape}")
    print("Original columns:")
    print(df.columns.tolist())
    print("\n-----------------------------------\n")
    
    print("Applying preprocessing and feature engineering...")
    cleaned_df, y_churn, scaler = preprocess_data(df, is_training=True)
    
    # Re-attach the target variable for the exported dataset
    if y_churn is not None:
        cleaned_df['churn_label'] = y_churn
        
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print("Relevant features after engineering & encoding:")
    for col in cleaned_df.columns:
        print(f" - {col}")
        
    output_path = "data/cleaned_customer_data.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully exported cleaned and engineered data to: {output_path}")

if __name__ == "__main__":
    export_data()
