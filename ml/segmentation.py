import os
import sys
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ensure we can import from siblings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import engine, SessionLocal
from database.models import Customer

def load_data_from_db():
    query = "SELECT * FROM customers"
    return pd.read_sql(query, engine)

def extract_features_for_clustering(df):
    """
    Extract relevant numerical features for customer segmentation (K-Means).
    We use Tenure, Monthly Charges, Total Charges, and Total Services count.
    """
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0)
    
    # Calculate total additional services
    services = ['online_security', 'online_backup', 'device_protection', 
                'tech_support', 'streaming_tv', 'streaming_movies']
    
    df['total_additional_services'] = 0
    for service in services:
        if service in df.columns:
            df['total_additional_services'] += (df[service] == 'Yes').astype(int)
            
    features = ['tenure', 'monthly_charges', 'total_charges', 'total_additional_services']
    return df[['customer_id'] + features]

def train_segmentation_model(n_clusters=4):
    print("Loading data for segmentation...")
    df = load_data_from_db()
    
    print("Extracting features...")
    cluster_df = extract_features_for_clustering(df)
    
    # We don't cluster on customer_id
    X = cluster_df.drop(columns=['customer_id'])
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Training K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    cluster_df['segment'] = cluster_labels
    
    # Save model and scaler
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'kmeans_scaler.pkl'))
    print("Segmentation model saved.")
    
    # Update the database with the assigned segments
    print("Updating database with segments...")
    db = SessionLocal()
    try:
        # We can bulk update but for 7k rows a simple loop or execute is fine
        for _, row in cluster_df.iterrows():
            # Create a string label for the segment
            segment_label = f"Segment {row['segment']}"
            db.query(Customer).filter(Customer.customer_id == row['customer_id']).update(
                {"segment": segment_label}
            )
        db.commit()
        print("Database updated successfully.")
    except Exception as e:
        print(f"Error updating database: {e}")
        db.rollback()
    finally:
        db.close()
        
    return cluster_df

if __name__ == "__main__":
    train_segmentation_model()
