import os
import sys
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Ensure we can import from siblings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data_preprocessing import load_data_from_db, preprocess_data

def train_churn_model():
    print("Loading data from database...")
    df = load_data_from_db()
    
    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df, is_training=True)
    
    if y is None:
        print("Error: Target variable 'churn' not found in data.")
        return
        
    print(f"Data shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize XGBoost Classifier
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\n--- Model Evaluation ---")
    evaluate_model(model, X_test, y_test)
    
    # Save the model and scaler
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'churn_xgboost_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_churn_model()
