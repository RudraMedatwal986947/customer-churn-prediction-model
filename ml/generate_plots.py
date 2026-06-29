import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.data_preprocessing import load_data_from_db, preprocess_data
from ml.clv_prediction import train_clv_model # Wait, we shouldn't retrain, we should just load. We can just load data directly.

# Paths
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_data_for_churn():
    df = load_data_from_db()
    X, y, _ = preprocess_data(df, is_training=True)
    return X, y

def generate_churn_plots():
    print("Generating Churn Plots...")
    try:
        X, y = load_data_for_churn()
        model = joblib.load(os.path.join(MODELS_DIR, 'churn_xgboost_model.pkl'))
        
        # We need a test set to show ROC, let's just use the whole set for viz purposes or split it.
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 1. Feature Importance
        plt.figure(figsize=(10, 8))
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances.nlargest(15).sort_values().plot(kind='barh', color='skyblue')
        plt.title('Top 15 Feature Importances (Churn Model)')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'feature_importance.png'))
        plt.close()
        
        # 2. ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Churn)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'roc_curve.png'))
        plt.close()
        
        # 3. Confusion Matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Churn)')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating churn plots: {e}")

def generate_segmentation_plots():
    print("Generating Segmentation Plots...")
    try:
        from database.connection import engine
        query = "SELECT tenure, monthly_charges, total_charges, segment FROM customers"
        df = pd.read_sql(query, engine)
        
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='tenure', y='monthly_charges', hue='segment', palette='viridis', alpha=0.6)
        plt.title('Customer Segments by Tenure and Monthly Charges')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'segmentation_scatter.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating segmentation plots: {e}")

if __name__ == "__main__":
    generate_churn_plots()
    generate_segmentation_plots()
    print("Done generating plots.")
