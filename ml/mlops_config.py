"""
ml/mlops_config.py
Centralized MLOps Configuration & MLflow Setup.
Configures tracking URI with SQLite backend for full Model Registry support,
standardizes experiment naming conventions, and defines artifact storage paths.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Centralized MLflow paths
MLRUNS_DB_PATH = os.path.join(PROJECT_ROOT, 'mlruns.db')
MLFLOW_TRACKING_URI = f"sqlite:///{MLRUNS_DB_PATH.replace(os.sep, '/')}"
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'mlruns_artifacts')

# Experiment naming definitions
CHURN_EXPERIMENT_NAME = "Customer_Churn_Prediction"
CLV_EXPERIMENT_NAME = "Customer_Lifetime_Value_Regression"
SEGMENTATION_EXPERIMENT_NAME = "Customer_Segmentation_KMeans"

# Model Registry registered model names
CHURN_MODEL_REGISTRY_NAME = "churn_xgboost_classifier"
CLV_MODEL_REGISTRY_NAME = "clv_xgboost_regressor"

# Champion Quality Gate Thresholds
CHAMPION_MIN_ACCURACY = 0.920
CHAMPION_MIN_ROC_AUC = 0.950
CHAMPION_MIN_RECALL = 0.850
CHAMPION_MAX_FPR = 0.070



def setup_mlflow():
    """Initializes MLflow tracking URI, artifact directories, and default experiments."""
    try:
        import mlflow

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Pre-create or fetch default experiments
        for exp_name in [CHURN_EXPERIMENT_NAME, CLV_EXPERIMENT_NAME, SEGMENTATION_EXPERIMENT_NAME]:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                mlflow.create_experiment(
                    name=exp_name,
                    artifact_location=os.path.join(ARTIFACTS_DIR, exp_name).replace(os.sep, '/')
                )

        return {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "artifacts_dir": ARTIFACTS_DIR,
            "churn_experiment": CHURN_EXPERIMENT_NAME,
            "clv_experiment": CLV_EXPERIMENT_NAME,
            "mlflow_available": True
        }
    except Exception as e:
        return {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "artifacts_dir": ARTIFACTS_DIR,
            "churn_experiment": CHURN_EXPERIMENT_NAME,
            "clv_experiment": CLV_EXPERIMENT_NAME,
            "mlflow_available": False,
            "error": str(e)
        }


def get_champion_gate_rules():
    """Returns the strict validation dictionary for Champion vs Challenger gating."""
    return {
        "min_accuracy": CHAMPION_MIN_ACCURACY,
        "min_roc_auc": CHAMPION_MIN_ROC_AUC,
        "min_recall": CHAMPION_MIN_RECALL,
        "max_fpr": CHAMPION_MAX_FPR,
    }
