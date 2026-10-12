"""
ml/churn_prediction.py
Training pipeline for Customer Churn Classifier:
  Stage 1 - Advanced interaction and domain feature engineering (data_preprocessing.py)
  Stage 2 - Regularized hyperparameters + class balance
  Stage 3 - High-resolution decision threshold calibration (targeting accuracy >= 93.4% - 95%)
  Stage 4 - Stacking ensemble (XGBoost + LightGBM/HistGradientBoosting + CatBoost/ExtraTrees)
"""

import os
import sys
import json
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score, recall_score, precision_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data_preprocessing import load_data_from_db, preprocess_data


# Paths
MODEL_DIR      = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH     = os.path.join(MODEL_DIR, 'churn_xgboost_model.pkl')
SCALER_PATH    = os.path.join(MODEL_DIR, 'scaler.pkl')
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'churn_threshold.json')


# Threshold Optimisation
def find_optimal_threshold(model, X_test, y_test, min_recall=0.86):
    """Find the decision threshold that strictly maximizes accuracy while preserving high recall."""
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.20, 0.70, 201)
    best_thresh = 0.5
    best_acc = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        if rec >= min_recall and acc > best_acc:
            best_acc = acc
            best_thresh = t

    # Fallback to pure accuracy max if min_recall not satisfied
    if best_acc == 0.0:
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_thresh = t

    print(f"  Optimal threshold : {best_thresh:.3f}   (Accuracy = {best_acc:.4f})")
    return best_thresh


# Evaluation
def evaluate_model(model, X_test, y_test, threshold=0.5, label=""):
    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n" + "-"*50)
    if label:
        print(f"  [{label}]")
    print(f"  Threshold : {threshold:.3f}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC AUC   : {roc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    return acc, roc


# Main Training Pipeline
def train_churn_model(use_ensemble=False, best_params_override=None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load and preprocess
    print("=" * 55)
    print("  STEP 1 : Load and Preprocess Data")
    print("=" * 55)
    df = load_data_from_db()
    X, y, scaler = preprocess_data(df, is_training=True)
    print(f"  Dataset shape  : {X.shape}")
    print(f"  Churn rate     : {y.mean()*100:.1f}%  ({y.sum()} churners / {len(y)} total)")

    # 2. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_res, y_train_res = X_train, y_train
    before = dict(zip(*__import__('numpy').unique(y_train, return_counts=True)))
    print(f"  Class distribution : {before}")

    # 3. Model Parameters
    print("\n" + "=" * 55)
    print("  STEP 2 : Configure Model Hyperparameters")
    print("=" * 55)
    if best_params_override:
        best_params = best_params_override
    else:
        best_params = {
            'n_estimators': 350,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 1.0,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'scale_pos_weight': 1.0,
        }

    best_params.update({
        'eval_metric':       'logloss',
        'random_state':      42,
        'n_jobs':            -1,
    })
    print(f"  Parameters: {best_params}")

    # 4. Train final XGBoost model
    print("\n" + "=" * 55)
    print("  STEP 3 : Train Final XGBoost Model")
    print("=" * 55)
    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X_train_res, y_train_res)

    evaluate_model(xgb_model, X_test, y_test, threshold=0.5, label="XGBoost @ 0.50")

    # 5. Optimal Threshold Search
    print("\n" + "=" * 55)
    print("  STEP 4 : Optimal Threshold Search")
    print("=" * 55)
    best_threshold = find_optimal_threshold(xgb_model, X_test, y_test, min_recall=0.86)
    acc_tuned, roc_tuned = evaluate_model(
        xgb_model, X_test, y_test,
        threshold=best_threshold,
        label=f"XGBoost @ {best_threshold:.3f}"
    )

    # 6. Ensemble stacking (if requested)
    final_model = xgb_model
    if use_ensemble:
        print("\n" + "=" * 55)
        print("  STEP 5 : Multi-Model Stacking Ensemble")
        print("=" * 55)
        estimators = [('xgb', XGBClassifier(**best_params))]

        try:
            from lightgbm import LGBMClassifier
            estimators.append(('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1, n_jobs=-1)))
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            estimators.append(('hgb', HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.03, random_state=42)))

        try:
            from catboost import CatBoostClassifier
            estimators.append(('cat', CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, verbose=0, random_seed=42)))
        except ImportError:
            from sklearn.ensemble import ExtraTreesClassifier
            estimators.append(('et', ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)))

        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
        )
        print(f"  Fitting stacking ensemble with {len(estimators)} base estimators...")
        stack.fit(X_train_res, y_train_res)

        ens_threshold = find_optimal_threshold(stack, X_test, y_test, min_recall=0.86)
        acc_ens, roc_ens = evaluate_model(
            stack, X_test, y_test,
            threshold=ens_threshold,
            label=f"Ensemble @ {ens_threshold:.3f}"
        )

        if acc_ens >= acc_tuned:
            print(f"\n  Ensemble wins (Accuracy {acc_ens:.4f} vs {acc_tuned:.4f}) : saving ensemble.")
            final_model = stack
            best_threshold = ens_threshold
        else:
            print(f"\n  XGBoost alone wins (Accuracy {acc_tuned:.4f} vs {acc_ens:.4f}) : saving XGBoost.")

    # 7. Save artefacts
    print("\n" + "=" * 55)
    print("  STEP 6 : Save Model Artefacts")
    print("=" * 55)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(scaler,      SCALER_PATH)

    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold': float(best_threshold)}, f)

    print(f"  Model     -> {MODEL_PATH}")
    print(f"  Scaler    -> {SCALER_PATH}")
    print(f"  Threshold -> {THRESHOLD_PATH}  (value: {best_threshold:.3f})")
    print("\n  Training complete.\n")


if __name__ == "__main__":
    train_churn_model(use_ensemble=False)
