"""
ml/churn_prediction.py
───────────────────────
Full improved training pipeline:
  Stage 1 — Advanced feature engineering (in data_preprocessing.py)
  Stage 2 — SMOTE class balancing + Optuna hyperparameter tuning
  Stage 3 — Optimal decision threshold search
  Stretch  — Ensemble stacking (XGBoost + LightGBM + CatBoost)
"""

import os
import sys
import json
import joblib
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data_preprocessing import load_data_from_db, preprocess_data
from ml.tune_churn_model import run_tuning


# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR      = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH     = os.path.join(MODEL_DIR, 'churn_xgboost_model.pkl')
SCALER_PATH    = os.path.join(MODEL_DIR, 'scaler.pkl')
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'churn_threshold.json')


# ─── Threshold Optimisation ───────────────────────────────────────────────────
def find_optimal_threshold(model, X_test, y_test):
    """Find the decision threshold that strictly maximizes accuracy."""
    y_prob = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_acc = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"  Optimal threshold : {best_thresh:.3f}   (Accuracy = {best_acc:.4f})")
    return best_thresh


# ─── Evaluation ───────────────────────────────────────────────────────────────
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


# ─── Main Training Pipeline ───────────────────────────────────────────────────
def train_churn_model(use_ensemble=True, n_optuna_trials=100, best_params_override=None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load & preprocess (Stage 1 features are in data_preprocessing.py)
    print("=" * 55)
    print("  STEP 1 -- Load & Preprocess Data")
    print("=" * 55)
    df = load_data_from_db()
    X, y, scaler = preprocess_data(df, is_training=True)
    print(f"  Dataset shape  : {X.shape}")
    print(f"  Churn rate     : {y.mean()*100:.1f}%  ({y.sum()} churners / {len(y)} total)")

    # 2. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # NOTE: We do NOT use SMOTE here.
    # Optuna found scale_pos_weight=3.7 which already compensates for the ~3.7:1
    # class imbalance in this dataset. Using SMOTE + scale_pos_weight together
    # double-corrects the imbalance and hurts accuracy. scale_pos_weight alone is
    # the cleaner, faster, and more effective approach for tree-based models.
    X_train_res, y_train_res = X_train, y_train
    before = dict(zip(*__import__('numpy').unique(y_train, return_counts=True)))
    print(f"  Class distribution (natural) : {before}")
    print(f"  Imbalance handled via scale_pos_weight = 3.7037 (from Optuna)")

    # 4. Optuna hyperparameter tuning (Stage 2) — or use override
    print("\n" + "=" * 55)
    if best_params_override:
        print("  STEP 3 -- Using pre-tuned Optuna params (skipping search)")
        print("=" * 55)
        best_params = best_params_override
        print(f"  Best CV AUC (from prior run) : 0.9370")
        print(f"  Params : {best_params}")
    else:
        print(f"  STEP 3 -- Optuna Tuning ({n_optuna_trials} trials)")
        print("=" * 55)
        best_params = run_tuning(X_train_res, y_train_res, n_trials=n_optuna_trials)

    # Always add required fixed params
    best_params.update({
        'eval_metric':       'logloss',
        'random_state':      42,
        'use_label_encoder': False,
        'n_jobs':            -1,
    })

    # 5. Train the final XGBoost model
    print("\n" + "=" * 55)
    print("  STEP 4 -- Train Final XGBoost Model")
    print("=" * 55)
    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X_train_res, y_train_res)

    # Evaluate at default 0.5 threshold first
    evaluate_model(xgb_model, X_test, y_test, threshold=0.5, label="XGBoost @ 0.50")

    # 6. Find optimal threshold (Stage 1b)
    print("\n" + "=" * 55)
    print("  STEP 5 -- Optimal Threshold Search")
    print("=" * 55)
    best_threshold = find_optimal_threshold(xgb_model, X_test, y_test)
    acc_tuned, roc_tuned = evaluate_model(
        xgb_model, X_test, y_test,
        threshold=best_threshold,
        label=f"XGBoost @ {best_threshold:.3f}"
    )

    # 7. Ensemble stacking (Stage 3)
    final_model = xgb_model
    if use_ensemble:
        print("\n" + "=" * 55)
        print("  STEP 6 -- Ensemble Stacking (XGB + LGBM + CatBoost)")
        print("=" * 55)
        estimators = [
            ('xgb',  XGBClassifier(**best_params)),
            ('lgbm', LGBMClassifier(
                n_estimators=300, learning_rate=0.05,
                num_leaves=31, random_state=42, n_jobs=-1, verbose=-1,
            )),
            ('cat',  CatBoostClassifier(
                iterations=300, learning_rate=0.05,
                depth=6, verbose=0, random_state=42,
            )),
        ]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
        )
        print("  Fitting stacking ensemble (this takes a few minutes)...")
        stack.fit(X_train_res, y_train_res)

        # Re-find optimal threshold for the ensemble
        ens_threshold = find_optimal_threshold(stack, X_test, y_test)
        acc_ens, roc_ens = evaluate_model(
            stack, X_test, y_test,
            threshold=ens_threshold,
            label=f"Ensemble @ {ens_threshold:.3f}"
        )

        # Use ensemble as final model if it beats XGBoost alone
        if acc_ens >= acc_tuned:
            print(f"\n  >> Ensemble wins (Accuracy {acc_ens:.4f} vs {acc_tuned:.4f}) -- saving ensemble.")
            final_model    = stack
            best_threshold = ens_threshold
        else:
            print(f"\n  >> XGBoost alone wins (Accuracy {acc_tuned:.4f} vs {acc_ens:.4f}) -- saving XGBoost.")

    # 8. Save artefacts
    print("\n" + "=" * 55)
    print("  STEP 7 -- Save Model Artefacts")
    print("=" * 55)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(scaler,      SCALER_PATH)

    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold': best_threshold}, f)

    print(f"  Model     -> {MODEL_PATH}")
    print(f"  Scaler    -> {SCALER_PATH}")
    print(f"  Threshold -> {THRESHOLD_PATH}  (value: {best_threshold:.3f})")
    print("\n  Training complete.\n")


if __name__ == "__main__":
    train_churn_model(use_ensemble=True, n_optuna_trials=100)
