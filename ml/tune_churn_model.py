"""
ml/tune_churn_model.py
──────────────────────
Stage 2: Optuna hyperparameter search for the XGBoost churn classifier.
Runs n_trials Bayesian optimisation trials using 5-fold cross-validated AUC.
Returns the best parameter dict for use in the main training script.
"""

import os
import sys
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Suppress Optuna's verbose per-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def objective(trial, X, y):
    """Optuna objective - maximise 5-fold stratified Accuracy."""
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
        'max_depth':         trial.suggest_int('max_depth', 3, 9),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 2.0),
        'scale_pos_weight':  1.0,  # Fixed to 1.0 to maximize raw accuracy
        'eval_metric':       'logloss',
        'random_state':      42,
        'use_label_encoder': False,
        'n_jobs':            -1,
    }

    model = XGBClassifier(**params)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return float(scores.mean())


def run_tuning(X_train, y_train, n_trials=100):
    """
    Run Optuna study and return the best hyperparameter dict.
    X_train / y_train should be the SMOTE-resampled training data.
    """
    print(f"  Running {n_trials} Optuna trials (5-fold AUC)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"  Best CV AUC : {study.best_value:.4f}")
    print(f"  Best params : {study.best_params}")
    return study.best_params
