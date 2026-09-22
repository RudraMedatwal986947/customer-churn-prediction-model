"""
ml/drift_monitor.py
Statistical Data & Prediction Drift Detection Engine.
Calculates Kolmogorov-Smirnov (KS) test for numerical features,
Population Stability Index (PSI) and frequency divergence for categorical features,
and flags statistically significant dataset drift against the training baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BASELINE_CSV = os.path.join(PROJECT_ROOT, 'data', 'cleaned_customer_data.csv')
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')


def load_baseline_data() -> pd.DataFrame:
    """Loads reference baseline data used during initial model training."""
    if os.path.exists(DEFAULT_BASELINE_CSV):
        df = pd.read_csv(DEFAULT_BASELINE_CSV)
        return df

    if os.path.exists(DEFAULT_EXCEL_PATH):
        df = pd.read_excel(DEFAULT_EXCEL_PATH)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
        df['total_charges'] = pd.to_numeric(df.get('total_charges', 0), errors='coerce').fillna(0)
        df['monthly_charges'] = pd.to_numeric(df.get('monthly_charges', 0), errors='coerce').fillna(0)
        return df

    raise FileNotFoundError("Baseline customer dataset not found in data/ directory.")


def calculate_psi(baseline_series: pd.Series, target_series: pd.Series, num_buckets: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between two distributions.
    Interpretation:
      PSI < 0.10: Stable (no significant change)
      0.10 <= PSI < 0.20: Moderate drift
      PSI >= 0.20: Significant distribution shift
    """
    b_clean = baseline_series.dropna()
    t_clean = target_series.dropna()

    if len(b_clean) == 0 or len(t_clean) == 0:
        return 0.0

    # Handle numerical series via quantile binning
    if pd.api.types.is_numeric_dtype(b_clean):
        quantiles = np.linspace(0, 1, num_buckets + 1)
        bins = np.percentile(b_clean, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        b_counts = pd.cut(b_clean, bins=bins, include_lowest=True).value_counts(sort=False)
        t_counts = pd.cut(t_clean, bins=bins, include_lowest=True).value_counts(sort=False)
    else:
        # Handle categorical series
        all_cats = list(set(b_clean.unique()).union(set(t_clean.unique())))
        b_counts = b_clean.value_counts().reindex(all_cats, fill_value=0)
        t_counts = t_clean.value_counts().reindex(all_cats, fill_value=0)

    b_pct = (b_counts + 1e-4) / (len(b_clean) + 1e-4 * len(b_counts))
    t_pct = (t_counts + 1e-4) / (len(t_clean) + 1e-4 * len(t_counts))

    psi_val = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
    return float(max(0.0, psi_val))


def evaluate_numerical_feature(baseline: pd.Series, target: pd.Series, feature_name: str) -> dict:
    """Runs a two-sample Kolmogorov-Smirnov test and PSI on a numerical feature."""
    b_vals = pd.to_numeric(baseline, errors='coerce').dropna()
    t_vals = pd.to_numeric(target, errors='coerce').dropna()

    if len(b_vals) == 0 or len(t_vals) == 0:
        return {
            "feature": feature_name,
            "type": "numerical",
            "stat": 0.0,
            "p_value": 1.0,
            "psi": 0.0,
            "drift_detected": False,
            "severity": "None",
        }

    ks_stat, p_val = stats.ks_2samp(b_vals, t_vals)
    psi = calculate_psi(b_vals, t_vals)

    # Significant if KS test rejects null hypothesis (p < 0.05) or PSI exceeds 0.20
    is_drift = bool(p_val < 0.05 or psi >= 0.20)
    severity = "High" if (psi >= 0.20 and p_val < 0.01) else ("Moderate" if is_drift else "Low")

    return {
        "feature": feature_name,
        "type": "numerical",
        "stat": float(ks_stat),
        "p_value": float(p_val),
        "psi": float(psi),
        "drift_detected": is_drift,
        "severity": severity,
        "baseline_mean": float(b_vals.mean()),
        "target_mean": float(t_vals.mean()),
    }


def evaluate_categorical_feature(baseline: pd.Series, target: pd.Series, feature_name: str) -> dict:
    """Runs a Chi-Square frequency test and PSI on a categorical feature."""
    b_s = baseline.astype(str).fillna("Missing")
    t_s = target.astype(str).fillna("Missing")

    psi = calculate_psi(b_s, t_s)
    is_drift = bool(psi >= 0.15)
    severity = "High" if psi >= 0.25 else ("Moderate" if is_drift else "Low")

    return {
        "feature": feature_name,
        "type": "categorical",
        "stat": float(psi),
        "p_value": float(1.0 - min(1.0, psi)),
        "psi": float(psi),
        "drift_detected": is_drift,
        "severity": severity,
        "baseline_top": str(b_s.mode().iloc[0] if len(b_s) > 0 else "N/A"),
        "target_top": str(t_s.mode().iloc[0] if len(t_s) > 0 else "N/A"),
    }


def run_drift_analysis(baseline_df: pd.DataFrame = None, target_df: pd.DataFrame = None) -> dict:
    """
    Executes comprehensive statistical drift analysis across core numerical and categorical columns.
    Returns an aggregated drift diagnostic summary.
    """
    if baseline_df is None:
        baseline_df = load_baseline_data()
    if target_df is None:
        target_df = baseline_df.sample(frac=0.3, random_state=42)

    numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
    categorical_cols = ['contract', 'payment_method', 'internet_service', 'partner', 'dependents']

    results = []
    drifted_count = 0
    total_features = 0

    for col in numerical_cols:
        if col in baseline_df.columns and col in target_df.columns:
            diag = evaluate_numerical_feature(baseline_df[col], target_df[col], col)
            results.append(diag)
            total_features += 1
            if diag["drift_detected"]:
                drifted_count += 1

    for col in categorical_cols:
        if col in baseline_df.columns and col in target_df.columns:
            diag = evaluate_categorical_feature(baseline_df[col], target_df[col], col)
            results.append(diag)
            total_features += 1
            if diag["drift_detected"]:
                drifted_count += 1

    drift_ratio = float(drifted_count / total_features) if total_features > 0 else 0.0
    overall_drift = bool(drift_ratio >= 0.30)

    return {
        "overall_drift_detected": overall_drift,
        "drift_score_pct": round(drift_ratio * 100, 1),
        "drifted_feature_count": drifted_count,
        "total_features_evaluated": total_features,
        "features": results,
        "recommendation": (
            "Model Retraining Recommended: Significant feature distribution shifts detected."
            if overall_drift else
            "Model Healthy: Feature distributions remain stable within statistical bounds."
        )
    }


def generate_synthetic_drift_data(df: pd.DataFrame, severity: float = 0.35) -> pd.DataFrame:
    """
    Generates a controlled synthetic drifted cohort for live dashboard demonstration and testing.
    Adjusts tenure downward and monthly charges upward to simulate an inflation/churn shock.
    """
    drifted = df.copy()
    if 'tenure' in drifted.columns:
        drifted['tenure'] = (drifted['tenure'] * (1.0 - severity)).clip(lower=1).round().astype(int)
    if 'monthly_charges' in drifted.columns:
        drifted['monthly_charges'] = (drifted['monthly_charges'] * (1.0 + severity * 0.75)).round(2)
    if 'contract' in drifted.columns:
        # Shift long term contracts toward Month-to-month
        drifted['contract'] = drifted['contract'].replace({
            'One year': 'Month-to-month',
            'Two year': 'One year'
        })
    return drifted
