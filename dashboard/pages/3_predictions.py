import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
EXCEL_PATH   = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

st.set_page_config(page_title="Predictions", layout="wide")
st.title("Churn & CLV Predictions")
st.markdown("Run real-time inference on customer data using trained XGBoost models.")

# ── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import json
    churn_model  = joblib.load(os.path.join(MODELS_DIR, 'churn_xgboost_model.pkl'))
    churn_scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    clv_model    = joblib.load(os.path.join(MODELS_DIR, 'clv_xgboost_model.pkl'))
    clv_scaler   = joblib.load(os.path.join(MODELS_DIR, 'clv_scaler.pkl'))
    
    try:
        with open(os.path.join(MODELS_DIR, 'churn_threshold.json'), 'r') as f:
            churn_threshold = json.load(f)['threshold']
    except Exception:
        churn_threshold = 0.5
        
    return churn_model, churn_scaler, clv_model, clv_scaler, churn_threshold

@st.cache_resource
def get_shap_explainer(_model):
    """Build a cached TreeExplainer for the churn model (or base XGBoost if stacked)."""
    try:
        import shap
        if hasattr(_model, 'named_estimators_') and 'xgb' in _model.named_estimators_:
            base_tree = _model.named_estimators_['xgb']
        else:
            base_tree = _model
        return shap.TreeExplainer(base_tree)
    except Exception:
        return None

# ── Load & preprocess raw data (cached) ──────────────────────────────────────
@st.cache_data(ttl=600)
def load_and_preprocess():
    """
    Load data from DB; fall back to Excel when DB is unavailable.
    Returns (raw_df, X_churn, num_cols_churn, X_clv, num_cols_clv, source).
    """
    # ── Try database first ───────────────────────────────────────────────────
    try:
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from database.connection import engine
        raw = pd.read_sql("SELECT * FROM customers", engine)
        source = "database"
    except Exception:
        raw = pd.read_excel(EXCEL_PATH)
        raw.columns = (
            raw.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_', regex=False)
        )
        rename_map = {
            'customerid':    'customer_id',
            'churn_label':   'churn',
            'tenure_months': 'tenure',
        }
        raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns}, inplace=True)
        source = "local Excel file"

    raw['total_charges'] = pd.to_numeric(raw['total_charges'], errors='coerce').fillna(0)

    # ── Feature engineering (mirrors ml/data_preprocessing.py) ───────────────
    def _engineer(df):
        df = df.copy()

        def map_tenure(t):
            if t <= 12:  return '0_1_year'
            elif t <= 24: return '1_2_years'
            elif t <= 36: return '2_3_years'
            elif t <= 48: return '3_4_years'
            elif t <= 60: return '4_5_years'
            else:         return '5_plus_years'

        df['tenure_group'] = df['tenure'].apply(map_tenure)

        services = ['online_security', 'online_backup', 'device_protection',
                    'tech_support', 'streaming_tv', 'streaming_movies']
        df['total_additional_services'] = sum(
            (df[s] == 'Yes').astype(int) for s in services if s in df.columns
        )
        df['avg_monthly_charge'] = df['total_charges'] / (df['tenure'] + 1)
        df['charge_difference']  = df['monthly_charges'] - df['avg_monthly_charge']

        if 'contract' in df.columns:
            contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
            df['contract_risk_score'] = df['contract'].map(contract_risk).fillna(1).astype(int)

        if 'senior_citizen' in df.columns and 'contract' in df.columns:
            is_senior = df['senior_citizen'].astype(str).isin(['1', 'Yes', 'True', '1.0'])
            is_mtm    = df['contract'] == 'Month-to-month'
            df['senior_no_contract'] = (is_senior & is_mtm).astype(int)

        if 'monthly_charges' in df.columns:
            high_threshold = df['monthly_charges'].median()
            no_security = df.get('online_security', pd.Series('No', index=df.index)) == 'No'
            no_support  = df.get('tech_support',    pd.Series('No', index=df.index)) == 'No'
            df['high_charge_no_support'] = (
                (df['monthly_charges'] > high_threshold) & no_security & no_support
            ).astype(int)

        if 'payment_method' in df.columns:
            payment_risk = {
                'Electronic check':            2,
                'Mailed check':                1,
                'Bank transfer (automatic)':   0,
                'Credit card (automatic)':     0,
            }
            df['payment_risk_score'] = df['payment_method'].map(payment_risk).fillna(1).astype(int)

        df['tenure_x_charges'] = df['tenure'] * df['monthly_charges']
        return df

    raw = _engineer(raw)

    # ── Helper: encode binary/categorical columns ─────────────────────────────
    def _encode(df):
        """Encode binary cols to int, then one-hot encode remaining object cols."""
        df = df.copy()
        binary_map = {
            'gender':           lambda s: (s == 'Female').astype(int),
            'senior_citizen':   lambda s: s.map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int),
            'partner':          lambda s: (s == 'Yes').astype(int),
            'dependents':       lambda s: (s == 'Yes').astype(int),
            'phone_service':    lambda s: (s == 'Yes').astype(int),
            'paperless_billing':lambda s: (s == 'Yes').astype(int),
        }
        for col, fn in binary_map.items():
            if col in df.columns:
                df[col] = fn(df[col])

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df.columns = [c.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for c in df.columns]
        return df

    # ── Build churn feature matrix X_churn ───────────────────────────────────
    churn_drop = [
        'customer_id', 'customerid', 'id', 'created_at',
        'churn', 'churn_value', 'cltv', 'churn_reason',
        'predicted_churn', 'predicted_clv', 'segment',
        'lat_long', 'latitude', 'longitude', 'city', 'state',
        'country', 'zip_code', 'count',
    ]
    X_churn = raw.drop(columns=[c for c in churn_drop if c in raw.columns], errors='ignore').copy()
    if 'churn_score' in X_churn.columns:
        X_churn['churn_score'] = pd.to_numeric(X_churn['churn_score'], errors='coerce').fillna(50.0)
    else:
        X_churn['churn_score'] = 50.0
    X_churn = _encode(X_churn)

    # Numeric columns the churn scaler was trained on
    num_cols_churn = [
        'tenure', 'monthly_charges', 'total_charges',
        'total_additional_services', 'avg_monthly_charge', 'charge_difference',
        'contract_risk_score', 'tenure_x_charges', 'churn_score',
    ]
    num_cols_churn = [c for c in num_cols_churn if c in X_churn.columns]

    # ── Build CLV feature matrix X_clv ───────────────────────────────────────
    clv_drop = churn_drop + ['total_charges', 'avg_monthly_charge', 'charge_difference']
    X_clv = raw.drop(columns=[c for c in clv_drop if c in raw.columns], errors='ignore').copy()
    X_clv = _encode(X_clv)

    # Numeric columns the CLV scaler was trained on
    num_cols_clv = [c for c in [
        'tenure', 'monthly_charges', 'total_additional_services',
    ] if c in X_clv.columns]

    return raw, X_churn, num_cols_churn, X_clv, num_cols_clv, source


# ── Initialise ───────────────────────────────────────────────────────────────
try:
    churn_model, churn_scaler, clv_model, clv_scaler, churn_threshold = load_models()
    models_ok = True
except Exception as e:
    st.error(f"Could not load models: {e}")
    models_ok = False

try:
    with st.spinner("Loading customer data..."):
        raw_df, X_churn, num_cols_churn, X_clv, num_cols_clv, data_source = load_and_preprocess()

    if data_source != "database":
        st.info(f"Data loaded from **{data_source}** (DB unavailable).")

    # Map customer_id → row index for fast lookup
    id_col = 'customer_id' if 'customer_id' in raw_df.columns else raw_df.columns[0]
    customer_ids = raw_df[id_col].astype(str).tolist()
    data_ok = True

except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.exception(e)
    data_ok = False
    customer_ids = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Select Customer")
customer_id = st.sidebar.selectbox(
    "Choose a Customer ID for Prediction:",
    [""] + customer_ids,
    help="Select a customer to run predictions on."
)
custom_id = st.sidebar.text_input("Or enter Customer ID manually:")
if custom_id.strip():
    customer_id = custom_id.strip()

# ── Main panel ────────────────────────────────────────────────────────────────
if not customer_id:
    st.info("Please select or enter a Customer ID in the sidebar to view predictions.")
elif not (models_ok and data_ok):
    st.warning("Cannot run predictions — models or data failed to load.")
else:
    st.subheader(f"Results for Customer: `{customer_id}`")

    # Locate the row
    idx_list = raw_df.index[raw_df[id_col].astype(str) == customer_id].tolist()
    if not idx_list:
        st.error(f"Customer `{customer_id}` not found in the dataset.")
    else:
        row_idx = idx_list[0]

        # Show a quick profile of this customer
        with st.expander("Customer Profile", expanded=False):
            profile_cols = [c for c in ['gender', 'senior_citizen', 'partner', 'dependents',
                                         'tenure', 'contract', 'monthly_charges',
                                         'total_charges', 'churn_score', 'internet_service', 'churn']
                            if c in raw_df.columns]
            st.dataframe(raw_df.loc[[row_idx], profile_cols])

        col1, col2 = st.columns(2)

        # ── Churn Prediction ──────────────────────────────────────────────────
        with col1:
            st.markdown("### Churn Prediction")
            if st.button("Predict Churn Risk", type="primary", key="btn_churn"):
                with st.spinner("Running XGBoost Churn Classifier..."):
                    try:
                        # Reindex to exactly match model's expected features
                        model_features = list(churn_model.feature_names_in_)
                        X_row = X_churn.iloc[[row_idx]].reindex(columns=model_features, fill_value=0).copy()

                        # Scale only the numeric cols that are present after reindex
                        scale_cols = [c for c in num_cols_churn if c in model_features]
                        X_row[scale_cols] = churn_scaler.transform(X_row[scale_cols])

                        probability = float(churn_model.predict_proba(X_row)[0][1])
                        prob_pct    = probability * 100

                        st.metric("Churn Probability", f"{prob_pct:.1f}%")
                        st.progress(probability)

                        st.caption(f"*(Decision Threshold: {churn_threshold:.3f})*")

                        if probability >= (churn_threshold * 1.5):
                            st.error("**Risk Level: High** — Immediate retention action required!")
                        elif probability >= churn_threshold:
                            st.warning("**Risk Level: Medium** — Monitor this customer closely.")
                        else:
                            st.success("**Risk Level: Low** — Customer is likely to stay.")

                        # ── SHAP Explainability Panel ─────────────────────────
                        with st.expander("Explain this Prediction (SHAP)", expanded=False):
                            with st.spinner("Computing SHAP values..."):
                                try:
                                    import shap
                                    explainer = get_shap_explainer(churn_model)
                                    if explainer is None:
                                        st.warning("SHAP library not installed. Run: `pip install shap`")
                                    else:
                                        shap_vals   = explainer.shap_values(X_row)
                                        shap_series = pd.Series(shap_vals[0], index=X_row.columns)

                                        # Top 10 by absolute impact
                                        top10      = shap_series.abs().nlargest(10).index
                                        shap_top   = shap_series[top10].sort_values()

                                        colors = [
                                            '#ef5350' if v > 0 else '#42a5f5'
                                            for v in shap_top.values
                                        ]

                                        fig_shap = go.Figure(go.Bar(
                                            x=shap_top.values,
                                            y=[f.replace('_', ' ').title() for f in shap_top.index],
                                            orientation='h',
                                            marker_color=colors,
                                            hovertemplate='%{y}: %{x:.4f}<extra></extra>',
                                        ))
                                        fig_shap.update_layout(
                                            title="Top 10 Features Driving This Prediction",
                                            xaxis_title="SHAP Value (impact on churn log-odds)",
                                            yaxis_title="",
                                            height=400,
                                            margin=dict(l=10, r=20, t=50, b=10),
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            paper_bgcolor='rgba(0,0,0,0)',
                                        )
                                        st.plotly_chart(fig_shap, width='stretch')
                                        st.caption(
                                            "**Red bars** = features pushing toward churn  |  "
                                            "**Blue bars** = features pushing toward staying"
                                        )
                                except ImportError:
                                    st.warning(
                                        "SHAP is not installed in this environment. "
                                        "Run `pip install shap` and restart the app."
                                    )
                                except Exception as shap_err:
                                    st.error(f"SHAP computation failed: {shap_err}")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.exception(e)

        # ── CLV Prediction ────────────────────────────────────────────────────
        with col2:
            st.markdown("### Customer Lifetime Value")
            if st.button("Predict CLV", type="primary", key="btn_clv"):
                with st.spinner("Running XGBoost CLV Regressor..."):
                    try:
                        # Reindex to exactly match model's expected features
                        clv_features = list(clv_model.feature_names_in_)
                        X_row_clv = X_clv.iloc[[row_idx]].reindex(columns=clv_features, fill_value=0).copy()

                        # Scale only the numeric cols that are present after reindex
                        scale_cols_clv = [c for c in num_cols_clv if c in clv_features]
                        X_row_clv[scale_cols_clv] = clv_scaler.transform(X_row_clv[scale_cols_clv])

                        predicted_clv = float(clv_model.predict(X_row_clv)[0])

                        st.metric("Estimated Lifetime Value", f"${predicted_clv:,.2f}")
                        st.success("Prediction generated successfully.")

                        # Contextual gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=predicted_clv,
                            title={'text': "CLV ($)"},
                            gauge={
                                'axis': {'range': [0, 8000]},
                                'bar':  {'color': "#42a5f5"},
                                'steps': [
                                    {'range': [0,    2000], 'color': '#ffcdd2'},
                                    {'range': [2000, 5000], 'color': '#fff9c4'},
                                    {'range': [5000, 8000], 'color': '#c8e6c9'},
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 3},
                                    'thickness': 0.75, 'value': 5000
                                },
                            }
                        ))
                        fig_gauge.update_layout(height=260, margin=dict(t=30, b=10))
                        st.plotly_chart(fig_gauge, width='stretch')

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.exception(e)
