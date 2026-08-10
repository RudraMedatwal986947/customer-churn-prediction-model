import streamlit as st
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
EXCEL_PATH   = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
st.title("🔮 Churn & CLV Predictions")
st.markdown("Run real-time inference on customer data using trained XGBoost models.")

# ── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    churn_model  = joblib.load(os.path.join(MODELS_DIR, 'churn_xgboost_model.pkl'))
    churn_scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    clv_model    = joblib.load(os.path.join(MODELS_DIR, 'clv_xgboost_model.pkl'))
    clv_scaler   = joblib.load(os.path.join(MODELS_DIR, 'clv_scaler.pkl'))
    return churn_model, churn_scaler, clv_model, clv_scaler

# ── Load & preprocess raw data (cached) ──────────────────────────────────────
@st.cache_data(ttl=600)
def load_and_preprocess():
    """
    Load data from DB; fall back to Excel when DB is unavailable.
    Returns (raw_df, X_churn, X_clv) where:
      - raw_df    has normalised column names + customer_id
      - X_churn   is the 38-feature matrix for the churn model
      - X_clv     is the feature matrix for the CLV model
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
        return df

    raw = _engineer(raw)

    # ── Build churn feature matrix X_churn ───────────────────────────────────
    churn_drop = ['customer_id', 'customerid', 'id', 'created_at',
                  'churn', 'churn_value', 'churn_score', 'cltv', 'churn_reason',
                  'predicted_churn', 'predicted_clv', 'segment',
                  'lat_long', 'latitude', 'longitude', 'city', 'state',
                  'country', 'zip_code', 'count']
    X_churn = raw.drop(columns=[c for c in churn_drop if c in raw.columns], errors='ignore').copy()

    # senior_citizen from Excel is 'Yes'/'No' string — encode to int BEFORE get_dummies
    # so it stays a plain 0/1 column (matching how the model was trained via the DB)
    binary_cols = ['gender', 'senior_citizen', 'partner', 'dependents',
                   'phone_service', 'paperless_billing']
    for col in binary_cols:
        if col in X_churn.columns:
            if col == 'gender':
                X_churn[col] = (X_churn[col] == 'Female').astype(int)
            elif col == 'senior_citizen':
                # DB stores as int already; Excel may store as 'Yes'/'No' or 0/1
                if X_churn[col].dtype == object:
                    X_churn[col] = (X_churn[col] == 'Yes').astype(int)
                else:
                    X_churn[col] = X_churn[col].astype(int)
            else:
                X_churn[col] = (X_churn[col] == 'Yes').astype(int)

    cat_cols = X_churn.select_dtypes(include='object').columns.tolist()
    X_churn = pd.get_dummies(X_churn, columns=cat_cols, drop_first=True)

    num_cols_churn = [c for c in ['tenure', 'monthly_charges', 'total_charges',
                                   'total_additional_services', 'avg_monthly_charge',
                                   'charge_difference'] if c in X_churn.columns]

    # ── Build CLV feature matrix X_clv ───────────────────────────────────────
    clv_drop = churn_drop + ['total_charges']
    X_clv = raw.drop(columns=[c for c in clv_drop if c in raw.columns], errors='ignore').copy()

    for col in binary_cols:
        if col in X_clv.columns:
            if col == 'gender':
                X_clv[col] = (X_clv[col] == 'Female').astype(int)
            elif col == 'senior_citizen':
                if X_clv[col].dtype == object:
                    X_clv[col] = (X_clv[col] == 'Yes').astype(int)
                else:
                    X_clv[col] = X_clv[col].astype(int)
            else:
                X_clv[col] = (X_clv[col] == 'Yes').astype(int)

    cat_cols_clv = X_clv.select_dtypes(include='object').columns.tolist()
    X_clv = pd.get_dummies(X_clv, columns=cat_cols_clv, drop_first=True)
    num_cols_clv = [c for c in ['tenure', 'monthly_charges', 'total_additional_services']
                    if c in X_clv.columns]

    return raw, X_churn, num_cols_churn, X_clv, num_cols_clv, source


# ── Initialise ───────────────────────────────────────────────────────────────
try:
    churn_model, churn_scaler, clv_model, clv_scaler = load_models()
    models_ok = True
except Exception as e:
    st.error(f"❌ Could not load models: {e}")
    models_ok = False

try:
    with st.spinner("Loading customer data..."):
        raw_df, X_churn, num_cols_churn, X_clv, num_cols_clv, data_source = load_and_preprocess()

    if data_source != "database":
        st.info(f"📂 Data loaded from **{data_source}** (DB unavailable).", icon="ℹ️")

    # Map customer_id → row index for fast lookup
    id_col = 'customer_id' if 'customer_id' in raw_df.columns else raw_df.columns[0]
    customer_ids = raw_df[id_col].astype(str).tolist()
    data_ok = True

except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
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
    st.info("👈 Please select or enter a Customer ID in the sidebar to view predictions.")
elif not (models_ok and data_ok):
    st.warning("⚠️ Cannot run predictions — models or data failed to load.")
else:
    st.subheader(f"Results for Customer: `{customer_id}`")

    # Locate the row
    idx_list = raw_df.index[raw_df[id_col].astype(str) == customer_id].tolist()
    if not idx_list:
        st.error(f"Customer `{customer_id}` not found in the dataset.")
    else:
        row_idx = idx_list[0]

        # Show a quick profile of this customer
        with st.expander("📋 Customer Profile", expanded=False):
            profile_cols = [c for c in ['gender', 'senior_citizen', 'partner', 'dependents',
                                         'tenure', 'contract', 'monthly_charges',
                                         'total_charges', 'internet_service', 'churn']
                            if c in raw_df.columns]
            st.dataframe(raw_df.loc[[row_idx], profile_cols])

        col1, col2 = st.columns(2)

        # ── Churn Prediction ──────────────────────────────────────────────────
        with col1:
            st.markdown("### 🚨 Churn Prediction")
            if st.button("Predict Churn Risk", type="primary", key="btn_churn"):
                with st.spinner("Running XGBoost Churn Classifier..."):
                    try:
                        X_row = X_churn.iloc[[row_idx]].copy()
                        # Reindex to exactly match the columns the model was trained on
                        model_features = list(churn_model.feature_names_in_)
                        X_row = X_row.reindex(columns=model_features, fill_value=0)
                        X_row[num_cols_churn] = churn_scaler.transform(X_row[num_cols_churn])

                        probability = float(churn_model.predict_proba(X_row)[0][1])
                        prob_pct    = probability * 100

                        st.metric("Churn Probability", f"{prob_pct:.1f}%")
                        st.progress(probability)

                        if probability > 0.6:
                            st.error(f"**Risk Level: High** — Immediate retention action required!")
                        elif probability > 0.3:
                            st.warning(f"**Risk Level: Medium** — Monitor this customer closely.")
                        else:
                            st.success(f"**Risk Level: Low** — Customer is likely to stay.")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.exception(e)

        # ── CLV Prediction ────────────────────────────────────────────────────
        with col2:
            st.markdown("### 💰 Customer Lifetime Value")
            if st.button("Predict CLV", type="primary", key="btn_clv"):
                with st.spinner("Running XGBoost CLV Regressor..."):
                    try:
                        X_row_clv = X_clv.iloc[[row_idx]].copy()
                        # Reindex to exactly match the columns the model was trained on
                        clv_features = list(clv_model.feature_names_in_)
                        X_row_clv = X_row_clv.reindex(columns=clv_features, fill_value=0)
                        X_row_clv[num_cols_clv] = clv_scaler.transform(X_row_clv[num_cols_clv])

                        predicted_clv = float(clv_model.predict(X_row_clv)[0])

                        st.metric("Estimated Lifetime Value", f"${predicted_clv:,.2f}")
                        st.success("Prediction generated successfully.")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.exception(e)
