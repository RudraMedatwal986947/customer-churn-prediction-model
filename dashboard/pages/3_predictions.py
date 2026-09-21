import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
import sys
import warnings

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(__file__)
DASHBOARD_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(DASHBOARD_DIR, '..'))

if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui_components import (
    apply_custom_css,
    render_page_header,
    render_kpi,
    apply_plotly_theme,
)

st.set_page_config(page_title="Predictions & Inference", layout="wide")
apply_custom_css()

render_page_header(
    title="Customer Churn & CLV Predictions",
    subtitle="Real-time inference engine powered by tuned XGBoost models with SHAP feature attribution.",
    category="Inference Engine"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

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
        churn_threshold = 0.440
        
    return churn_model, churn_scaler, clv_model, clv_scaler, churn_threshold

@st.cache_resource
def get_shap_explainer(_model):
    """Build a cached TreeExplainer for the churn model."""
    try:
        import shap
        if hasattr(_model, 'named_estimators_') and 'xgb' in _model.named_estimators_:
            base_tree = _model.named_estimators_['xgb']
        else:
            base_tree = _model
        return shap.TreeExplainer(base_tree)
    except Exception:
        return None

@st.cache_data(ttl=600)
def load_and_preprocess():
    """Load data from DB; fall back to Excel when DB is unavailable."""
    try:
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

    # Feature engineering (mirrors ml/data_preprocessing.py)
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

    def _encode(df):
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

    num_cols_churn = [
        'tenure', 'monthly_charges', 'total_charges',
        'total_additional_services', 'avg_monthly_charge', 'charge_difference',
        'contract_risk_score', 'tenure_x_charges', 'churn_score',
    ]
    num_cols_churn = [c for c in num_cols_churn if c in X_churn.columns]

    clv_drop = churn_drop + ['total_charges', 'avg_monthly_charge', 'charge_difference']
    X_clv = raw.drop(columns=[c for c in clv_drop if c in raw.columns], errors='ignore').copy()
    X_clv = _encode(X_clv)

    num_cols_clv = [c for c in [
        'tenure', 'monthly_charges', 'total_additional_services',
    ] if c in X_clv.columns]

    return raw, X_churn, num_cols_churn, X_clv, num_cols_clv, source

# ── Load Model & Data ────────────────────────────────────────────────────────
try:
    churn_model, churn_scaler, clv_model, clv_scaler, churn_threshold = load_models()
    models_ok = True
except Exception as e:
    st.error(f"Could not load models: {e}")
    models_ok = False

try:
    with st.spinner("Loading cohort features..."):
        raw_df, X_churn, num_cols_churn, X_clv, num_cols_clv, data_source = load_and_preprocess()

    if data_source != "database":
        st.info(f"Data loaded from **{data_source}** (DB unavailable).")

    id_col = 'customer_id' if 'customer_id' in raw_df.columns else raw_df.columns[0]
    customer_ids = raw_df[id_col].astype(str).tolist()
    data_ok = True
except Exception as e:
    st.error(f"Failed to load data: {e}")
    data_ok = False
    customer_ids = []

# ── Sidebar Selection ────────────────────────────────────────────────────────
st.sidebar.header("Customer Selection")

# Identify sample customers for 1-click quick-testing
sample_high_risk = "7590-VHVEG"  # classic month-to-month high churner
sample_low_risk  = "7055-JCGNI"  # high tenure, two-year contract

if "selected_cust_id" not in st.session_state:
    st.session_state["selected_cust_id"] = customer_ids[0] if customer_ids else ""

st.sidebar.markdown("**Quick Preset Profiles:**")
q_col1, q_col2 = st.sidebar.columns(2)
with q_col1:
    if st.button("High Risk", use_container_width=True):
        if sample_high_risk in customer_ids:
            st.session_state["selected_cust_id"] = sample_high_risk
        else:
            st.session_state["selected_cust_id"] = customer_ids[0]
with q_col2:
    if st.button("Low Risk", use_container_width=True):
        if sample_low_risk in customer_ids:
            st.session_state["selected_cust_id"] = sample_low_risk
        else:
            st.session_state["selected_cust_id"] = customer_ids[-1]

default_idx = 0
if st.session_state["selected_cust_id"] in customer_ids:
    default_idx = customer_ids.index(st.session_state["selected_cust_id"])

chosen_id = st.sidebar.selectbox(
    "Choose Customer ID:",
    customer_ids,
    index=default_idx
)
manual_id = st.sidebar.text_input("Or enter Customer ID manually:")
if manual_id.strip():
    chosen_id = manual_id.strip()

st.session_state["selected_cust_id"] = chosen_id

# ── Main Content ─────────────────────────────────────────────────────────────
if not chosen_id:
    st.info("Please select or enter a Customer ID in the sidebar to view predictions.")
elif not (models_ok and data_ok):
    st.warning("Cannot run predictions — models or data failed to load.")
else:
    idx_list = raw_df.index[raw_df[id_col].astype(str) == chosen_id].tolist()
    if not idx_list:
        st.error(f"Customer `{chosen_id}` not found in the dataset.")
    else:
        row_idx = idx_list[0]
        cust_row = raw_df.iloc[row_idx]

        # ── Customer Profile Summary Bar ─────────────────────────────────────
        st.markdown(f"### Customer Dossier: `{chosen_id}`")
        
        prof_c1, prof_c2, prof_c3, prof_c4, prof_c5 = st.columns(5)
        with prof_c1:
            st.markdown(f"**Tenure:** {cust_row.get('tenure', 'N/A')} Months")
            st.markdown(f"**Contract:** {cust_row.get('contract', 'N/A')}")
        with prof_c2:
            st.markdown(f"**Monthly Charges:** ${cust_row.get('monthly_charges', 0):.2f}")
            st.markdown(f"**Total Charges:** ${cust_row.get('total_charges', 0):.2f}")
        with prof_c3:
            st.markdown(f"**Internet:** {cust_row.get('internet_service', 'N/A')}")
            st.markdown(f"**Tech Support:** {cust_row.get('tech_support', 'N/A')}")
        with prof_c4:
            st.markdown(f"**Security:** {cust_row.get('online_security', 'N/A')}")
            st.markdown(f"**Payment:** {cust_row.get('payment_method', 'N/A')}")
        with prof_c5:
            hist_churn = cust_row.get('churn', 'No')
            status_style = "badge-red" if hist_churn == 'Yes' else "badge-green"
            st.markdown(f"**Historical Status:**")
            st.markdown(f"<span class='badge {status_style}'>Churn: {hist_churn}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Two-Column Inference Grid ────────────────────────────────────────
        col_churn, col_clv = st.columns(2)

        # ── 1. Churn Classifier Column ───────────────────────────────────────
        with col_churn:
            st.markdown("### Churn Risk Intelligence")
            try:
                model_features = list(churn_model.feature_names_in_)
                X_row = X_churn.iloc[[row_idx]].reindex(columns=model_features, fill_value=0).copy()
                scale_cols = [c for c in num_cols_churn if c in model_features]
                X_row[scale_cols] = churn_scaler.transform(X_row[scale_cols])

                probability = float(churn_model.predict_proba(X_row)[0][1])
                prob_pct = probability * 100

                # Risk styling
                if probability >= 0.60:
                    risk_label = "HIGH RISK"
                    risk_color = "#EF4444"
                    action_advice = "URGENT ACTION: Customer displays strong attrition signals. Immediate outreach recommended with an exclusive renewal offer or loyalty credit."
                elif probability >= churn_threshold:
                    risk_label = "MODERATE RISK"
                    risk_color = "#F59E0B"
                    action_advice = "MONITOR CLOSELY: Churn probability exceeds the optimal threshold (0.440). Recommend customer success check-in and review of service satisfaction."
                else:
                    risk_label = "LOW RISK (STABLE)"
                    risk_color = "#10B981"
                    action_advice = "HEALTHY ACCOUNT: Low probability of churn. Prime candidate for cross-selling advanced security bundles or multi-year contract renewals."

                render_kpi(
                    label=f"Predicted Churn Probability ({risk_label})",
                    value=f"{prob_pct:.1f}%",
                    caption=f"Optimal Decision Cutoff: {churn_threshold:.3f}",
                    accent_color=risk_color
                )

                st.progress(probability)

                st.markdown(f"""
                <div style='background-color: #F8FAFC; border-left: 4px solid {risk_color}; padding: 0.85rem 1rem; border-radius: 6px; font-size: 0.85rem; color: #334155; line-height: 1.4; margin: 1rem 0;'>
                    {action_advice}
                </div>
                """, unsafe_allow_html=True)

                # ── SHAP Explainability ───────────────────────────────────────
                with st.expander("Explain this Prediction (SHAP Drivers)", expanded=False):
                    try:
                        import shap
                        explainer = get_shap_explainer(churn_model)
                        if explainer is None:
                            st.info("SHAP explainability engine is unavailable in this environment.")
                        else:
                            shap_vals = explainer.shap_values(X_row)
                            shap_series = pd.Series(shap_vals[0], index=X_row.columns)

                            top10 = shap_series.abs().nlargest(10).index
                            shap_top = shap_series[top10].sort_values()

                            bar_colors = ['#EF4444' if v > 0 else '#3B82F6' for v in shap_top.values]

                            fig_shap = go.Figure(go.Bar(
                                x=shap_top.values,
                                y=[f.replace('_', ' ').title() for f in shap_top.index],
                                orientation='h',
                                marker=dict(color=bar_colors, line=dict(width=0.5, color='#CBD5E1')),
                                hovertemplate='%{y}: %{x:.4f}<extra></extra>',
                            ))
                            fig_shap.update_layout(
                                title="Top 10 Feature Drivers (SHAP Log-Odds Impact)",
                                xaxis_title="SHAP Value (Positive = Pushes to Churn, Negative = Retains)",
                                yaxis_title="",
                            )
                            fig_shap = apply_plotly_theme(fig_shap, height=360)
                            st.plotly_chart(fig_shap, width='stretch')

                            st.markdown(
                                "<span style='color: #EF4444; font-weight: 600;'>Red bars</span> = factors increasing churn risk  |  "
                                "<span style='color: #3B82F6; font-weight: 600;'>Blue bars</span> = factors promoting customer retention",
                                unsafe_allow_html=True
                            )
                    except Exception as shap_e:
                        st.info(f"SHAP explanation: {shap_e}")

            except Exception as ex:
                st.error(f"Inference error: {ex}")

        # ── 2. CLV Regressor Column ──────────────────────────────────────────
        with col_clv:
            st.markdown("### Customer Lifetime Value (CLV)")
            try:
                clv_features = list(clv_model.feature_names_in_)
                X_row_clv = X_clv.iloc[[row_idx]].reindex(columns=clv_features, fill_value=0).copy()
                scale_cols_clv = [c for c in num_cols_clv if c in clv_features]
                X_row_clv[scale_cols_clv] = clv_scaler.transform(X_row_clv[scale_cols_clv])

                predicted_clv = float(clv_model.predict(X_row_clv)[0])

                # Value tier
                if predicted_clv >= 5000:
                    tier_name = "Platinum VIP Tier"
                    tier_color = "#10B981"
                    tier_advice = "Top 15% value account. Provide priority technical support, assigned account manager, and premier relationship terms."
                elif predicted_clv >= 3000:
                    tier_name = "Gold Tier"
                    tier_color = "#3B82F6"
                    tier_advice = "High-performing revenue generator. Focus on loyalty retention and premium hardware / feature upsells."
                elif predicted_clv >= 1500:
                    tier_name = "Silver Tier"
                    tier_color = "#6366F1"
                    tier_advice = "Established mid-tier account. Cross-sell complementary services to deepen customer lock-in."
                else:
                    tier_name = "Bronze Tier"
                    tier_color = "#F59E0B"
                    tier_advice = "Early-stage or entry tier. Incentivize contract extensions to elevate long-term lifetime yield."

                render_kpi(
                    label=f"Estimated Lifetime Value ({tier_name})",
                    value=f"${predicted_clv:,.2f}",
                    caption=f"Historical Baseline: ${cust_row.get('total_charges', 0):,.2f}",
                    accent_color=tier_color
                )

                # Contextual Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_clv,
                    number={'prefix': "$", 'valueformat': ",.0f", 'font': {'size': 26, 'color': '#0F172A'}},
                    gauge={
                        'axis': {'range': [0, 8500], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                        'bar':  {'color': tier_color},
                        'steps': [
                            {'range': [0,    2000], 'color': '#F1F5F9'},
                            {'range': [2000, 4500], 'color': '#E2E8F0'},
                            {'range': [4500, 8500], 'color': '#CBD5E1'},
                        ],
                        'threshold': {
                            'line': {'color': "#0F172A", 'width': 3},
                            'thickness': 0.75,
                            'value': 5000
                        },
                    }
                ))
                fig_gauge = apply_plotly_theme(fig_gauge, height=220)
                fig_gauge.update_layout(margin=dict(t=20, b=10, l=30, r=30))
                st.plotly_chart(fig_gauge, width='stretch')

                st.markdown(f"""
                <div style='background-color: #F8FAFC; border-left: 4px solid {tier_color}; padding: 0.85rem 1rem; border-radius: 6px; font-size: 0.85rem; color: #334155; line-height: 1.4;'>
                    {tier_advice}
                </div>
                """, unsafe_allow_html=True)

            except Exception as clv_ex:
                st.error(f"CLV inference error: {clv_ex}")
