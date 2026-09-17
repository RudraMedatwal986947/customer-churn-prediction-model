import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
EXCEL_PATH   = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Analysis")
st.markdown("View K-Means clustering results and segment-level statistics.")

# ── Load KMeans model + scaler (cached) ──────────────────────────────────────
@st.cache_resource
def load_kmeans():
    import joblib
    kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'kmeans_scaler.pkl'))
    return kmeans, scaler

# ── Load & segment data (cached) ─────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_and_segment():
    """
    Try DB first; fall back to Excel.
    Runs KMeans prediction locally using the saved model.
    Returns (df_full, df_summary, source).
    """
    # ── Try database ─────────────────────────────────────────────────────────
    try:
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

    # ── Clean & engineer features ─────────────────────────────────────────────
    raw['total_charges']   = pd.to_numeric(raw['total_charges'],   errors='coerce').fillna(0)
    raw['monthly_charges'] = pd.to_numeric(raw['monthly_charges'], errors='coerce').fillna(0)
    raw['tenure']          = pd.to_numeric(raw['tenure'],          errors='coerce').fillna(0)

    services = ['online_security', 'online_backup', 'device_protection',
                'tech_support', 'streaming_tv', 'streaming_movies']
    raw['total_additional_services'] = sum(
        (raw[s] == 'Yes').astype(int) for s in services if s in raw.columns
    )

    # ── Predict segments using saved KMeans model ─────────────────────────────
    kmeans, scaler = load_kmeans()
    feature_cols = ['tenure', 'monthly_charges', 'total_charges', 'total_additional_services']
    X = raw[feature_cols].copy()
    X_scaled = scaler.transform(X)
    raw['segment'] = [f"Segment {label}" for label in kmeans.predict(X_scaled)]

    # ── Build summary table ───────────────────────────────────────────────────
    summary = (
        raw.groupby('segment')
        .agg(
            customer_count  = ('customer_id', 'count'),
            avg_tenure      = ('tenure', 'mean'),
            avg_monthly     = ('monthly_charges', 'mean'),
            avg_total       = ('total_charges', 'mean'),
            avg_services    = ('total_additional_services', 'mean'),
        )
        .reset_index()
        .sort_values('segment')
    )

    # Churn rate per segment if available
    if 'churn' in raw.columns:
        # Handle both 'Yes'/'No' strings and 0/1 integers
        if pd.api.types.is_numeric_dtype(raw['churn']):
            churn_num = raw['churn']
        else:
            churn_num = (raw['churn'] == 'Yes').astype(int)
        churn_rate = raw.groupby('segment')['churn'].apply(
            lambda s: (s == 'Yes').mean() * 100 if not pd.api.types.is_numeric_dtype(s)
                      else s.mean() * 100
        ).reset_index(name='churn_rate_pct')
        summary = summary.merge(churn_rate, on='segment', how='left')

    return raw, summary, source


# ── Main ─────────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading and segmenting customer data..."):
        df_full, df_summary, data_source = load_and_segment()

    if data_source != "database":
        st.info(f"Data loaded from **{data_source}** (DB unavailable).")

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Segment Summary Statistics")
    fmt = {
        "avg_tenure":   "{:.1f} mo",
        "avg_monthly":  "${:.2f}",
        "avg_total":    "${:.2f}",
        "avg_services": "{:.1f}",
    }
    if "churn_rate_pct" in df_summary.columns:
        fmt["churn_rate_pct"] = "{:.1f}%"

    st.dataframe(df_summary.style.format(fmt), width='stretch')

    st.markdown("---")

    # ── Cluster metrics row ───────────────────────────────────────────────────
    n_segs = df_summary['segment'].nunique()
    cols = st.columns(n_segs)
    for i, (_, row) in enumerate(df_summary.iterrows()):
        with cols[i]:
            st.metric(row['segment'], f"{int(row['customer_count']):,} customers")
            st.caption(f"Avg tenure: {row['avg_tenure']:.0f} mo | Avg monthly: ${row['avg_monthly']:.0f}")

    st.markdown("---")

    # ── Visualizations ────────────────────────────────────────────────────────
    st.subheader("Cluster Visualization")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        sample = df_full.sample(min(2000, len(df_full)), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="tenure", y="monthly_charges", color="segment",
            title="Tenure vs Monthly Charges by Segment",
            color_discrete_map={
                "Segment 0": "#42a5f5",
                "Segment 1": "#66bb6a",
                "Segment 2": "#ffa726",
                "Segment 3": "#ef5350"
            },
            opacity=0.7,
            labels={"tenure": "Tenure (Months)", "monthly_charges": "Monthly Charges ($)", "segment": "Cluster Segment"}
        )
        fig_scatter.update_layout(height=420, margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig_scatter, width='stretch')

    with row1_col2:
        fig_box = px.box(
            df_full, x="segment", y="total_charges", color="segment",
            title="Total Charges Distribution by Segment",
        )
        st.plotly_chart(fig_box, width='stretch')

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig_services = px.bar(
            df_summary, x="segment", y="avg_services", color="segment",
            title="Average Additional Services per Segment",
        )
        st.plotly_chart(fig_services, width='stretch')

    with row2_col2:
        if "churn_rate_pct" in df_summary.columns:
            fig_churn = px.bar(
                df_summary, x="segment", y="churn_rate_pct", color="segment",
                title="Churn Rate (%) by Segment",
                labels={"churn_rate_pct": "Churn Rate (%)"},
            )
            st.plotly_chart(fig_churn, width='stretch')
        else:
            st.info("Churn data not available for segment churn rate chart.")

    st.markdown("---")
    st.subheader("Raw Data Sample (with Segments)")
    display_cols = [c for c in ['customer_id', 'tenure', 'monthly_charges',
                                 'total_charges', 'total_additional_services',
                                 'segment', 'churn'] if c in df_full.columns]
    st.dataframe(df_full[display_cols].head(100))

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)
