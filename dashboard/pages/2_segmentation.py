import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(page_title="Customer Segmentation", layout="wide")
apply_custom_css()

render_page_header(
    title="Customer Segmentation Analysis",
    subtitle="Unsupervised K-Means clustering (k=4) revealing distinct behavioral archetypes, spending tiers, and churn propensities.",
    category="Clustering & Personas"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

@st.cache_resource
def load_kmeans():
    import joblib
    kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'kmeans_scaler.pkl'))
    return kmeans, scaler

@st.cache_data(ttl=600)
def load_and_segment():
    """Try DB first; fallback to Excel. Runs KMeans inference using saved model."""
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

    raw['total_charges']   = pd.to_numeric(raw['total_charges'],   errors='coerce').fillna(0)
    raw['monthly_charges'] = pd.to_numeric(raw['monthly_charges'], errors='coerce').fillna(0)
    raw['tenure']          = pd.to_numeric(raw['tenure'],          errors='coerce').fillna(0)

    services = ['online_security', 'online_backup', 'device_protection',
                'tech_support', 'streaming_tv', 'streaming_movies']
    raw['total_additional_services'] = sum(
        (raw[s] == 'Yes').astype(int) for s in services if s in raw.columns
    )

    kmeans, scaler = load_kmeans()
    feature_cols = ['tenure', 'monthly_charges', 'total_charges', 'total_additional_services']
    X = raw[feature_cols].copy()
    X_scaled = scaler.transform(X)
    raw['segment'] = [f"Segment {label}" for label in kmeans.predict(X_scaled)]

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

    if 'churn' in raw.columns:
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

try:
    with st.spinner("Clustering customer cohort..."):
        df_full, df_summary, data_source = load_and_segment()

    if data_source != "database":
        st.info(f"Data loaded from **{data_source}** (DB unavailable).")

    # ── Persona Overview Cards ────────────────────────────────────────────────
    st.markdown("### Behavioral Cluster Archetypes")
    
    seg_metadata = {
        "Segment 0": {
            "name": "New / Low Spend",
            "desc": "Short tenure, basic plan, low service count. Highest churn risk.",
            "action": "Immediate onboarding check-in & multi-month discounts.",
            "color": "#3B82F6"
        },
        "Segment 1": {
            "name": "Moderate Retained",
            "desc": "Medium tenure, standard spend, steady usage pattern.",
            "action": "Cross-sell cybersecurity & cloud backup add-ons.",
            "color": "#10B981"
        },
        "Segment 2": {
            "name": "Long-Term Premium",
            "desc": "Longest tenure, highest cumulative spend, multi-service.",
            "action": "VIP loyalty appreciation, priority support, referral perks.",
            "color": "#F59E0B"
        },
        "Segment 3": {
            "name": "High-Spend At-Risk",
            "desc": "High monthly spend with medium tenure. Vulnerable to competitor offers.",
            "action": "Proactive service review & specialized contract renewals.",
            "color": "#EF4444"
        }
    }

    p_cols = st.columns(len(df_summary))
    for i, (_, row) in enumerate(df_summary.iterrows()):
        seg_key = row['segment']
        meta = seg_metadata.get(seg_key, {
            "name": seg_key,
            "desc": "Behavioral cluster cohort",
            "action": "Targeted engagement",
            "color": "#6366F1"
        })
        with p_cols[i]:
            churn_str = f" | Churn: {row['churn_rate_pct']:.1f}%" if "churn_rate_pct" in row else ""
            render_kpi(
                label=f"{seg_key}: {meta['name']}",
                value=f"{int(row['customer_count']):,}",
                caption=f"Tenure: {row['avg_tenure']:.0f} mo | Spend: ${row['avg_monthly']:.0f}/mo{churn_str}",
                accent_color=meta["color"]
            )
            st.caption(f"**Strategy:** {meta['action']}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Summary Statistics Table ──────────────────────────────────────────────
    st.markdown("### Segment Quantitative Profiles")
    fmt = {
        "customer_count": "{:,}",
        "avg_tenure":     "{:.1f} mo",
        "avg_monthly":    "${:.2f}",
        "avg_total":      "${:.2f}",
        "avg_services":   "{:.2f}",
    }
    if "churn_rate_pct" in df_summary.columns:
        fmt["churn_rate_pct"] = "{:.1f}%"

    styled_summary = (
        df_summary.rename(columns={
            "segment": "Segment",
            "customer_count": "Customers",
            "avg_tenure": "Avg Tenure",
            "avg_monthly": "Avg Monthly Spend",
            "avg_total": "Avg Total Spend",
            "avg_services": "Avg Add-on Services",
            "churn_rate_pct": "Churn Rate"
        })
        .style.format({
            "Customers": "{:,}",
            "Avg Tenure": "{:.1f} mo",
            "Avg Monthly Spend": "${:.2f}",
            "Avg Total Spend": "${:.2f}",
            "Avg Add-on Services": "{:.2f}",
            "Churn Rate": "{:.1f}%" if "churn_rate_pct" in df_summary.columns else "{}"
        })
    )
    st.dataframe(styled_summary, width='stretch')

    st.markdown("---")

    # ── Interactive Cluster Scatter Plot & Box Plot ────────────────────────────
    st.markdown("### Cluster Visualizations")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### Tenure vs Monthly Charges Scatter")
        sample = df_full.sample(min(2000, len(df_full)), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="tenure",
            y="monthly_charges",
            color="segment",
            color_discrete_map={
                "Segment 0": "#3B82F6",
                "Segment 1": "#10B981",
                "Segment 2": "#F59E0B",
                "Segment 3": "#EF4444"
            },
            opacity=0.75,
            labels={
                "tenure": "Tenure (Months)",
                "monthly_charges": "Monthly Charges ($)",
                "segment": "Cluster"
            },
            hover_data=["customer_id", "total_charges"]
        )
        fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))
        fig_scatter.update_layout(
            xaxis_title="Tenure in Months",
            yaxis_title="Monthly Charges ($)"
        )
        fig_scatter = apply_plotly_theme(fig_scatter, height=380)
        st.plotly_chart(fig_scatter, width='stretch')

    with row1_col2:
        st.markdown("#### Cumulative Spend (Total Charges) by Segment")
        fig_box = px.box(
            df_full,
            x="segment",
            y="total_charges",
            color="segment",
            color_discrete_map={
                "Segment 0": "#3B82F6",
                "Segment 1": "#10B981",
                "Segment 2": "#F59E0B",
                "Segment 3": "#EF4444"
            },
            labels={"segment": "Cluster", "total_charges": "Total Charges ($)"}
        )
        fig_box.update_layout(
            xaxis_title="Cluster Segment",
            yaxis_title="Total Historical Charges ($)"
        )
        fig_box = apply_plotly_theme(fig_box, height=380)
        st.plotly_chart(fig_box, width='stretch')

    # ── Services and Churn Rate Rows ───────────────────────────────────────────
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### Average Add-On Services Adopted")
        fig_services = px.bar(
            df_summary,
            x="segment",
            y="avg_services",
            color="segment",
            color_discrete_map={
                "Segment 0": "#3B82F6",
                "Segment 1": "#10B981",
                "Segment 2": "#F59E0B",
                "Segment 3": "#EF4444"
            },
            text_auto=".2f",
            labels={"segment": "Cluster", "avg_services": "Avg Services"}
        )
        fig_services.update_traces(textposition='outside')
        fig_services.update_layout(
            xaxis_title="Cluster Segment",
            yaxis_title="Average Add-on Services Count"
        )
        fig_services = apply_plotly_theme(fig_services, height=340)
        st.plotly_chart(fig_services, width='stretch')

    with row2_col2:
        st.markdown("#### Historical Churn Rate (%) by Segment")
        if "churn_rate_pct" in df_summary.columns:
            fig_churn = px.bar(
                df_summary,
                x="segment",
                y="churn_rate_pct",
                color="segment",
                color_discrete_map={
                    "Segment 0": "#3B82F6",
                    "Segment 1": "#10B981",
                    "Segment 2": "#F59E0B",
                    "Segment 3": "#EF4444"
                },
                text_auto=".1f",
                labels={"segment": "Cluster", "churn_rate_pct": "Churn Rate (%)"}
            )
            fig_churn.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            fig_churn.update_layout(
                xaxis_title="Cluster Segment",
                yaxis_title="Observed Churn Rate (%)"
            )
            fig_churn = apply_plotly_theme(fig_churn, height=340)
            st.plotly_chart(fig_churn, width='stretch')
        else:
            st.info("Churn percentage not available.")

    # ── Segment Drill-Down & Data Export ───────────────────────────────────────
    st.markdown("---")
    with st.expander("Filter & Inspect Segment Customer Records", expanded=False):
        chosen_seg = st.selectbox(
            "Select Segment to Inspect:",
            sorted(df_full['segment'].unique())
        )
        seg_slice = df_full[df_full['segment'] == chosen_seg]
        st.markdown(f"Found **{len(seg_slice):,}** customers classified in **{chosen_seg}**:")

        inspect_cols = [c for c in [
            'customer_id', 'tenure', 'contract', 'internet_service',
            'monthly_charges', 'total_charges', 'total_additional_services', 'churn'
        ] if c in seg_slice.columns]
        st.dataframe(seg_slice[inspect_cols].head(100), width='stretch')

except Exception as e:
    st.error(f"Clustering error: {e}")
    st.exception(e)
