import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

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

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
apply_custom_css()

render_page_header(
    title="Exploratory Data Analysis",
    subtitle="Interactive cohort analysis exploring demographics, tenure dynamics, contract types, and financial churn drivers.",
    category="Data Discovery"
)

EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

@st.cache_data(ttl=600)
def load_data():
    """Load data from PostgreSQL; fallback to local Excel file if DB is unavailable."""
    try:
        from database.connection import engine
        query = "SELECT * FROM customers"
        df = pd.read_sql(query, engine)
        df['TotalCharges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
        return df, "database"
    except Exception as db_err:
        df = pd.read_excel(EXCEL_PATH)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_', regex=False)
        )
        rename_map = {
            'customerid':       'customer_id',
            'churn_label':      'churn',
            'tenure_months':    'tenure',
            'monthly_charges':  'monthly_charges',
            'total_charges':    'total_charges',
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        if 'churn' in df.columns and pd.api.types.is_numeric_dtype(df['churn']):
            df['churn'] = df['churn'].map({1: 'Yes', 0: 'No'})

        df['TotalCharges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
        return df, f"local file (DB unavailable: {db_err})"

try:
    with st.spinner("Loading cohort data..."):
        raw_df, source = load_data()

    if "local file" in source:
        st.info(f"Data loaded from **{source}**")

    # ── Sidebar Filters ──────────────────────────────────────────────────────────
    st.sidebar.header("Filter Cohort")

    # Contract filter
    contract_opts = ["All"]
    if "contract" in raw_df.columns:
        contract_opts += sorted(raw_df["contract"].dropna().unique().tolist())
    selected_contract = st.sidebar.selectbox("Contract Type:", contract_opts)

    # Internet Service filter
    internet_opts = ["All"]
    if "internet_service" in raw_df.columns:
        internet_opts += sorted(raw_df["internet_service"].dropna().unique().tolist())
    selected_internet = st.sidebar.selectbox("Internet Service:", internet_opts)

    # Senior Citizen filter
    senior_opts = ["All", "Yes", "No"] if "senior_citizen" in raw_df.columns else ["All"]
    selected_senior = st.sidebar.selectbox("Senior Citizen:", senior_opts)

    # Apply filters
    filtered_df = raw_df.copy()
    if selected_contract != "All":
        filtered_df = filtered_df[filtered_df["contract"] == selected_contract]
    if selected_internet != "All":
        filtered_df = filtered_df[filtered_df["internet_service"] == selected_internet]
    if selected_senior != "All" and "senior_citizen" in filtered_df.columns:
        val = 1 if selected_senior == "Yes" else 0
        filtered_df = filtered_df[filtered_df["senior_citizen"] == val]

    # ── High-Level Metric Cards ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total_count = len(filtered_df)
    churn_rate = (filtered_df['churn'] == 'Yes').mean() * 100 if ('churn' in filtered_df.columns and total_count > 0) else 0
    avg_tenure = filtered_df['tenure'].mean() if ('tenure' in filtered_df.columns and total_count > 0) else 0
    avg_monthly = filtered_df['MonthlyCharges'].mean() if ('MonthlyCharges' in filtered_df.columns and total_count > 0) else 0

    with col1:
        render_kpi(
            label="Filtered Cohort",
            value=f"{total_count:,}",
            caption=f"Out of {len(raw_df):,} total records",
            accent_color="#2563EB"
        )
    with col2:
        churn_color = "#EF4444" if churn_rate > 30 else ("#F59E0B" if churn_rate > 20 else "#10B981")
        render_kpi(
            label="Cohort Churn Rate",
            value=f"{churn_rate:.1f}%",
            caption="Customers with Churn = Yes",
            accent_color=churn_color
        )
    with col3:
        render_kpi(
            label="Average Tenure",
            value=f"{avg_tenure:.1f} Mo",
            caption="Customer relationship length",
            accent_color="#6366F1"
        )
    with col4:
        render_kpi(
            label="Avg Monthly Spend",
            value=f"${avg_monthly:.2f}",
            caption="Recurring monthly revenue",
            accent_color="#059669"
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── Primary Visualizations (Row 1) ─────────────────────────────────────────
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### Churn Distribution")
        fig_churn = px.pie(
            filtered_df,
            names='churn',
            hole=0.45,
            color='churn',
            color_discrete_map={'No': '#3B82F6', 'Yes': '#EF4444'},
        )
        fig_churn.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
        )
        fig_churn = apply_plotly_theme(fig_churn, height=360)
        st.plotly_chart(fig_churn, width='stretch')

    with row1_col2:
        st.markdown("#### Tenure Distribution by Churn Status")
        fig_tenure = px.histogram(
            filtered_df,
            x="tenure",
            color="churn",
            nbins=30,
            opacity=0.8,
            barmode="overlay",
            color_discrete_map={'No': '#3B82F6', 'Yes': '#EF4444'},
            labels={"tenure": "Tenure (Months)", "count": "Customer Count"}
        )
        fig_tenure.update_layout(
            xaxis_title="Tenure in Months",
            yaxis_title="Customer Count"
        )
        fig_tenure = apply_plotly_theme(fig_tenure, height=360)
        st.plotly_chart(fig_tenure, width='stretch')

    # ── Secondary Visualizations (Row 2) ───────────────────────────────────────
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### Monthly Charges vs Churn")
        fig_charges = px.box(
            filtered_df,
            x="churn",
            y="MonthlyCharges",
            color="churn",
            color_discrete_map={'No': '#3B82F6', 'Yes': '#EF4444'},
            labels={"MonthlyCharges": "Monthly Charges ($)", "churn": "Churn Status"},
            points="outliers"
        )
        fig_charges.update_layout(
            xaxis_title="Churn Status",
            yaxis_title="Monthly Charges ($)"
        )
        fig_charges = apply_plotly_theme(fig_charges, height=360)
        st.plotly_chart(fig_charges, width='stretch')

    with row2_col2:
        st.markdown("#### Churn by Contract Type")
        if 'contract' in filtered_df.columns:
            contract_churn = (
                filtered_df.groupby(['contract', 'churn'])
                .size()
                .reset_index(name='count')
            )
            fig_contract = px.bar(
                contract_churn,
                x="contract",
                y="count",
                color="churn",
                barmode="group",
                color_discrete_map={'No': '#3B82F6', 'Yes': '#EF4444'},
                labels={"count": "Number of Customers", "contract": "Contract Type"}
            )
            fig_contract.update_layout(
                xaxis_title="Contract Agreement",
                yaxis_title="Number of Customers"
            )
            fig_contract = apply_plotly_theme(fig_contract, height=360)
            st.plotly_chart(fig_contract, width='stretch')
        else:
            st.info("Contract column is not present in the current dataset.")

    # ── Service Adoption Breakdown ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Additional Service Adoption & Retention Impact")
    st.markdown(
        "Compare churn incidence across security and support add-ons. Customers with protective services exhibit markedly lower churn."
    )

    services = [
        ('online_security', 'Online Security'),
        ('tech_support', 'Tech Support'),
        ('online_backup', 'Online Backup'),
        ('device_protection', 'Device Protection'),
        ('streaming_tv', 'Streaming TV'),
        ('streaming_movies', 'Streaming Movies')
    ]

    service_data = []
    for col_name, label in services:
        if col_name in filtered_df.columns and 'churn' in filtered_df.columns:
            sub = filtered_df[filtered_df[col_name] == 'Yes']
            if len(sub) > 0:
                rate = (sub['churn'] == 'Yes').mean() * 100
                service_data.append({
                    'Service': label,
                    'Adoption Count': len(sub),
                    'Churn Rate (%)': round(rate, 1)
                })

    if service_data:
        svc_df = pd.DataFrame(service_data).sort_values('Churn Rate (%)', ascending=True)
        fig_svc = px.bar(
            svc_df,
            x='Churn Rate (%)',
            y='Service',
            orientation='h',
            text='Churn Rate (%)',
            color='Churn Rate (%)',
            color_continuous_scale=['#10B981', '#F59E0B', '#EF4444']
        )
        fig_svc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_svc.update_layout(
            xaxis_title="Churn Rate for Subscribed Customers (%)",
            yaxis_title="",
            coloraxis_showscale=False
        )
        fig_svc = apply_plotly_theme(fig_svc, height=300)
        st.plotly_chart(fig_svc, width='stretch')

    # ── Raw Data Table ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Inspect Filtered Raw Dataset", expanded=False):
        st.markdown(f"Displaying top 100 records for the selected cohort filter ({total_count:,} total matches):")
        display_cols = [c for c in [
            'customer_id', 'gender', 'senior_citizen', 'tenure', 'contract',
            'paperless_billing', 'payment_method', 'MonthlyCharges', 'TotalCharges', 'churn'
        ] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].head(100), width='stretch')

except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.exception(e)
