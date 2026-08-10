import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Ensure we can import from database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📈", layout="wide")
st.title("📈 Exploratory Data Analysis")
st.markdown("Analyze the distribution of customer attributes, churn, and overall dataset characteristics.")

# Resolve paths relative to project root (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

@st.cache_data(ttl=600)
def load_data():
    """Load data from PostgreSQL; fallback to the local Excel file if DB is unavailable."""
    try:
        from database.connection import engine
        query = "SELECT * FROM customers"
        df = pd.read_sql(query, engine)
        df['TotalCharges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
        return df, "database"
    except Exception as db_err:
        # Fallback: read from Excel file
        df = pd.read_excel(EXCEL_PATH)

        # Normalise column names to lowercase with underscores
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_', regex=False)
        )

        # Rename key columns to match what the rest of the page expects
        rename_map = {
            'customerid':       'customer_id',
            'churn_label':      'churn',
            'tenure_months':    'tenure',
            'monthly_charges':  'monthly_charges',
            'total_charges':    'total_charges',
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Derive churn column (Yes/No string) if it came in as 0/1
        if 'churn' in df.columns and df['churn'].dtype != object:
            df['churn'] = df['churn'].map({1: 'Yes', 0: 'No'})

        df['TotalCharges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
        return df, f"local file (DB unavailable: {db_err})"

try:
    with st.spinner("Loading data..."):
        df, source = load_data()

    if "local file" in source:
        st.info(f"ℹ️ Data loaded from **{source}**", icon="📂")

    st.write(f"**Total Customers loaded:** {len(df):,}")

    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df['churn'] == 'Yes').mean() * 100 if 'churn' in df.columns else 0
    avg_tenure = df['tenure'].mean() if 'tenure' in df.columns else 0
    avg_monthly = df['MonthlyCharges'].mean() if 'MonthlyCharges' in df.columns else 0

    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Average Tenure", f"{avg_tenure:.1f} Months")
    col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

    st.markdown("---")

    # Visualizations
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Churn Distribution")
        fig_churn = px.pie(
            df, names='churn', title='Customer Churn Rate', hole=0.4,
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        st.plotly_chart(fig_churn, width='stretch')

    with row1_col2:
        st.subheader("Tenure Distribution by Churn")
        fig_tenure = px.histogram(
            df, x="tenure", color="churn", nbins=30, opacity=0.7,
            title="Customer Tenure (Months)", barmode="overlay"
        )
        st.plotly_chart(fig_tenure, width='stretch')

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Monthly Charges Distribution")
        fig_charges = px.box(
            df, x="churn", y="MonthlyCharges", color="churn",
            title="Monthly Charges vs Churn"
        )
        st.plotly_chart(fig_charges, use_container_width=True)

    with row2_col2:
        st.subheader("Contract Type vs Churn")
        # 'contract' column may be absent in encoded CSV; guard gracefully
        if 'contract' in df.columns:
            contract_churn = df.groupby(['contract', 'churn']).size().reset_index(name='count')
            fig_contract = px.bar(
                contract_churn, x="contract", y="count", color="churn",
                title="Churn by Contract Type", barmode="group"
            )
            st.plotly_chart(fig_contract, width='stretch')
        else:
            st.info("Contract column not available in current data source.")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(100))

except Exception as e:
    st.error(f"Failed to load data. Error: {e}")
    st.exception(e)
