import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Ensure we can import from database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.connection import engine

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📈", layout="wide")
st.title("📈 Exploratory Data Analysis")
st.markdown("Analyze the distribution of customer attributes, churn, and overall dataset characteristics.")

@st.cache_data(ttl=600)
def load_data():
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, engine)
    
    # Preprocess slightly for visualization
    df['TotalCharges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
    return df

try:
    with st.spinner("Loading data from database..."):
        df = load_data()
    
    st.write(f"**Total Customers loaded:** {len(df):,}")
    
    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df['churn'] == 'Yes').mean() * 100 if 'churn' in df.columns else 0
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()
    
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Average Tenure", f"{avg_tenure:.1f} Months")
    col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    
    st.markdown("---")
    
    # Visualizations
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Churn Distribution")
        fig_churn = px.pie(df, names='churn', title='Customer Churn Rate', hole=0.4, color_discrete_sequence=['#ff9999','#66b3ff'])
        st.plotly_chart(fig_churn, use_container_width=True)
        
    with row1_col2:
        st.subheader("Tenure Distribution by Churn")
        fig_tenure = px.histogram(df, x="tenure", color="churn", nbins=30, opacity=0.7, 
                                  title="Customer Tenure (Months)", barmode="overlay")
        st.plotly_chart(fig_tenure, use_container_width=True)
        
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("Monthly Charges Distribution")
        fig_charges = px.box(df, x="churn", y="MonthlyCharges", color="churn", 
                             title="Monthly Charges vs Churn")
        st.plotly_chart(fig_charges, use_container_width=True)
        
    with row2_col2:
        st.subheader("Contract Type vs Churn")
        contract_churn = df.groupby(['contract', 'churn']).size().reset_index(name='count')
        fig_contract = px.bar(contract_churn, x="contract", y="count", color="churn", 
                              title="Churn by Contract Type", barmode="group")
        st.plotly_chart(fig_contract, use_container_width=True)
        
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(100))

except Exception as e:
    st.error(f"Failed to load data from the database. Error: {e}")
