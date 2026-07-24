import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
import sys

API_URL = os.getenv("API_URL", "http://localhost:8000")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.connection import engine

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation Analysis")
st.markdown("View K-Means clustering results and RFM (Recency, Frequency, Monetary) based segments.")

@st.cache_data(ttl=60)
def fetch_segment_summary():
    response = requests.get(f"{API_URL}/api/v1/insights/segmentation/summary")
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=600)
def load_segment_data():
    query = "SELECT customer_id, tenure, CAST(monthly_charges AS FLOAT) as monthly_charges, CAST(total_charges AS FLOAT) as total_charges, segment FROM customers WHERE segment IS NOT NULL"
    return pd.read_sql(query, engine)

try:
    with st.spinner("Fetching segmentation summary from API..."):
        summary_data = fetch_segment_summary()
        
    if summary_data:
        st.subheader("Segment Summary Statistics")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.style.format({"avg_tenure": "{:.1f}", "avg_monthly_charges": "${:.2f}"}), use_container_width=True)
        
        # Load raw data for scatter plot
        df_segs = load_segment_data()
        
        if not df_segs.empty:
            st.markdown("---")
            st.subheader("Cluster Visualization")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter = px.scatter(
                    df_segs.sample(min(2000, len(df_segs))), # Sample to avoid overwhelming the browser
                    x="tenure", y="monthly_charges", color="segment",
                    title="Tenure vs Monthly Charges by Segment",
                    opacity=0.7
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col2:
                fig_box = px.box(
                    df_segs, x="segment", y="total_charges", color="segment",
                    title="Total Charges Distribution by Segment"
                )
                st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No segmentation data available. Have you run the segmentation ML script yet?")

except requests.exceptions.RequestException as e:
    st.error(f"Error connecting to Insights API: {e}. Make sure the FastAPI server is running.")
except Exception as e:
    st.error(f"An error occurred: {e}")
