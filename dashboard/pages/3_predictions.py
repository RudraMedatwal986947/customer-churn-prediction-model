import streamlit as st
import requests
import os
import pandas as pd
import sys

API_URL = os.getenv("API_URL", "http://localhost:8000")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.connection import engine

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
st.title("🔮 Churn & CLV Predictions")
st.markdown("Run real-time inference on customer data using our trained XGBoost models.")

@st.cache_data(ttl=600)
def get_sample_customers():
    query = "SELECT customer_id FROM customers LIMIT 100"
    df = pd.read_sql(query, engine)
    return df['customer_id'].tolist()

try:
    sample_customers = get_sample_customers()
except:
    sample_customers = []

st.sidebar.header("Select Customer")
customer_id = st.sidebar.selectbox(
    "Choose a Customer ID for Prediction:", 
    [""] + sample_customers,
    help="Select a customer from the database to run predictions on."
)

custom_id = st.sidebar.text_input("Or enter Customer ID manually:")
if custom_id:
    customer_id = custom_id

if not customer_id:
    st.info("👈 Please select or enter a Customer ID in the sidebar to view predictions.")
else:
    st.subheader(f"Results for Customer: `{customer_id}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Churn Prediction")
        if st.button("Predict Churn Risk", type="primary"):
            with st.spinner("Running XGBoost Churn Classifier..."):
                try:
                    response = requests.post(f"{API_URL}/api/v1/predict/churn", json={"customer_id": customer_id})
                    if response.status_code == 200:
                        res = response.json()
                        prob = res['churn_probability'] * 100
                        
                        st.metric("Churn Probability", f"{prob:.1f}%")
                        
                        # Show risk level with color
                        if res['risk_level'] == "High":
                            st.error(f"Risk Level: **{res['risk_level']}** - Immediate retention action required!")
                        elif res['risk_level'] == "Medium":
                            st.warning(f"Risk Level: **{res['risk_level']}** - Monitor closely.")
                        else:
                            st.success(f"Risk Level: **{res['risk_level']}** - Customer is likely to stay.")
                            
                        st.progress(res['churn_probability'])
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    with col2:
        st.markdown("### Customer Lifetime Value")
        if st.button("Predict CLV", type="primary"):
            with st.spinner("Running XGBoost CLV Regressor..."):
                try:
                    response = requests.post(f"{API_URL}/api/v1/predict/clv", json={"customer_id": customer_id})
                    if response.status_code == 200:
                        res = response.json()
                        clv = res['predicted_clv']
                        
                        st.metric("Estimated Lifetime Value", f"${clv:,.2f}")
                        st.success("Prediction generated successfully.")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
