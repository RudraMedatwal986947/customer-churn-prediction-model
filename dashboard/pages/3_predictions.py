import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Predictions", page_icon="🔮")
st.title("Churn & CLV Predictions")

st.write("Enter customer details below to get predictions from the model API.")

if st.button("Predict Churn"):
    try:
        response = requests.post(f"{API_URL}/api/v1/predict/churn")
        st.json(response.json())
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
