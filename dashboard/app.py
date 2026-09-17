import streamlit as st

st.set_page_config(
    page_title="Customer Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Customer Analytics Platform")
st.markdown("---")

st.markdown("""
### Welcome to the Intelligent Customer Lifetime Value & Churn Prediction Platform.

This platform allows you to:
- **Explore Data**: Dive into the demographic, service, and account information of our customers on the **EDA** page.
- **Customer Segmentation**: Understand different customer groups based on behavioral and monetary characteristics on the **Segmentation** page.
- **Live Predictions**: Run our advanced Machine Learning models (XGBoost) to predict churn risk and estimate Customer Lifetime Value (CLV) for specific customers on the **Predictions** page.

Select a page from the sidebar to get started!
""")

st.sidebar.success("Select a dashboard page above.")
st.sidebar.info("Application Status: Online")
