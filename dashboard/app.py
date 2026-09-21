import streamlit as st
import sys
import os

# Allow imports from current directory and project root
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui_components import (
    apply_custom_css,
    render_page_header,
    render_kpi,
    render_feature_card,
)

st.set_page_config(
    page_title="Customer Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

render_page_header(
    title="Customer Lifetime Value & Churn Prediction Platform",
    subtitle="Enterprise machine learning intelligence portal for customer retention, value estimation, and behavioral clustering.",
    category="Platform Overview"
)

# Executive KPI Summary Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi(
        label="Total Customers",
        value="7,043",
        caption="Full cohort in PostgreSQL database",
        accent_color="#2563EB"
    )
with k2:
    render_kpi(
        label="Churn Model Accuracy",
        value="93.40%",
        caption="Tuned XGBoost Classifier (Target: >92%)",
        accent_color="#10B981"
    )
with k3:
    render_kpi(
        label="ROC AUC Score",
        value="0.9813",
        caption="Near-optimal class discrimination",
        accent_color="#6366F1"
    )
with k4:
    render_kpi(
        label="CLV Explanatory Power",
        value="R² = 0.9986",
        caption="XGBoost Regressor (MAE: $57.91)",
        accent_color="#F59E0B"
    )

st.markdown("---")

# Feature Modules Overview
st.markdown("### Platform Modules")
st.markdown("Select a module from the left sidebar navigation to begin exploration.")

col_a, col_b = st.columns(2)
with col_a:
    render_feature_card(
        title="1. Exploratory Data Analysis",
        description="Inspect demographic distributions, tenure profiles, contract churn correlations, and monthly charge distributions across all 7,043 records with interactive filters.",
        tag="Module: EDA",
        tag_color="#2563EB"
    )
with col_b:
    render_feature_card(
        title="2. Behavioral Segmentation",
        description="Unsupervised K-Means clustering (k=4) grouping customers into distinct behavioral cohorts (e.g. New Low-Spend, Long-Term Premium) with retention strategies.",
        tag="Module: Segmentation",
        tag_color="#10B981"
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    render_feature_card(
        title="3. Real-Time Inference & SHAP",
        description="Run real-time predictions for individual customers. Computes churn risk probability, estimates Customer Lifetime Value (CLV), and explains predictions using SHAP attributions.",
        tag="Module: Predictions",
        tag_color="#6366F1"
    )
with col_d:
    render_feature_card(
        title="4. Model Performance Report",
        description="Comprehensive evaluation including confusion matrices, ROC curves, feature importance ranking, regression residual diagnostics, and comparative model benchmarks.",
        tag="Module: Evaluation",
        tag_color="#F59E0B"
    )

st.markdown("---")

# Architecture & Specifications
st.markdown("### System Architecture")
arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)

with arch_col1:
    st.markdown("""
    **Data Store**
    - PostgreSQL 15
    - SQLAlchemy 2.0 ORM
    - 7,043 Customer Records
    - Auto-seeding Pipeline
    """)

with arch_col2:
    st.markdown("""
    **Machine Learning**
    - XGBoost Classifier (93.4%)
    - XGBoost Regressor (R² 0.998)
    - K-Means Clustering (k=4)
    - SHAP Explainability Engine
    """)

with arch_col3:
    st.markdown("""
    **Backend Services**
    - FastAPI REST Service
    - Uvicorn ASGI Server
    - JSON Schemas & Validation
    - Automated Unit Tests
    """)

with arch_col4:
    st.markdown("""
    **Interactive Dashboard**
    - Streamlit Wide Layout
    - Plotly Interactive Charts
    - Modular Page Structure
    - Real-Time Model Inference
    """)

# Sidebar info
st.sidebar.header("Navigation")
st.sidebar.markdown("""
Use the sidebar menu above to switch between platform pages:
- **1 EDA**: Dataset exploration & visual analysis
- **2 Segmentation**: K-Means clustering & persona analysis
- **3 Predictions**: Customer-level inference & SHAP
- **4 Model Performance**: Validation metrics & charts
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.markdown("""
- **Backend API**: `Operational`
- **Database**: `Connected`
- **Model Engine**: `XGBoost v2.0`
""")
