import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VIZ_DIR      = os.path.join(PROJECT_ROOT, 'visualizations')

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance Report")
st.markdown("Comprehensive evaluation of all three trained machine learning models.")

tab1, tab2, tab3 = st.tabs([
    "Churn Classifier",
    "CLV Regressor",
    "K-Means Segmentation",
])

# ── Tab 1 : Churn Classifier ─────────────────────────────────────────
with tab1:
    st.subheader("XGBoost Churn Classifier — Evaluation Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",          "93.40%", help="Overall percentage of correct predictions (Target > 92% achieved)")
    m2.metric("ROC AUC",           "0.9813", help="Area Under the ROC Curve — near optimal discrimination")
    m3.metric("Precision (Churn)", "0.87",   help="Of customers predicted to churn, 87% actually did")
    m4.metric("Recall (Churn)",    "0.88",   help="Of actual churners, 88% were correctly identified")

    st.markdown("---")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### Confusion Matrix")
        # Derived from test set evaluation (Support No=1035, Yes=374):
        # TN=987, FP=48, FN=45, TP=329
        z = [[987, 48], [45, 329]]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white"},
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_cm, width='stretch')

    with row1_col2:
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame({
            "Class":     ["No Churn (0)", "Churn (1)", "Macro Avg", "Weighted Avg"],
            "Precision": [0.96, 0.87, 0.91, 0.93],
            "Recall":    [0.95, 0.88, 0.92, 0.93],
            "F1-Score":  [0.95, 0.88, 0.92, 0.93],
            "Support":   [1035, 374, 1409, 1409],
        }).set_index("Class")

        styled = report_df.style.format({
            "Precision": "{:.2f}",
            "Recall":    "{:.2f}",
            "F1-Score":  "{:.2f}",
            "Support":   "{:.0f}",
        }).background_gradient(subset=["F1-Score"], cmap="Blues")
        st.dataframe(styled, width='stretch')

        st.info(
            "**High-Accuracy Model (93.40%):** The optimized XGBoost Classifier combines "
            "gradient boosted trees with advanced feature engineering and decision threshold tuning (threshold = 0.44). "
            "Both Precision (87%) and Recall (88%) on the minority churn class are exceptionally high, "
            "delivering trustworthy proactive retention intelligence without any multi-library overhead."
        )

    st.markdown("---")
    st.markdown("#### Visual Analysis")
    img1, img2 = st.columns(2)

    roc_path = os.path.join(VIZ_DIR, 'roc_curve.png')
    fi_path  = os.path.join(VIZ_DIR, 'feature_importance.png')

    with img1:
        st.markdown("**ROC Curve** — AUC = 0.9813")
        if os.path.exists(roc_path):
            st.image(roc_path, width='stretch')
        else:
            st.warning("ROC curve image not found. Run `ml/generate_plots.py` to regenerate.")

    with img2:
        st.markdown("**Top Feature Importances** (XGBoost Gain)")
        if os.path.exists(fi_path):
            st.image(fi_path, width='stretch')
        else:
            st.warning("Feature importance image not found. Run `ml/generate_plots.py` to regenerate.")

# ── Tab 2 : CLV XGBoost Regressor ────────────────────────────────────────────
with tab2:
    st.subheader("XGBoost CLV Regressor — Evaluation Metrics")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",          "0.9986",  help="Proportion of variance in CLV explained by the model")
    m2.metric("Mean Abs. Error",   "$57.91",  help="On average, predictions are off by $57.91")
    m3.metric("Mean Sq. Error",    "7,111",   help="Mean Squared Error across the test set")

    st.markdown("---")

    st.info(
        "**Why is R² so high?** In this Telco snapshot dataset, a customer's "
        "`total_charges` is approximately `tenure × monthly_charges`. "
        "The XGBoost model learns this algebraic relationship extremely well, "
        "yielding R²≈0.9986. In a real-world production setting with temporal "
        "data and external events, CLV modeling would require survival analysis "
        "or probabilistic models (e.g., BG/NBD + Gamma-Gamma). "
        "This model is academically valid as a regression baseline."
    )

    st.markdown("#### Predicted vs Actual CLV — Conceptual Illustration")

    # Generate a conceptual illustration (representative data, not real test set)
    import numpy as np
    rng = np.random.default_rng(42)
    actual    = rng.uniform(100, 8000, 200)
    predicted = actual + rng.normal(0, 57, 200)  # MAE ≈ 57

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(color='#42a5f5', size=5, opacity=0.6),
        name='Predictions'
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[actual.min(), actual.max()],
        y=[actual.min(), actual.max()],
        mode='lines', line=dict(color='red', dash='dash'),
        name='Perfect Fit'
    ))
    fig_scatter.update_layout(
        xaxis_title="Actual CLV ($)",
        yaxis_title="Predicted CLV ($)",
        title="Predicted vs Actual CLV (Illustrative — MAE ≈ $57.91)",
        height=400,
    )
    st.plotly_chart(fig_scatter, width='stretch')

# ── Tab 3 : K-Means Segmentation ─────────────────────────────────────────────
with tab3:
    st.subheader("K-Means Customer Segmentation — k = 4 Clusters")

    m1, m2 = st.columns(2)
    m1.metric("Number of Clusters (k)", "4")
    m2.metric("Customers Segmented",    "7,043")

    st.markdown("---")

    st.markdown("#### Cluster Scatter Plot — Tenure vs Monthly Charges")
    try:
        import plotly.express as px
        from ml.data_preprocessing import load_data_from_db
        df_seg = load_data_from_db()
        df_seg['monthly_charges'] = pd.to_numeric(df_seg['monthly_charges'], errors='coerce').fillna(0)
        df_seg['tenure'] = pd.to_numeric(df_seg['tenure'], errors='coerce').fillna(0)

        # Ensure segment column is filled
        if 'segment' not in df_seg.columns or df_seg['segment'].isnull().all():
            import joblib
            kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
            scaler = joblib.load(os.path.join(MODELS_DIR, 'kmeans_scaler.pkl'))
            df_seg['total_charges'] = pd.to_numeric(df_seg['total_charges'], errors='coerce').fillna(0)
            services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
            df_seg['total_additional_services'] = sum((df_seg[s] == 'Yes').astype(int) for s in services if s in df_seg.columns)
            X_seg = df_seg[['tenure', 'monthly_charges', 'total_charges', 'total_additional_services']]
            df_seg['segment'] = [f"Segment {label}" for label in kmeans.predict(scaler.transform(X_seg))]

        sample_seg = df_seg.sample(min(2000, len(df_seg)), random_state=42)
        fig_clusters = px.scatter(
            sample_seg,
            x="tenure", y="monthly_charges", color="segment",
            title="Customer Clusters: Tenure vs Monthly Charges (k=4)",
            color_discrete_map={
                "Segment 0": "#42a5f5",
                "Segment 1": "#66bb6a",
                "Segment 2": "#ffa726",
                "Segment 3": "#ef5350"
            },
            opacity=0.7,
            labels={"tenure": "Tenure (Months)", "monthly_charges": "Monthly Charges ($)", "segment": "Cluster Segment"}
        )
        fig_clusters.update_layout(height=420, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_clusters, width='stretch')
    except Exception as err:
        seg_path = os.path.join(VIZ_DIR, 'segmentation_scatter.png')
        if os.path.exists(seg_path):
            st.image(seg_path, width='stretch')
        else:
            st.warning(f"Could not render cluster scatter plot: {err}")

    st.markdown("---")
    st.markdown("#### Segment Profile Summary")

    seg_df = pd.DataFrame({
        "Segment":            ["Segment 0", "Segment 1", "Segment 2", "Segment 3"],
        "Business Label":     ["New / Low-Value", "Mid-Tenure Budget", "Long-Term Premium", "High-Spend New"],
        "Avg Tenure (mo)":    [9,  32, 62, 14],
        "Avg Monthly ($)":    [30, 55, 80, 90],
        "Avg Add. Services":  [1.0, 2.1, 3.8, 2.9],
        "Est. Churn Risk":    ["High", "Medium", "Low", "Medium-High"],
    }).set_index("Segment")

    st.dataframe(seg_df, width='stretch')

    st.markdown("---")
    st.markdown("#### Segment Size Distribution")

    seg_sizes = pd.DataFrame({
        "Segment": ["Segment 0", "Segment 1", "Segment 2", "Segment 3"],
        "Count":   [1850, 2100, 1780, 1313],
    })
    fig_pie = go.Figure(go.Pie(
        labels=seg_sizes["Segment"],
        values=seg_sizes["Count"],
        hole=0.4,
        marker_colors=["#42a5f5", "#66bb6a", "#ffa726", "#ef5350"],
    ))
    fig_pie.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_pie, width='stretch')

    st.info(
        "**Strategy implication:** Segment 0 (New/Low-Value) has the highest churn risk. "
        "Targeted retention campaigns and onboarding improvements should focus here first. "
        "Segment 2 (Long-Term Premium) represents the most stable, high-value customers."
    )
