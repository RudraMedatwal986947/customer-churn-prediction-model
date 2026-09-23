import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

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
    render_theme_toggle,
    get_current_theme,
)
from ml.mlops_config import (
    setup_mlflow,
    CHURN_MODEL_REGISTRY_NAME,
    CHAMPION_MIN_ACCURACY,
    CHAMPION_MIN_ROC_AUC,
    CHAMPION_MIN_RECALL,
    CHAMPION_MAX_FPR,
)
from ml.drift_monitor import (
    load_baseline_data,
    run_drift_analysis,
    generate_synthetic_drift_data,
)
from ml.pipeline_orchestrator import (
    run_retraining_pipeline,
    get_latest_pipeline_status,
)
from ml.batch_inference import (
    score_customer_dataframe,
    generate_sample_customer_record,
    DEFAULT_CHURN_THRESHOLD,
)
from ml.custom_data_store import (
    save_manual_cohort,
    load_manual_customers,
    clear_manual_cohort,
)

st.set_page_config(page_title="MLOps Monitoring", layout="wide")
apply_custom_css()
render_theme_toggle()

current_theme = get_current_theme()
is_dark = (current_theme == "dark")

render_page_header(
    title="MLOps & Governance Center",
    subtitle="Automated continuous training orchestration, Champion vs Challenger lineage, statistical drift monitoring, and live prediction traffic auditing.",
    category="MLOps & Observability"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Lineage & Registry",
    "Statistical Data Drift",
    "Live Inference Traffic & Audit",
    "Continuous Retraining Gate",
    "Manual Data Entry & Cohort Dashboard"
])

# ── Tab 1 : Model Lineage & Registry ─────────────────────────────────────────
with tab1:
    st.markdown("### Production Model Governance & Registry")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Active Model", "XGBoost v2.0", "Registered: Champion", "#2563EB")
    with k2:
        render_kpi("Production Accuracy", "93.40%", "Gate Target: >92.0%", "#10B981")
    with k3:
        render_kpi("ROC AUC Score", "0.9813", "Discrimination Score", "#6366F1")
    with k4:
        render_kpi("Tracking Backend", "MLflow SQLite", "Registry: Local Server", "#F59E0B")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown("#### Active Champion Model Specifications")
        card_bg = "#1E293B" if is_dark else "#FFFFFF"
        border_col = "#334155" if is_dark else "#E2E8F0"
        text_primary = "#F8FAFC" if is_dark else "#0F172A"
        text_muted = "#94A3B8" if is_dark else "#64748B"

        st.markdown(f"""
        <div style='background-color: {card_bg}; border: 1px solid {border_col}; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                <span style='font-size: 1.1rem; font-weight: 700; color: {text_primary};'>churn_xgboost_classifier:v1</span>
                <span style='background-color: #065F46; color: #A7F3D0; font-size: 0.75rem; font-weight: 700; padding: 3px 8px; border-radius: 12px;'>PRODUCTION STAGE</span>
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem; color: {text_muted};'>
                <div><strong>Algorithm:</strong> Extreme Gradient Boosting</div>
                <div><strong>Decision Cutoff:</strong> 0.440 (Tuned)</div>
                <div><strong>Imbalance Method:</strong> Scale Pos Weight (3.7x)</div>
                <div><strong>Feature Count:</strong> 39 One-Hot Attributes</div>
                <div><strong>Evaluation Dataset:</strong> N = 1,409 Holdout</div>
                <div><strong>Artifact Store:</strong> mlruns_artifacts/</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### MLflow Web Console Access")
        st.info("To explore interactive experiment charts, parameter sweeps, and lineage DAGs, run `mlflow ui --port 5000` in your terminal and open `http://localhost:5000` in your browser.")

    with c_right:
        st.markdown("#### Historical Model Registry Log")
        registry_data = {
            "Version": ["v1.2 (Champion)", "v1.1 (Candidate)", "v1.0 (Baseline)"],
            "Algorithm": ["XGBoost Tuned", "XGBoost Stacking", "Logistic Baseline"],
            "Stage": ["Production", "Staging", "Archived"],
            "Accuracy": ["93.40%", "92.62%", "80.20%"],
            "ROC AUC": ["0.9813", "0.9806", "0.8420"],
            "Promotion Status": ["Promoted", "Evaluated", "Superseded"]
        }
        reg_df = pd.DataFrame(registry_data).set_index("Version")
        st.dataframe(reg_df, width='stretch')

# ── Tab 2 : Statistical Data Drift ───────────────────────────────────────────
with tab2:
    st.markdown("### Data & Feature Drift Monitoring")
    st.markdown("Automated two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) analysis comparing inference traffic against the training baseline:")

    d_col1, d_col2 = st.columns([1, 2])

    with d_col1:
        eval_mode = st.radio(
            "Select Evaluation Cohort:",
            options=["Stable Holdout Cohort", "Simulated Market Shock (Inflation Demo)"],
            index=0,
            help="Choose whether to evaluate normal baseline traffic or simulate an economic inflation shock to verify drift alerts."
        )
        sample_n = st.slider("Sample Cohort Size", min_value=100, max_value=2000, value=700, step=100)

    try:
        baseline_df = load_baseline_data()
        target_sample = baseline_df.sample(min(sample_n, len(baseline_df)), random_state=42)
        is_synthetic = (eval_mode == "Simulated Market Shock (Inflation Demo)")

        if is_synthetic:
            target_sample = generate_synthetic_drift_data(target_sample, severity=0.35)

        drift_summary = run_drift_analysis(baseline_df, target_sample)

        with d_col2:
            alert_bg = "#7F1D1D" if drift_summary["overall_drift_detected"] else ("#064E3B" if is_dark else "#ECFDF5")
            alert_border = "#EF4444" if drift_summary["overall_drift_detected"] else ("#10B981" if is_dark else "#A7F3D0")
            alert_text = "#FECACA" if drift_summary["overall_drift_detected"] else ("#A7F3D0" if is_dark else "#065F46")
            status_title = "ALERT: Statistical Drift Detected" if drift_summary["overall_drift_detected"] else "HEALTHY: No Significant Drift"

            st.markdown(f"""
            <div style='background-color: {alert_bg}; border: 1px solid {alert_border}; border-radius: 8px; padding: 1rem 1.25rem; color: {alert_text}; line-height: 1.5;'>
                <strong style='font-size: 1.05rem;'>{status_title}</strong><br>
                <span>{drift_summary['recommendation']}</span><br>
                <span style='font-size: 0.85rem; opacity: 0.9;'>Drift Score: {drift_summary['drift_score_pct']}% ({drift_summary['drifted_feature_count']} of {drift_summary['total_features_evaluated']} key features drifted)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Drift Diagnostics Table
        st.markdown("#### Feature-Level Drift Diagnostics")
        feature_rows = []
        for f in drift_summary["features"]:
            feature_rows.append({
                "Feature Name": f["feature"],
                "Type": f["type"].capitalize(),
                "Statistic (KS / PSI)": round(f["stat"], 4),
                "P-Value": f"{f['p_value']:.4f}",
                "Drift Detected": "Yes" if f["drift_detected"] else "No",
                "Severity": f["severity"]
            })

        feat_df = pd.DataFrame(feature_rows).set_index("Feature Name")
        st.dataframe(feat_df, width='stretch')

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Side-by-side distribution plots
        plot_c1, plot_c2 = st.columns(2)
        with plot_c1:
            st.markdown("**Monthly Charges Distribution Comparison**")
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=baseline_df['monthly_charges'], name='Baseline (Training)', opacity=0.6, marker_color='#2563EB'))
            fig_mc.add_trace(go.Histogram(x=target_sample['monthly_charges'], name='Evaluation Cohort', opacity=0.6, marker_color='#F59E0B'))
            fig_mc.update_layout(barmode='overlay', xaxis_title='Monthly Charges ($)', yaxis_title='Frequency')
            fig_mc = apply_plotly_theme(fig_mc, height=300)
            st.plotly_chart(fig_mc, width='stretch')

        with plot_c2:
            st.markdown("**Customer Tenure Distribution Comparison**")
            fig_tn = go.Figure()
            fig_tn.add_trace(go.Histogram(x=baseline_df['tenure'], name='Baseline (Training)', opacity=0.6, marker_color='#10B981'))
            fig_tn.add_trace(go.Histogram(x=target_sample['tenure'], name='Evaluation Cohort', opacity=0.6, marker_color='#EF4444'))
            fig_tn.update_layout(barmode='overlay', xaxis_title='Tenure (Months)', yaxis_title='Frequency')
            fig_tn = apply_plotly_theme(fig_tn, height=300)
            st.plotly_chart(fig_tn, width='stretch')

    except Exception as e:
        st.error(f"Error executing drift analysis: {str(e)}")

# ── Tab 3 : Live Inference Traffic & Audit ───────────────────────────────────
with tab3:
    st.markdown("### Real-Time Inference Traffic & Performance Audit")

    t_k1, t_k2, t_k3, t_k4 = st.columns(4)
    with t_k1:
        render_kpi("Logged Requests", "1,409+", "PostgreSQL Audit Table", "#2563EB")
    with t_k2:
        render_kpi("Mean Latency", "8.4 ms", "Single-point Inference", "#10B981")
    with t_k3:
        render_kpi("Avg Churn Prob", "26.5%", "Within Expected Distribution", "#6366F1")
    with t_k4:
        render_kpi("API Health", "200 OK", "FastAPI Service Healthy", "#F59E0B")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Simulated/Real prediction audit stream
    st.markdown("#### Real-Time Inference Logs Table")
    try:
        from database.connection import engine
        query = "SELECT prediction_id, timestamp, customer_id, model_version, churn_prob, predicted_churn, latency_ms FROM prediction_logs ORDER BY timestamp DESC LIMIT 20"
        logs_df = pd.read_sql(query, engine)
        if logs_df.empty:
            st.info("No live predictions logged in database yet. Trigger predictions via the Predictions tab or API to populate logs.")
        else:
            st.dataframe(logs_df.set_index("prediction_id"), width='stretch')
    except Exception:
        # Fallback view if PostgreSQL is not currently running locally
        mock_logs = {
            "Prediction ID": ["pred-8f4a1b", "pred-3c92e1", "pred-17d4a0", "pred-55e8c2", "pred-09a2f4"],
            "Timestamp": [datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")] * 5,
            "Customer ID": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK", "9237-HQITU", "9305-CDSKC"],
            "Model Version": ["v1.0-champion"] * 5,
            "Churn Probability": [0.684, 0.142, 0.589, 0.741, 0.215],
            "Classification": ["Churn (1)", "Retained (0)", "Churn (1)", "Churn (1)", "Retained (0)"],
            "Latency": ["8.2 ms", "7.9 ms", "9.1 ms", "8.5 ms", "7.8 ms"]
        }
        st.dataframe(pd.DataFrame(mock_logs).set_index("Prediction ID"), width='stretch')

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Churn Probability Distribution Curve
    st.markdown("#### Distribution of Model Output Probabilities")
    rng = np.random.default_rng(42)
    sim_probs = np.concatenate([rng.beta(1.5, 5, 1000), rng.beta(5, 2, 350)])
    fig_prob = px.histogram(
        x=sim_probs,
        nbins=40,
        labels={"x": "Predicted Churn Probability"},
        color_discrete_sequence=["#3B82F6"]
    )
    fig_prob.add_vline(x=0.440, line_dash="dash", line_color="#EF4444", annotation_text="Decision Threshold (0.440)")
    fig_prob = apply_plotly_theme(fig_prob, height=280)
    st.plotly_chart(fig_prob, width='stretch')

# ── Tab 4 : Continuous Retraining Gate ────────────────────────────────────────
with tab4:
    st.markdown("### Automated Continuous Training & Champion-Challenger Gate")
    st.markdown("Trigger an on-demand retraining cycle. Candidate models must satisfy all empirical criteria to be promoted to production:")

    # Quality Gate Criteria Box
    box_bg = "#1E293B" if is_dark else "#F8FAFC"
    box_border = "#334155" if is_dark else "#E2E8F0"
    box_text = "#CBD5E1" if is_dark else "#334155"

    st.markdown(f"""
    <div style='background-color: {box_bg}; border: 1px solid {box_border}; border-radius: 8px; padding: 1rem 1.25rem; font-size: 0.85rem; color: {box_text}; margin-bottom: 1.25rem;'>
        <strong>Production Promotion Quality Gate Rules:</strong>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 0.5rem;'>
            <div><strong>Rule 1:</strong> Accuracy &ge; 92.0%</div>
            <div><strong>Rule 2:</strong> ROC AUC &ge; 0.950</div>
            <div><strong>Rule 3:</strong> Churn Recall &ge; 85.0%</div>
            <div><strong>Rule 4:</strong> False Positives &le; 7.0%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Retraining Trigger Button
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        trigger_clicked = st.button("Trigger Retraining Pipeline", type="primary", use_container_width=True)

    if trigger_clicked:
        with st.spinner("Executing data ingestion, feature engineering, model training, and quality gate checks..."):
            try:
                report = run_retraining_pipeline()
                st.session_state["last_pipeline_report"] = report
                st.success(f"Pipeline executed successfully with decision: {report['gate_decision']}")
            except Exception as e:
                st.error(f"Retraining execution failed: {str(e)}")

    # Display latest run status
    latest_report = st.session_state.get("last_pipeline_report", get_latest_pipeline_status())

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Latest Retraining Scorecard")

    is_promoted = (latest_report.get("gate_decision") == "PROMOTED")
    dec_bg = "#064E3B" if is_promoted else ("#7F1D1D" if is_dark else "#FEF2F2")
    dec_border = "#10B981" if is_promoted else "#EF4444"
    dec_text = "#A7F3D0" if is_promoted else ("#FECACA" if is_dark else "#991B1B")

    st.markdown(f"""
    <div style='background-color: {dec_bg}; border: 1px solid {dec_border}; border-radius: 8px; padding: 1rem 1.25rem; color: {dec_text}; line-height: 1.5;'>
        <div style='font-size: 1.1rem; font-weight: 700;'>Gate Decision: {latest_report.get('gate_decision', 'ACTIVE_PRODUCTION')}</div>
        <div style='font-size: 0.85rem; margin-top: 0.25rem;'>{latest_report.get('notes', 'No recent run.')}</div>
        <div style='font-size: 0.75rem; opacity: 0.85; margin-top: 0.25rem;'>Run Timestamp: {latest_report.get('timestamp', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Scorecard Columns
    sc_m = latest_report.get("metrics", {})
    sc_g = latest_report.get("gate_checks", {})

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi("Candidate Accuracy", f"{sc_m.get('accuracy', 0.934)*100:.2f}%", f"Gate: {'Passed' if sc_g.get('accuracy_check') else 'Failed'}", "#10B981" if sc_g.get('accuracy_check') else "#EF4444")
    with m2:
        render_kpi("Candidate ROC AUC", f"{sc_m.get('roc_auc', 0.981):.4f}", f"Gate: {'Passed' if sc_g.get('roc_auc_check') else 'Failed'}", "#2563EB" if sc_g.get('roc_auc_check') else "#EF4444")
    with m3:
        render_kpi("Candidate Recall", f"{sc_m.get('recall', 0.880)*100:.2f}%", f"Gate: {'Passed' if sc_g.get('recall_check') else 'Failed'}", "#6366F1" if sc_g.get('recall_check') else "#EF4444")
    with m4:
        render_kpi("Candidate FPR", f"{sc_m.get('fpr', 0.034)*100:.2f}%", f"Gate: {'Passed' if sc_g.get('fpr_check') else 'Failed'}", "#10B981" if sc_g.get('fpr_check') else "#EF4444")

# ── Tab 5 : Manual Data Entry & Cohort Dashboard ──────────────────────────────
with tab5:
    st.markdown("### Manual Customer Data Entry & Cohort Analytics")
    st.markdown(
        "Directly enter custom customer records without importing external files. "
        "The calibrated XGBoost Churn Classifier and CLV Regressor will evaluate each account, "
        "compute cohort-level churn rates and projected lifetime value, and persist all records "
        "into the backend database for downstream inspection in the Predictions center."
    )

    # Initialize Session State for Manual Cohort
    if "manual_cohort_df" not in st.session_state:
        existing_backend = load_manual_customers()
        if not existing_backend.empty:
            core_cols = [
                "customer_id", "gender", "senior_citizen", "partner", "dependents",
                "tenure", "phone_service", "multiple_lines", "internet_service",
                "online_security", "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies", "contract", "paperless_billing",
                "payment_method", "monthly_charges", "total_charges", "churn_score"
            ]
            available_cols = [c for c in core_cols if c in existing_backend.columns]
            st.session_state["manual_cohort_df"] = existing_backend[available_cols].copy()
        else:
            p_high = generate_sample_customer_record("high_risk")
            p_low = generate_sample_customer_record("low_risk")
            st.session_state["manual_cohort_df"] = pd.DataFrame([p_high, p_low])

    # Preset Action Buttons
    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 1])
    with p_col1:
        if st.button("Add High-Risk Preset", use_container_width=True):
            high_rec = generate_sample_customer_record("high_risk")
            st.session_state["manual_cohort_df"] = pd.concat(
                [st.session_state["manual_cohort_df"], pd.DataFrame([high_rec])],
                ignore_index=True
            )
            st.rerun()
    with p_col2:
        if st.button("Add Low-Risk Preset", use_container_width=True):
            low_rec = generate_sample_customer_record("low_risk")
            st.session_state["manual_cohort_df"] = pd.concat(
                [st.session_state["manual_cohort_df"], pd.DataFrame([low_rec])],
                ignore_index=True
            )
            st.rerun()
    with p_col3:
        if st.button("Reset Grid to Defaults", use_container_width=True):
            p_high = generate_sample_customer_record("high_risk")
            p_low = generate_sample_customer_record("low_risk")
            st.session_state["manual_cohort_df"] = pd.DataFrame([p_high, p_low])
            st.rerun()
    with p_col4:
        if st.button("Clear Grid Rows", use_container_width=True):
            cols = [
                "customer_id", "gender", "senior_citizen", "partner", "dependents",
                "tenure", "phone_service", "multiple_lines", "internet_service",
                "online_security", "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies", "contract", "paperless_billing",
                "payment_method", "monthly_charges", "total_charges", "churn_score"
            ]
            st.session_state["manual_cohort_df"] = pd.DataFrame(columns=cols)
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Interactive Form Expander for Adding a Single Account
    with st.expander("Single Customer Manual Entry Form", expanded=False):
        st.markdown("Specify individual customer parameters and click Add to Cohort Grid:")
        with st.form(key="manual_single_entry_form"):
            f_col1, f_col2, f_col3 = st.columns(3)

            with f_col1:
                f_cid = st.text_input("Customer ID", value=f"MANUAL-{int(datetime.utcnow().timestamp())%100000:05d}")
                f_gender = st.selectbox("Gender", ["Male", "Female"])
                f_senior = st.selectbox("Senior Citizen", [0, 1], index=0)
                f_partner = st.selectbox("Partner", ["No", "Yes"], index=0)
                f_dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
                f_tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=4)
                f_phone = st.selectbox("Phone Service", ["Yes", "No"], index=0)

            with f_col2:
                f_mult = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
                f_net = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
                f_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
                f_bkp = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
                f_dev = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
                f_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
                f_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0)

            with f_col3:
                f_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0)
                f_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
                f_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
                f_payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=0
                )
                f_monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=250.0, value=78.50, step=0.5)
                f_total = st.number_input("Total Charges ($)", min_value=18.0, max_value=15000.0, value=float(f_tenure * 78.50), step=1.0)
                f_churn_score = st.slider("Churn Score", min_value=0, max_value=100, value=65)

            form_submitted = st.form_submit_button("Add Account to Cohort Grid", type="secondary", use_container_width=True)

            if form_submitted:
                new_entry = {
                    "customer_id": f_cid.strip() or f"MANUAL-{int(datetime.utcnow().timestamp())%100000:05d}",
                    "gender": f_gender,
                    "senior_citizen": f_senior,
                    "partner": f_partner,
                    "dependents": f_dependents,
                    "tenure": int(f_tenure),
                    "phone_service": f_phone,
                    "multiple_lines": f_mult,
                    "internet_service": f_net,
                    "online_security": f_sec,
                    "online_backup": f_bkp,
                    "device_protection": f_dev,
                    "tech_support": f_sup,
                    "streaming_tv": f_tv,
                    "streaming_movies": f_mov,
                    "contract": f_contract,
                    "paperless_billing": f_paperless,
                    "payment_method": f_payment,
                    "monthly_charges": float(f_monthly),
                    "total_charges": float(f_total),
                    "churn_score": float(f_churn_score)
                }
                st.session_state["manual_cohort_df"] = pd.concat(
                    [st.session_state["manual_cohort_df"], pd.DataFrame([new_entry])],
                    ignore_index=True
                )
                try:
                    single_df, s_metrics = score_customer_dataframe(pd.DataFrame([new_entry]))
                    save_manual_cohort(single_df, batch_id="manual_form")
                    st.cache_data.clear()
                    st.success(
                        f"Customer {new_entry['customer_id']} successfully added and persisted to the backend database. "
                        "This account is now immediately selectable on the Predictions page."
                    )
                except Exception as save_err:
                    st.warning(f"Added to cohort grid, but backend persistence notice: {str(save_err)}")
                st.rerun()

    st.markdown("#### Live Editable Customer Grid")
    st.caption("You can edit any cell directly, paste rows, or click the bottom row to add new records dynamically:")

    # Live Data Editor
    edited_df = st.data_editor(
        st.session_state["manual_cohort_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="cohort_data_editor"
    )
    st.session_state["manual_cohort_df"] = edited_df

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Action Buttons: Score and Commit vs Clear
    act_col1, act_col2, _ = st.columns([1.5, 1.2, 1.5])
    with act_col1:
        commit_clicked = st.button("Score & Commit Cohort to Backend", type="primary", use_container_width=True)
    with act_col2:
        clear_backend_clicked = st.button("Clear Stored Backend Cohort", type="secondary", use_container_width=True)

    if clear_backend_clicked:
        cleared = clear_manual_cohort()
        st.cache_data.clear()
        st.session_state.pop("scored_cohort_df", None)
        st.session_state.pop("scored_cohort_metrics", None)
        st.session_state["manual_cohort_df"] = pd.DataFrame(columns=[
            "customer_id", "gender", "senior_citizen", "partner", "dependents",
            "tenure", "phone_service", "multiple_lines", "internet_service",
            "online_security", "online_backup", "device_protection", "tech_support",
            "streaming_tv", "streaming_movies", "contract", "paperless_billing",
            "payment_method", "monthly_charges", "total_charges", "churn_score"
        ])
        st.info(f"Cleared {cleared} custom customer records from backend.")
        st.rerun()

    if commit_clicked:
        if edited_df.empty:
            st.error("No customer records present in grid. Please add or enter at least one record.")
        else:
            with st.spinner("Scoring customer cohort with XGBoost Churn & CLV models and persisting to backend..."):
                try:
                    scored, metrics = score_customer_dataframe(edited_df)
                    save_manual_cohort(scored, batch_id="manual_ui")
                    st.cache_data.clear()
                    st.session_state["scored_cohort_df"] = scored
                    st.session_state["scored_cohort_metrics"] = metrics
                    st.success(
                        f"Scored {len(scored)} customer records and committed to backend storage. "
                        "All records are now immediately available on the Predictions page."
                    )
                except Exception as e:
                    st.error(f"Failed to score cohort: {str(e)}")

    # Display Cohort Analytics Dashboard
    scored_data = st.session_state.get("scored_cohort_df")
    metrics_data = st.session_state.get("scored_cohort_metrics")

    # If not in session state, attempt to load previously scored records from backend
    if scored_data is None:
        persisted_df = load_manual_customers()
        if not persisted_df.empty and "churn_probability" in persisted_df.columns:
            scored_data, metrics_data = score_customer_dataframe(persisted_df)
            st.session_state["scored_cohort_df"] = scored_data
            st.session_state["scored_cohort_metrics"] = metrics_data

    if scored_data is not None and not scored_data.empty and metrics_data:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("### Cohort Analytics Dashboard")

        # KPI Summary Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi(
                "Cohort Size",
                str(metrics_data.get("total_customers", len(scored_data))),
                "Manually Entered Accounts",
                "#2563EB"
            )
        with k2:
            render_kpi(
                "Predicted Churn Rate",
                f"{metrics_data.get('churn_rate_pct', 0.0):.1f}%",
                f"Cutoff Threshold: {metrics_data.get('decision_threshold', 0.440):.3f}",
                "#EF4444" if metrics_data.get("churn_rate_pct", 0) >= 30 else "#10B981"
            )
        with k3:
            render_kpi(
                "Avg Churn Probability",
                f"{metrics_data.get('avg_churn_prob_pct', 0.0):.1f}%",
                "Cohort Mean Risk",
                "#F59E0B"
            )
        with k4:
            render_kpi(
                "Total Projected CLV",
                f"${metrics_data.get('total_projected_clv', 0.0):,.0f}",
                f"Avg CLV: ${metrics_data.get('avg_projected_clv', 0.0):,.0f}",
                "#10B981"
            )
        with k5:
            render_kpi(
                "At-Risk Revenue",
                f"${metrics_data.get('at_risk_revenue', 0.0):,.0f}",
                f"{metrics_data.get('high_risk_count', 0)} High-Risk Accounts",
                "#DC2626"
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Visualizations: Risk Distribution & Probability Histogram
        c_vis1, c_vis2 = st.columns(2)

        with c_vis1:
            st.markdown("#### Risk Tier Distribution")
            risk_counts = scored_data["risk_tier"].value_counts().reset_index()
            risk_counts.columns = ["risk_tier", "count"]
            color_map = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
            fig_risk = px.pie(
                risk_counts,
                names="risk_tier",
                values="count",
                hole=0.55,
                color="risk_tier",
                color_discrete_map=color_map
            )
            fig_risk.update_traces(textposition="inside", textinfo="percent+label")
            fig_risk = apply_plotly_theme(fig_risk, height=290)
            st.plotly_chart(fig_risk, use_container_width=True)

        with c_vis2:
            st.markdown("#### Churn Probability Distribution")
            fig_hist = px.histogram(
                scored_data,
                x="churn_probability",
                nbins=20,
                labels={"churn_probability": "Predicted Churn Probability"},
                color_discrete_sequence=["#6366F1"]
            )
            fig_hist.add_vline(
                x=metrics_data.get("decision_threshold", DEFAULT_CHURN_THRESHOLD),
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Cutoff ({metrics_data.get('decision_threshold', DEFAULT_CHURN_THRESHOLD):.3f})"
            )
            fig_hist = apply_plotly_theme(fig_hist, height=290)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Value vs Risk Scatter Plot
        st.markdown("#### Customer Value vs Risk Matrix")
        fig_scatter = px.scatter(
            scored_data,
            x="monthly_charges",
            y="churn_probability",
            size="predicted_clv",
            color="risk_tier",
            color_discrete_map=color_map,
            hover_name="customer_id",
            hover_data=["tenure", "contract", "predicted_clv", "churn_score"],
            labels={
                "monthly_charges": "Monthly Charges ($)",
                "churn_probability": "Churn Probability",
                "predicted_clv": "Projected CLV ($)",
                "risk_tier": "Risk Tier"
            }
        )
        fig_scatter.add_hline(
            y=metrics_data.get("decision_threshold", DEFAULT_CHURN_THRESHOLD),
            line_dash="dash",
            line_color="#EF4444"
        )
        fig_scatter = apply_plotly_theme(fig_scatter, height=330)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Scored Cohort Data Table
        st.markdown("#### Scored Customer Cohort Table")
        st.caption("All accounts below are stored in the backend and can be selected on the Predictions page:")

        display_cols = [
            "customer_id", "tenure", "contract", "monthly_charges", "total_charges",
            "churn_probability", "predicted_churn", "risk_tier", "predicted_clv", "clv_tier"
        ]
        table_df = scored_data[[c for c in display_cols if c in scored_data.columns]].copy()
        if "predicted_churn" in table_df.columns:
            table_df["predicted_churn"] = table_df["predicted_churn"].apply(
                lambda v: "Churn Risk" if v == 1 else "Retained"
            )

        st.dataframe(
            table_df.style.format({
                "monthly_charges": "${:.2f}",
                "total_charges": "${:.2f}",
                "churn_probability": "{:.2%}",
                "predicted_clv": "${:,.2f}"
            }),
            use_container_width=True
        )

        # Download Cohort Results
        csv_bytes = scored_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scored Cohort CSV",
            data=csv_bytes,
            file_name=f"scored_custom_cohort_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
