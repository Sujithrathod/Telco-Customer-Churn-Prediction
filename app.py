"""
app.py  –  Streamlit frontend
Run: ./venv/bin/streamlit run app.py
"""

import os
import streamlit as st
import requests

API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000")) + "/predict"

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")
st.title("📡 Telco Customer Churn Predictor")
st.markdown("Fill in customer details to predict churn risk in real time.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender         = st.selectbox("Gender",         ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner        = st.selectbox("Partner",        ["Yes", "No"])
    dependents     = st.selectbox("Dependents",     ["Yes", "No"])
    tenure         = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("📶 Services")
    phone_service  = st.selectbox("Phone Service",     ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines",    ["Yes", "No", "No phone service"])
    internet       = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    online_security= st.selectbox("Online Security",   ["Yes", "No", "No internet service"])
    online_backup  = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support   = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV",      ["Yes", "No", "No internet service"])
    streaming_mv   = st.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"])

with col3:
    st.subheader("💳 Billing")
    contract        = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment         = st.selectbox("Payment Method",    [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",   0.0, 9000.0, 840.0, step=10.0)

st.divider()

if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):
    payload = {
        "gender": gender, "SeniorCitizen": senior_citizen,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_prot, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    with st.spinner("Running prediction..."):
        try:
            res    = requests.post(API_URL, json=payload, timeout=10)
            result = res.json()

            prob  = result["churn_probability"]
            label = result["churn_prediction"]
            risk  = result["risk_level"]
            rec   = result["recommendation"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Churn Probability", f"{prob*100:.1f}%")
            m2.metric("Prediction",        "Will Churn 🚨" if label else "Will Stay ✅")
            m3.metric("Risk Level",        risk)

            st.progress(prob, text=f"Churn probability: {prob*100:.1f}%")
            st.info(f"💡 **Recommendation:** {rec}")

            with st.expander("📄 Raw API Response"):
                st.json(result)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure uvicorn is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** Soft-Voting Ensemble
    - Random Forest
    - Logistic Regression
    - XGBoost

    **Threshold:** 0.45 (tuned for recall)

    ---
    **Risk Levels:**
    - 🔴 High Risk: ≥ 65%
    - 🟡 Medium Risk: 40–65%
    - 🟢 Low Risk: < 40%
    """)
