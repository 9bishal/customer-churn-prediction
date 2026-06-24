"""
streamlit_app.py

Purpose:
    Interactive dashboard for the Customer Churn Prediction System.
    A user (e.g. a retention team member) enters a customer's details
    and gets back:
      - Churn probability
      - Risk tier (Low / Medium / High)
      - A chart showing the top global factors driving churn predictions

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import matplotlib.pyplot as plt
import streamlit as st

# Allow importing from src/ regardless of where streamlit is launched from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
# pyrefly: ignore [missing-import]
from predict import predict_churn, get_feature_importance, load_model  # noqa: E402


st.set_page_config(page_title="Customer Churn Predictor", layout="centered")


# Cache the model load so it's only read from disk once per session,
# not on every button click / widget interaction (Streamlit re-runs the
# whole script on every interaction by default).
@st.cache_resource
def get_cached_model():
    try:
        return load_model()
    except Exception as e:
        return e


model_or_exception = get_cached_model()
if isinstance(model_or_exception, Exception):
    st.title("Customer Churn Prediction System")
    st.error("### ⚠️ Model Loading Error")
    st.error(str(model_or_exception))
    st.info("To train a new model compatible with the current environment, please run the following command in your terminal:")
    st.code("python src/train_model.py")
    st.stop()

st.title("Customer Churn Prediction System")
st.write(
    "Enter a customer's details below to predict their likelihood of "
    "churning (cancelling their subscription) in the next billing cycle."
)

st.divider()

# --- Input form ---
# Kept to the most predictive / common-sense fields rather than all 19
# original columns, to keep the UI manageable while still covering the
# strongest churn signals (contract type, tenure, charges, internet
# service, payment method).

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months with company)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges (₹/$)", 18.0, 120.0, 70.0)
    total_charges = st.number_input(
        "Total Charges to date", min_value=0.0, value=float(tenure * monthly_charges)
    )
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

with st.expander("Additional service details (optional)"):
    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col4:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()

if st.button("Predict Churn Risk", type="primary"):
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    result = predict_churn(customer, model=get_cached_model())

    # --- Display results ---
    probability_pct = result["churn_probability"] * 100
    risk_tier = result["risk_tier"]

    risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{probability_pct:.1f}%")
    st.markdown(
        f"**Risk Tier:** :{risk_colors[risk_tier]}[{risk_tier}]"
    )

    if risk_tier == "High":
        st.warning(
            "This customer is at high risk of churning. Consider proactive "
            "retention outreach (discount offer, support check-in)."
        )
    elif risk_tier == "Medium":
        st.info(
            "This customer shows moderate churn risk. Monitor and consider "
            "a light-touch engagement (satisfaction survey, usage tips)."
        )
    else:
        st.success("This customer is likely to stay. No action needed.")

    st.divider()

    # --- Feature importance chart ---
    st.subheader("What Drives Churn Predictions? (Top Factors)")
    st.caption(
        "These are the model's overall top factors across all customers "
        "(not specific to this one prediction). Positive values increase "
        "churn likelihood; negative values decrease it."
    )

    importance_df = get_feature_importance(top_n=10, model=get_cached_model())

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d62728" if val > 0 else "#2ca02c" for val in importance_df["importance"]]
    ax.barh(importance_df["feature"], importance_df["importance"], color=colors)
    ax.set_xlabel("Effect on Churn Probability (Logistic Regression Coefficient)")
    ax.invert_yaxis()
    st.pyplot(fig)