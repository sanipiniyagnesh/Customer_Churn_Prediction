import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Risk Prediction Engine")
st.write("Enter customer subscription details to calculate real-time churn probability.")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        cols = pickle.load(f)
    return model, scaler, cols

model, scaler, model_columns = load_artifacts()

# User Inputs
st.subheader("Account Metrics")
tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8500.0, value=780.0)

st.subheader("Subscription Features")
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

if st.button("Calculate Churn Risk"):
    # Create Raw Input DataFrame
    raw_input = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method
    }
    input_df = pd.DataFrame([raw_input])
    
    # One-Hot Encoding & Aligning Columns
    input_encoded = pd.get_dummies(input_df)
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]
    
    # Scale continuous variables & Predict
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!** Estimated Probability: **{probability:.1f}%**")
    else:
        st.success(f"✅ **Low Churn Risk.** Retained Probability: **{100 - probability:.1f}%**")