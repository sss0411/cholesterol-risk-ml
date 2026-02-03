import streamlit as st
import numpy as np
import joblib
import urllib.request
import os

# =========================
# Model configuration
# =========================
MODEL_URL = "https://github.com/sss0411/cholesterol-risk-ml/releases/download/v1.0/rf_cholesterol_model.pkl"
MODEL_PATH = "rf_cholesterol_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)

# =========================
# Page layout
# =========================
st.set_page_config(page_title="Cholesterol Risk Prediction", layout="centered")

st.title("Predicting the Risk of Elevated Cholesterol")
st.subheader("Enter patient data")

# =========================
# Inputs
# =========================
age = st.number_input(
    "Age (years)",
    min_value=18,
    max_value=100,
    value=60,
    step=1
)

sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex_label == "Female" else 1

smoking_label = st.selectbox("Smoking", ["No", "Yes"])
smoking = 1 if smoking_label == "Yes" else 0

alcohol_label = st.selectbox("Alcohol drinking", ["No", "Yes"])
alcohol = 1 if alcohol_label == "Yes" else 0

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=50.0,
    step=0.1,
    format="%.2f"
)

waist = st.number_input(
    "Waist circumference (cm)",
    min_value=50.0,
    max_value=150.0,
    value=120.0,
    step=0.1,
    format="%.2f"
)

inactivity_label = st.selectbox("Physical inactivity", ["Yes", "No"])
physical_inactivity = 1 if inactivity_label == "Yes" else 0

# =========================
# Prediction
# =========================
if st.button("Predict risk"):
    model = load_model()

    X = np.array([[
        age,
        sex,
        smoking,
        alcohol,
        bmi,
        waist,
        physical_inactivity
    ]])

    probability = model.predict_proba(X)[0, 1]

    st.markdown("### Result")
    st.write(f"**Probability of elevated cholesterol:** `{probability:.2f}`")

    if probability >= 0.5:
        st.error("High risk of elevated cholesterol")
    else:
        st.success("Low risk of elevated cholesterol")
