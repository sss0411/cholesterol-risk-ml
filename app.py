import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load trained model
# ===============================
model = joblib.load("rf_cholesterol_model.pkl")

# ===============================
# App title
# ===============================
st.title("Predicting the Risk of Elevated Cholesterol")
st.write("Enter patient data")

# ===============================
# User inputs
# ===============================
age = st.number_input("Age (years)", min_value=18, max_value=100, value=40)

sex = st.selectbox("Sex", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol drinking", ["No", "Yes"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
waist = st.number_input(
    "Waist circumference (cm)",
    min_value=50.0,
    max_value=150.0,
    value=90.0
)

physical_inactivity = st.selectbox("Physical inactivity", ["No", "Yes"])

# ===============================
# Build input строго по модели
# ===============================
# model.feature_names_in_ ==
# ['age', 'sex', 'smoking', 'alcohol_drinking',
#  'bmi', 'waist_circumference', 'physical_inactivity']

# ===============================
# Build input EXACTLY as model expects
# ===============================
input_data = pd.DataFrame(
    [[
        float(age),
        1 if sex == "Male" else 0,
        1 if smoking == "Yes" else 0,
        1 if alcohol == "Yes" else 0,
        float(bmi),
        float(waist),
        1 if physical_inactivity == "Yes" else 0
    ]],
    columns=model.feature_names_in_
)



# ===============================
# Prediction
# ===============================
if st.button("Predict risk"):

    # CHOL = 1 → elevated cholesterol
    risk_label = 1
    risk_index = list(model.classes_).index(risk_label)

    probability = model.predict_proba(input_data)[0, risk_index]

    # ===============================
    # Output
    # ===============================
    st.markdown("### Result")
    st.write(f"**Probability of elevated cholesterol:** {probability:.2f}")

    # Thresholds adapted to class imbalance (~11.8%)
    if probability >= 0.20:
        st.error("High risk of elevated cholesterol")
    elif probability >= 0.10:
        st.warning("Moderate risk of elevated cholesterol")
    else:
        st.success("Low risk of elevated cholesterol")
