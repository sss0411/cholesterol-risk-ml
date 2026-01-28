import streamlit as st
import joblib
import pandas as pd

st.title("Predicting the Risk of Elevated Cholesterol")

# загрузка модели
model = joblib.load("rf_cholesterol_model.pkl")
st.success("Модель успешно загружена")

st.subheader("Введите данные пациента")

# === ВВОД ДАННЫХ ===
age = st.number_input("Age (years)", min_value=0, max_value=120, value=40)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

smoking = st.selectbox(
    "Smoking",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

alcohol = st.selectbox(
    "Alcohol drinking",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

waist = st.number_input(
    "Waist circumference (cm)",
    min_value=40.0,
    max_value=150.0,
    value=90.0
)

physical_inactivity = st.selectbox(
    "Physical inactivity",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# === ПРЕДСКАЗАНИЕ ===
if st.button("Predict risk"):
    input_df = pd.DataFrame(
        [[
            age,
            sex,
            smoking,
            alcohol,
            bmi,
            waist,
            physical_inactivity
        ]],
        columns=[
            'Age',
            'Sex',
            'Smoking',
            'Alcohol_drinking',
            'BMI',
            'Waist_circumference',
            'Physical_inactivity'
        ]
    )

    probability = model.predict_proba(input_df)[0, 1]

    st.subheader("Result")
    st.write(f"Probability of elevated cholesterol: **{probability:.2f}**")

    if probability >= 0.5:
        st.error("High risk of elevated cholesterol")
    else:
        st.success("Low risk of elevated cholesterol")
