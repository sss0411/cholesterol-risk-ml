import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Predicting the Risk of Elevated Cholesterol")

# загрузка модели
model = joblib.load("rf_cholesterol_model.pkl")
st.success("Модель успешно загружена")

st.write("Тестовое предсказание (проверка подключения модели)")

if st.button("Run test prediction"):
    # создаём фиктивную строку с нужным числом признаков
    X_dummy = pd.DataFrame(
        np.zeros((1, model.n_features_in_))
    )

    proba = model.predict_proba(X_dummy)[0, 1]
    st.write(f"Вероятность высокого холестерина: **{proba:.3f}**")
