import streamlit as st
import pandas as pd
import numpy as np
from phik import phik_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline_transformers import *


@st.cache_resource
def load_model():
    with open("models/LR_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    return lr_model


model = load_model()

st.title("Car Price Prediction App")

st.header("EDA для CSV-файла")

uploaded_eda = st.file_uploader("Загрузите CSV-файл для EDA", type=["csv"])

if uploaded_eda:
    df_eda = pd.read_csv(uploaded_eda)

    st.subheader("Первые строки данных")
    st.write(df_eda.head())

    st.subheader("Статистики числовых признаков (describe)")
    st.write(df_eda.describe())

    st.subheader("Пропуски и дубликаты")
    st.write("Количество пропущенных значений по столбцам:")
    st.write(df_eda.isna().sum())
    st.write(f"Количество дубликатов в датасете: {df_eda.duplicated().sum()}")

    num_cols = df_eda.select_dtypes(include=[np.number]).columns

    st.subheader("Распределение числовых признаков")
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(8, 4 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]
    for i, col in enumerate(num_cols):
        sns.histplot(df_eda[col], kde=True, ax=axes[i])
    st.pyplot(fig)

    corr_method = st.selectbox("Выберите метод корреляции", ["Pearson", "Spearman", "Phik"])

    st.subheader(f"Корреляционная матрица ({corr_method})")
    if corr_method in ["Pearson", "Spearman"]:
        corr_matrix = df_eda[num_cols].corr(method=corr_method.lower())
    elif corr_method == "Phik":
        corr_matrix = df_eda.phik_matrix()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)



st.header("Предсказание для CSV-файла")

uploaded_predict = st.file_uploader("Загрузите CSV для предсказания", type=["csv"], key="pred")

if uploaded_predict:
    df_pred = pd.read_csv(uploaded_predict)
    preds = np.exp(model.predict(df_pred))
    df_pred["predicted_price"] = preds

    st.subheader("Результат:")
    st.write(df_pred)

    csv = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button(label="Скачать результат", data=csv, file_name="predictions.csv")


st.header("Веса обученной модели")

if hasattr(model.named_steps["model"], "coef_"):
    coefs = model.named_steps["model"].coef_

    feature_names = model.named_steps["prep"].get_feature_names_out()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})

    coef_df = coef_df.sort_values("coef", ascending=False)

    st.write(coef_df)
else:
    st.warning("Модель не содержит коэффициентов.")
