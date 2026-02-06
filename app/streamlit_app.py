import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
COLS_PATH = MODELS_DIR / "feature_columns.json"

NUMERIC_COLS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
]
CAT_COLS = ["departments", "salary"]

DEPARTMENTS = ("sales", "technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting", "hr", "management")
SALARY_LEVELS = ("low", "medium", "high")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = json.loads(COLS_PATH.read_text())
    return model, scaler, columns


def build_features(input_df: pd.DataFrame, scaler, columns):
    # scale numeric
    X_num = pd.DataFrame(scaler.transform(input_df[NUMERIC_COLS]), columns=NUMERIC_COLS)
    X_cat = pd.get_dummies(input_df[CAT_COLS], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    return X.reindex(columns=columns, fill_value=0)


def main():
    st.set_page_config(page_title="Employee Churn Prediction", layout="centered")
    st.title("Employee Churn Prediction")

    st.markdown(
        "Predict whether an employee is likely to leave the company based on HR metrics. "
        "This is a portfolio demo using a synthetic sample dataset and a scikit-learn model."
    )

    model, scaler, columns = load_artifacts()

    satisfaction_level = st.slider("Satisfaction level", 0.0, 1.0, step=0.01)
    last_evaluation = st.slider("Last evaluation", 0.0, 1.0, step=0.01)
    number_project = st.slider("Number of projects", 0, 10, step=1, value=4)
    average_montly_hours = st.slider("Average monthly hours", 50, 350, step=1, value=160)
    time_spend_company = st.slider("Years at company", 0, 15, step=1, value=4)
    work_accident = st.selectbox("Work accident?", ("No", "Yes"))
    promotion_last_5years = st.selectbox("Promotion in last 5 years?", ("No", "Yes"))
    departments = st.selectbox("Department", DEPARTMENTS)
    salary = st.selectbox("Salary level", SALARY_LEVELS)

    df_input = pd.DataFrame([{
        "satisfaction_level": satisfaction_level,
        "last_evaluation": last_evaluation,
        "number_project": number_project,
        "average_montly_hours": average_montly_hours,
        "time_spend_company": time_spend_company,
        "Work_accident": 1 if work_accident == "Yes" else 0,
        "promotion_last_5years": 1 if promotion_last_5years == "Yes" else 0,
        "departments": departments,
        "salary": salary,
    }])

    st.subheader("Model input")
    st.dataframe(df_input)

    X = build_features(df_input, scaler, columns)
    proba = float(model.predict_proba(X)[0, 1])
    pred = 1 if proba >= 0.5 else 0

    st.subheader("Prediction")
    st.write(f"Churn probability: **{proba:.2%}**")

    if pred == 1:
        st.error("⚠️ High risk: this employee is likely to churn.")
    else:
        st.success("✅ Low risk: this employee is likely to stay.")

    with st.expander("Feature vector (debug)"):
        st.dataframe(X)


if __name__ == "__main__":
    main()
