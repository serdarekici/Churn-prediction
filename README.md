# Employee Churn Prediction (Portfolio Project)

A compact, production-style **classification** project that predicts whether an employee is likely to leave the company ("churn") based on HR metrics.

This repository contains:
- A clean training pipeline (scikit-learn)
- Saved model artifacts (model + scaler + feature schema)
- A Streamlit demo app
- A small **synthetic sample dataset** for reproducible runs

---

## Problem

Employee churn can be costly due to recruiting, onboarding, and lost productivity.  
Using historical HR signals (satisfaction, evaluation, workload, tenure, promotions, etc.), we estimate the likelihood that an employee will churn.

---

## Data

For portfolio purposes, this repo ships with a synthetic dataset (same schema as common HR churn datasets):

- `satisfaction_level` (0–1)
- `last_evaluation` (0–1)
- `number_project` (int)
- `average_montly_hours` (int)
- `time_spend_company` (int)
- `Work_accident` (0/1)
- `promotion_last_5years` (0/1)
- `departments` (categorical)
- `salary` (categorical)
- `left` (target, 0/1)

Sample data: `data/sample/sample_employee_churn.csv`

---

## Project Structure

```text
churn-prediction/
  app/
    streamlit_app.py
  data/
    sample/
      sample_employee_churn.csv
  models/
    model.joblib
    scaler.joblib
    feature_columns.json
  notebooks/
    Churn_Prediction_original.ipynb
  src/
    churn_prediction/
      __init__.py
      train.py
  requirements.txt
  README.md
```

---

## Quickstart

### 1) Create environment & install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train the model (creates artifacts in `models/`)

```bash
python -m src.churn_prediction.train
```

### 3) Run Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Notes

- The Streamlit demo loads:
  - `models/model.joblib`
  - `models/scaler.joblib`
  - `models/feature_columns.json`
- The model is intentionally simple (Gradient Boosting) for clarity and speed.
- Swap in a real dataset easily by keeping the same column schema.

