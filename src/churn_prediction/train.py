from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "sample" / "sample_employee_churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"


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
TARGET_COL = "left"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # basic validation
    missing = set(NUMERIC_COLS + CAT_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")
    return df


def preprocess_fit(df: pd.DataFrame):
    # scale numeric
    scaler = MinMaxScaler()
    X_num = df[NUMERIC_COLS].copy()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=NUMERIC_COLS)

    # one-hot for categorical
    X_cat = pd.get_dummies(df[CAT_COLS], drop_first=False)

    X = pd.concat([X_num_scaled, X_cat], axis=1)
    y = df[TARGET_COL].astype(int)

    columns = X.columns.tolist()
    return X, y, scaler, columns


def preprocess_apply(df: pd.DataFrame, scaler: MinMaxScaler, columns: list[str]) -> pd.DataFrame:
    X_num = pd.DataFrame(scaler.transform(df[NUMERIC_COLS]), columns=NUMERIC_COLS)
    X_cat = pd.get_dummies(df[CAT_COLS], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    # align columns
    return X.reindex(columns=columns, fill_value=0)


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    X, y, scaler, columns = preprocess_fit(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(columns, indent=2))
    print("Saved: models/model.joblib, models/scaler.joblib, models/feature_columns.json")


if __name__ == "__main__":
    train()
