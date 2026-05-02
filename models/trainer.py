import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

FEATURE_COLS = ["Study", "Sleep", "Phone"]


@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a Linear Regression model on the student dataset.
    Returns: model, r2, mae, X_test, y_test
    """
    X = df[FEATURE_COLS]
    y = df["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predict = model.predict(X_test)
    r2  = round(r2_score(y_test, predict), 3)
    mae = round(mean_absolute_error(y_test, predict), 2)

    return model, r2, mae, X_test, y_test


def predict(model, study: float, sleep: float, phone: float) -> float:
    """Run a single prediction and clip result to [0, 100]."""
    inp  = pd.DataFrame([[study, sleep, phone]], columns=FEATURE_COLS)
    pred = float(model.predict(inp)[0])
    return round(float(np.clip(pred, 0, 100)), 1)


def get_grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 60: return "B"
    if score >= 40: return "C"
    return "F"

