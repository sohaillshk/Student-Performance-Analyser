import os
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load student data from CSV if available,
    otherwise generate a synthetic dataset.
    """
    if os.path.exists("student_v2.csv"):
        return pd.read_csv("student_v2.csv")

    np.random.seed(42)
    n     = 150
    study = np.random.uniform(1, 10, n)
    sleep = np.random.uniform(4,  9, n)
    phone = np.random.uniform(0,  8, n)
    marks = (
        study * 7.0 +
        sleep * 3.0 -
        phone * 2.5 +
        np.random.normal(0, 5, n)
    ).clip(20, 100)

    return pd.DataFrame({
        "Study": study.round(1),
        "Sleep": sleep.round(1),
        "Phone": phone.round(1),
        "Marks": marks.round(1),
    })
