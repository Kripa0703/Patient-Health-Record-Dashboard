# utils/data_loader.py
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Project root = the folder that contains app.py (one level above utils/)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# Single source of truth for disease names (file keys) and display labels
DISEASES = ["diabetes", "hypertension", "ckd", "hyperlipidemia", "anemia"]
DISEASE_LABELS = {
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "ckd": "Chronic Kidney Disease",
    "hyperlipidemia": "Hyperlipidemia",
    "anemia": "Anemia",
}


def _read_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        st.error(f"Missing data file: {path}. Copy your parquet files into the data/ folder.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_patients() -> pd.DataFrame:
    return _read_parquet("patients")


@st.cache_data(show_spinner=False)
def load_conditions() -> pd.DataFrame:
    return _read_parquet("conditions")


@st.cache_data(show_spinner=False)
def load_labels() -> pd.DataFrame:
    return _read_parquet("labels")


@st.cache_data(show_spinner=False)
def load_numeric_obs() -> pd.DataFrame:
    return _read_parquet("numeric_obs")


@st.cache_data(show_spinner=False)
def load_features(disease: str) -> pd.DataFrame:
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease: {disease}")
    return _read_parquet(f"features_{disease}")


@st.cache_resource(show_spinner=False)
def load_model_bundle(disease: str) -> dict:
    """Returns the saved dict: logreg, rf, (hgb), scaler, feature_columns, lab_medians."""
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease: {disease}")
    path = MODEL_DIR / f"{disease}_model.pkl"
    if not path.exists():
        st.error(f"Missing model file: {path}. Copy your .pkl files into the models/ folder.")
        st.stop()
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def patient_options() -> dict:
    p = load_patients().sort_values("PATIENT")
    return {r.PATIENT: f"{r.PATIENT[:8]} · {int(r.AGE)} y · {r.GENDER}" for r in p.itertuples()}


def patient_selector() -> str:
    options = patient_options()
    ids = list(options)
    default = st.session_state.get("selected_patient", ids[0])
    pid = st.selectbox(
        "Select patient (type the first characters of an ID to search)",
        ids,
        index=ids.index(default) if default in ids else 0,
        format_func=lambda i: options[i],
    )
    st.session_state["selected_patient"] = pid
    return pid