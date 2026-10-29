# utils/predict.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from utils.data_loader import (
    DISEASES, DISEASE_LABELS, load_features, load_labels, load_model_bundle,
)

# Probability cut-offs for the display bands (tunable, not clinical thresholds)
RISK_BANDS = [(0.25, "Low"), (0.50, "Monitor"), (0.75, "Elevated"), (1.01, "High")]


def risk_level(prob: float) -> str:
    for cutoff, level in RISK_BANDS:
        if prob < cutoff:
            return level
    return "High"


@st.cache_data(show_spinner=False)
def _features_indexed(disease: str) -> pd.DataFrame:
    return load_features(disease).set_index("PATIENT")


def _primary_key(bundle: dict) -> str:
    return "hgb" if "hgb" in bundle else "rf"


def predict_patient(patient_id: str, disease: str) -> dict | None:
    feats = _features_indexed(disease)
    if patient_id not in feats.index:
        return None

    bundle = load_model_bundle(disease)
    cols = bundle["feature_columns"]
    row = feats.loc[[patient_id]].reindex(columns=cols).astype(float)

    # Safety net: fill any gaps with training medians (features are normally complete)
    medians = bundle.get("lab_medians", {})
    row = row.fillna(pd.Series({c: medians.get(c, 0.0) for c in cols}))

    probs = {}
    if "logreg" in bundle:
        probs["logreg"] = float(bundle["logreg"].predict_proba(bundle["scaler"].transform(row))[0, 1])
    if "rf" in bundle:
        probs["rf"] = float(bundle["rf"].predict_proba(row)[0, 1])
    if "hgb" in bundle:
        probs["hgb"] = float(bundle["hgb"].predict_proba(row)[0, 1])

    primary = _primary_key(bundle)
    prob = probs[primary]
    return {
        "disease": disease,
        "label": DISEASE_LABELS[disease],
        "probability": prob,
        "level": risk_level(prob),
        "primary_model": primary,
        "model_probs": probs,
        "features": row.iloc[0],
    }


def predict_all(patient_id: str) -> pd.DataFrame:
    rows = []
    for d in DISEASES:
        r = predict_patient(patient_id, d)
        if r is not None:
            rows.append({
                "Disease": r["label"],
                "Probability": r["probability"],
                "Risk": r["level"],
                "Model": r["primary_model"],
                "_key": d,
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _holdout_ids(disease: str) -> set:
    """Patients in the 20% test split, reproduced with the training settings."""
    feats = load_features(disease)
    labels = load_labels().set_index("PATIENT")[f"{DISEASE_LABELS_COL[disease]}_LABEL"]
    y = feats["PATIENT"].map(labels)
    _, test_ids = train_test_split(
        feats["PATIENT"], test_size=0.2, random_state=42, stratify=y
    )
    return set(test_ids)


# labels.parquet uses these prefixes
DISEASE_LABELS_COL = {
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "ckd": "CKD",
    "hyperlipidemia": "Hyperlipidemia",
    "anemia": "Anemia",
}


def is_holdout(patient_id: str, disease: str) -> bool:
    return patient_id in _holdout_ids(disease)


def actual_outcome(patient_id: str, disease: str) -> dict | None:
    """Ground truth from labels.parquet: did they get diagnosed, and the snapshot date."""
    labels = load_labels().set_index("PATIENT")
    if patient_id not in labels.index:
        return None
    p = DISEASE_LABELS_COL[disease]
    row = labels.loc[patient_id]
    return {
        "diagnosed": bool(row[f"{p}_LABEL"]),
        "snapshot": row[f"{p}_SNAPSHOT"],
        "diag_date": row[f"{p}_DIAG_DATE"],
    }


def feature_importances(disease: str, top: int = 10) -> pd.DataFrame:
    """Global importances from the random forest, for the explainability panel."""
    bundle = load_model_bundle(disease)
    imp = pd.Series(bundle["rf"].feature_importances_, index=bundle["feature_columns"])
    out = imp.sort_values(ascending=False).head(top).rename("Importance").reset_index()
    return out.rename(columns={"index": "Feature"})