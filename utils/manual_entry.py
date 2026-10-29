# utils/manual_entry.py
import pandas as pd
import streamlit as st

from utils.data_loader import (
    DISEASES, DISEASE_LABELS, load_features, load_model_bundle, load_numeric_obs,
)
from utils.predict import risk_level

# (model column, "tested" flag column, friendly name)
LABS = [
    ("Body Height", "TESTED_Body_Height", "Height"),
    ("Body Weight", "TESTED_Body_Weight", "Weight"),
    ("Body mass index (BMI) [Ratio]", "TESTED_Body_mass_index_BMI", "BMI"),
    ("Systolic Blood Pressure", "TESTED_Systolic_Blood_Pressure", "Systolic BP"),
    ("Diastolic Blood Pressure", "TESTED_Diastolic_Blood_Pressure", "Diastolic BP"),
    ("Heart rate", "TESTED_Heart_rate", "Heart rate"),
    ("Respiratory rate", "TESTED_Respiratory_rate", "Respiratory rate"),
    ("Glucose [Mass/volume] in Blood", "TESTED_Glucose", "Glucose"),
    ("Hemoglobin A1c/Hemoglobin.total in Blood", "TESTED_Hemoglobin_A1c/Hemoglobin.total_in_Blood", "HbA1c"),
    ("Hemoglobin [Mass/volume] in Blood", "TESTED_Hemoglobin", "Hemoglobin"),
    ("Cholesterol [Mass/volume] in Serum or Plasma", "TESTED_Cholesterol", "Total cholesterol"),
    ("Cholesterol in HDL [Mass/volume] in Serum or Plasma", "TESTED_Cholesterol_in_HDL", "HDL cholesterol"),
    ("Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay", "TESTED_Cholesterol_in_LDL", "LDL cholesterol"),
    ("Triglyceride [Mass/volume] in Serum or Plasma", "TESTED_Triglyceride", "Triglycerides"),
    ("Creatinine [Mass/volume] in Blood", "TESTED_Creatinine", "Creatinine"),
    ("Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M", "TESTED_Glomerular_filtration_rate", "eGFR"),
    ("Urea nitrogen [Mass/volume] in Blood", "TESTED_Urea_nitrogen", "Urea nitrogen"),
    ("Microalbumin/Creatinine [Mass Ratio] in Urine", "TESTED_Microalbumin/Creatinine", "Microalbumin/creatinine"),
]

HIST_CONDITIONS = {
    "HIST_Body_mass_index_30+_-_obesity": "Obesity (BMI 30+)",
    "HIST_Metabolic_syndrome_X": "Metabolic syndrome",
    "HIST_Prediabetes": "Prediabetes",
    "HIST_Diabetes_mellitus_type_2": "Type 2 diabetes",
    "HIST_Disorder_of_kidney_due_to_diabetes_mellitus": "Kidney disorder due to diabetes",
    "HIST_Microalbuminuria_due_to_type_2_diabetes_mellitus": "Microalbuminuria (type 2 diabetes)",
    "HIST_Essential_hypertension": "Essential hypertension",
    "HIST_Ischemic_heart_disease": "Ischemic heart disease",
    "HIST_Chronic_kidney_disease_stage_1": "CKD stage 1",
    "HIST_Chronic_kidney_disease_stage_2": "CKD stage 2",
    "HIST_Chronic_kidney_disease_stage_3": "CKD stage 3",
    "HIST_Chronic_kidney_disease_stage_4": "CKD stage 4",
}
HIST_MEDS = {
    "HIST_MED_metformin": "Metformin",
    "HIST_MED_insulin": "Insulin",
    "HIST_MED_amlodipine": "Amlodipine",
    "HIST_MED_hydrochlorothiazide": "Hydrochlorothiazide",
    "HIST_MED_lisinopril": "Lisinopril",
    "HIST_MED_losartan": "Losartan",
}
HIST_ALL = {**HIST_CONDITIONS, **HIST_MEDS}

# Asian is the reference category (all race dummies 0)
RACE_COLS = {
    "Asian": None, "Black": "RACE_black", "Hawaiian": "RACE_hawaiian",
    "Native": "RACE_native", "Other": "RACE_other", "White": "RACE_white",
}


@st.cache_data(show_spinner=False)
def untested_values(disease: str) -> dict:
    """Value the training pipeline stored for each lab when it was never tested."""
    feats = load_features(disease)
    out = {}
    for col, flag, _ in LABS:
        if col in feats.columns and flag in feats.columns:
            rows = feats.loc[feats[flag] == 0, col]
            out[col] = float(rows.mode().iloc[0]) if len(rows) else None
    return out


@st.cache_data(show_spinner=False)
def lab_units() -> dict:
    o = load_numeric_obs().dropna(subset=["UNITS"])
    return o.groupby("DESCRIPTION")["UNITS"].first().to_dict()


@st.cache_data(show_spinner=False)
def lab_typical() -> dict:
    return load_numeric_obs().groupby("DESCRIPTION")["VALUE"].median().to_dict()


@st.cache_data(show_spinner=False)
def default_visits() -> dict:
    f = load_features("diabetes")
    return {"total": int(f["TOTAL_VISITS"].median()), "emergency": int(f["EMERGENCY_VISITS"].median())}


def _base_row(inputs: dict) -> dict:
    row = {
        "AGE_AT_SNAPSHOT": float(inputs["age"]),
        "GENDER_M": int(inputs["male"]),
        "ETHNICITY_nonhispanic": int(inputs["nonhispanic"]),
        "TOTAL_VISITS": float(inputs["total_visits"]),
        "EMERGENCY_VISITS": float(inputs["emergency_visits"]),
    }
    for col in RACE_COLS.values():
        if col:
            row[col] = 0
    chosen = RACE_COLS[inputs["race"]]
    if chosen:
        row[chosen] = 1
    for col in HIST_ALL:
        row[col] = int(col in inputs["history"])
    return row


def build_rows(inputs: dict) -> dict:
    """One model-ready, correctly ordered feature row per disease."""
    base = _base_row(inputs)
    rows = {}
    for d in DISEASES:
        bundle = load_model_bundle(d)
        cols = bundle["feature_columns"]
        fill = untested_values(d)
        row = dict(base)
        for col, flag, _ in LABS:
            v = inputs["labs"].get(col)
            tested = v is not None and not pd.isna(v)
            row[flag] = int(tested)
            if tested:
                row[col] = float(v)
            else:
                fv = fill.get(col)
                row[col] = fv if fv is not None else bundle.get("lab_medians", {}).get(col, -1.0)
        missing = [c for c in cols if c not in row]
        if missing:
            raise ValueError(f"{d}: model expects columns the form doesn't provide: {missing}")
        rows[d] = pd.DataFrame([row])[cols].astype(float)
    return rows


def predict_manual(inputs: dict) -> pd.DataFrame:
    out = []
    for d, X in build_rows(inputs).items():
        bundle = load_model_bundle(d)
        key = "hgb" if "hgb" in bundle else "rf"
        p = float(bundle[key].predict_proba(X)[0, 1])
        out.append({"key": d, "Condition": DISEASE_LABELS[d], "probability": p,
                    "level": risk_level(p), "model": key})
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def typical_visits(age: float) -> dict:
    """Median visit counts among training patients within 5 years of this age."""
    f = load_features("diabetes")
    near = f[(f["AGE_AT_SNAPSHOT"] - age).abs() <= 5]
    if len(near) < 30:
        near = f
    return {
        "total": int(near["TOTAL_VISITS"].median()),
        "emergency": int(near["EMERGENCY_VISITS"].median()),
        "p5": int(near["TOTAL_VISITS"].quantile(0.05)),
        "n": len(near),
    }