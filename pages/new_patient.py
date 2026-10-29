# pages/new_patient.py
import pandas as pd
import streamlit as st

from utils.data_loader import DISEASES, DISEASE_LABELS
from utils.manual_entry import (
    HIST_CONDITIONS, HIST_MEDS, LABS, RACE_COLS,
    typical_visits, lab_typical, lab_units, predict_manual,
)
from utils.theme import RISK, inject_css, risk_badge

st.set_page_config(page_title="New Patient", layout="wide")
inject_css()
st.title("New Patient")
st.write(
    "Enter a patient's details to estimate their risk of future diagnosis."
    "Leave any lab blank if it was not measured."
)

units = lab_units()
typical = lab_typical()

VITALS = {
    "Body Height", "Body Weight", "Body mass index (BMI) [Ratio]",
    "Systolic Blood Pressure", "Diastolic Blood Pressure",
    "Heart rate", "Respiratory rate",
    "Hemoglobin [Mass/volume] in Blood",
}

with st.form("new_patient"):
    st.subheader("Demographics")
    a, b, c, d = st.columns(4)
    age = a.number_input("Age (years)", min_value=0, max_value=110, value=50, step=1)
    sex = b.selectbox("Sex", ["Female", "Male"])
    race = c.selectbox("Race", list(RACE_COLS), index=list(RACE_COLS).index("White"))
    ethnicity = d.selectbox("Ethnicity", ["Non-Hispanic", "Hispanic"])

    st.subheader("Visit history")
    v1, v2 = st.columns(2)
    total_visits = v1.number_input(
        "Total visits on record (optional)", min_value=1, value=None, step=1,
        help="Leave blank to use the typical count for a patient of this age.",
    )
    emergency_visits = v2.number_input(
        "Emergency visits (optional)", min_value=0, value=None, step=1,
        help="Leave blank to use the typical count for a patient of this age.",
    )

    st.subheader("Labs and vitals")
    st.caption("Blank means 'not tested'. The models treat that as information, not as a normal value.")
    lab_inputs = {}
    lab_cols = st.columns(3)
    for i, (col, flag, name) in enumerate(LABS):
        unit = units.get(col, "")
        t = typical.get(col)
        start = round(float(t), 1) if (col in VITALS and t is not None) else None
        lab_inputs[col] = lab_cols[i % 3].number_input(
            f"{name} ({unit})" if unit else name,
            value=start, format="%.2f", key=f"lab_{i}",
            help=f"Typical value in this dataset: {t:.1f}" if t is not None else None,
        )

    st.subheader("Past conditions and medications")
    h1, h2 = st.columns(2)
    conditions = h1.multiselect("Past conditions", list(HIST_CONDITIONS.values()))
    meds = h2.multiselect("Past medications", list(HIST_MEDS.values()))

    submitted = st.form_submit_button("Predict risk", type="primary")

if submitted:
    typ = typical_visits(age)
    used_total = total_visits if total_visits is not None else typ["total"]
    used_emerg = emergency_visits if emergency_visits is not None else typ["emergency"]

    reverse = {v: k for k, v in {**HIST_CONDITIONS, **HIST_MEDS}.items()}
    inputs = {
        "age": age,
        "male": sex == "Male",
        "race": race,
        "nonhispanic": ethnicity == "Non-Hispanic",
        "total_visits": used_total,
        "emergency_visits": used_emerg,
        "labs": lab_inputs,
        "history": {reverse[x] for x in conditions + meds},
    }
    res = predict_manual(inputs)
    n_labs = sum(v is not None for v in lab_inputs.values())

    defaulted = []
    if total_visits is None:
        defaulted.append(f"{typ['total']} total visits")
    if emergency_visits is None:
        defaulted.append(f"{typ['emergency']} emergency visits")
    if defaulted:
        st.info(
            "Left blank, so the typical values for patients around age "
            f"{age} were used: {' and '.join(defaulted)} (from {typ['n']:,} patients)."
        )
    if used_total < typ["p5"]:
        st.warning(
            f"{used_total} visits is unusually low for age {age}: fewer than 5% of similar-age "
            f"training patients had so few (the typical count is {typ['total']}). The training data "
            "has very little coverage here, so the scores below are unreliable."
        )

    st.subheader("Predicted risk of future diagnosis")
    if n_labs < 5:
        st.warning(
            f"Only {n_labs} of {len(LABS)} labs entered. With so little measured data the "
            "estimate depends mostly on demographics and history."
        )

    cols = st.columns(len(DISEASES))
    for col, (_, r) in zip(cols, res.iterrows()):
        with col:
            st.metric(r["Condition"], f"{r['probability'] * 100:.1f}%")
            st.markdown(risk_badge(r["level"]), unsafe_allow_html=True)

    st.dataframe(
        pd.DataFrame({
            "Condition": res["Condition"],
            "Risk": [f"{RISK[l]['mark']} {l}" for l in res["level"]],
            "Probability": res["probability"] * 100,
            "Model": res["model"],
        }),
        hide_index=True, use_container_width=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
    st.caption(
        f"Based on {n_labs} entered lab values plus the history above. Real patient predictions "
        "use their full record, so fewer inputs mean less reliable estimates. These are risk "
        "estimates from a prototype trained on synthetic data, not diagnoses."
    )