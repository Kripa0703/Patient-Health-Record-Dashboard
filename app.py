# app.py
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import DISEASES, DISEASE_LABELS, load_labels, load_patients
from utils.predict import DISEASE_LABELS_COL
from utils.theme import SERIES, inject_css, style_fig

st.set_page_config(page_title="Patient Health Record Dashboard", layout="wide")
inject_css()

patients = load_patients()
labels = load_labels()

st.title("Patient Health Record Dashboard")
st.write(
    "A staff-facing analytics tool. Browse individual patient records, explore the "
    "population, and see each patient's predicted risk of being diagnosed with one of "
    "five chronic conditions, based only on data recorded two years before diagnosis. "
    "This is a prototype built on synthetic (Synthea) data and is not for clinical use."
)

# --- Summary cards ---
counts = {
    d: int(labels[f"{DISEASE_LABELS_COL[d]}_LABEL"].sum()) for d in DISEASES
}
cols = st.columns(len(DISEASES) + 1)
cols[0].metric("Patients", f"{len(patients):,}")
for col, d in zip(cols[1:], DISEASES):
    col.metric(DISEASE_LABELS[d], f"{counts[d]:,}")
# replace:  st.caption("Cards show the number of patients later diagnosed with each condition.")
st.caption(
    f"Cards show patients later diagnosed with each condition. Rates are computed over the "
    f"{len(labels):,} patients with usable labels ({len(patients) - len(labels):,} records "
    f"were excluded for invalid dates)."
)

# delete:  st.info("Use the sidebar to open Patient Record, Population Overview, or Risk Predictions.")

# --- Diagnosis rate chart ---
rate = pd.DataFrame({
    "Disease": [DISEASE_LABELS[d] for d in DISEASES],
    "Diagnosed": [counts[d] for d in DISEASES],
})
rate["Rate (%)"] = rate["Diagnosed"] / len(labels) * 100
rate = rate.sort_values("Rate (%)")

fig = px.bar(
    rate, x="Rate (%)", y="Disease", orientation="h",
    text=rate["Rate (%)"].map("{:.1f}%".format),
    hover_data={"Diagnosed": True, "Rate (%)": ":.1f"},
    color_discrete_sequence=[SERIES[0]],
    title="Share of patients diagnosed, by condition",
)
fig.update_traces(textposition="outside", marker_cornerradius=4)
fig.update_xaxes(title=None)
fig.update_yaxes(title=None)
st.plotly_chart(style_fig(fig, height=320), use_container_width=True)

