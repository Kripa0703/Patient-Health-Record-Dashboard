# pages/patient_record.py
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import (
    DISEASES, DISEASE_LABELS, load_conditions, load_labels, load_numeric_obs, load_patients,
)
from utils.predict import DISEASE_LABELS_COL
from utils.theme import SERIES, inject_css, style_fig

st.set_page_config(page_title="Patient Record", layout="wide")
inject_css()
st.title("Patient Record")


def fmt_date(x) -> str:
    return "—" if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False)
def patient_options() -> dict:
    p = load_patients().sort_values("PATIENT")
    return {
        r.PATIENT: f"{r.PATIENT[:8]} · {int(r.AGE)} y · {r.GENDER}"
        for r in p.itertuples()
    }


# ---------- Patient picker ----------
options = patient_options()
ids = list(options.keys())
default = st.session_state.get("selected_patient", ids[0])
pid = st.selectbox(
    "Select patient (type the first characters of an ID to search)",
    ids,
    index=ids.index(default) if default in ids else 0,
    format_func=lambda i: options[i],
)
st.session_state["selected_patient"] = pid

p = load_patients().set_index("PATIENT").loc[pid]
deceased = pd.notna(p["DEATHDATE"])

# ---------- Demographics ----------
st.subheader("Demographics")
c = st.columns(5)
c[0].metric("Age", f"{int(p['AGE'])}")
c[1].metric("Gender", {"M": "Male", "F": "Female"}.get(p["GENDER"], str(p["GENDER"])))
c[2].metric("Race", str(p["RACE"]).title())
c[3].metric("Ethnicity", str(p["ETHNICITY"]).title())
c[4].metric("Status", "Deceased" if deceased else "Alive")
st.caption(
    f"Born {fmt_date(p['BIRTHDATE'])}"
    + (f" · Died {fmt_date(p['DEATHDATE'])}" if deceased else "")
    + f" · ID {pid}"
)

# ---------- Tracked diagnoses ----------
st.subheader("Tracked diagnoses")
labels = load_labels().set_index("PATIENT")
if pid in labels.index:
    rows = []
    for d in DISEASES:
        pre = DISEASE_LABELS_COL[d]
        date = labels.loc[pid, f"{pre}_DIAG_DATE"]
        rows.append({
            "Condition": DISEASE_LABELS[d],
            "Diagnosed": "Not diagnosed" if pd.isna(date) else fmt_date(date),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.caption("No outcome labels for this patient (excluded for invalid dates).")

# ---------- Condition history ----------
st.subheader("Condition history")
cond = load_conditions()
cond = cond[cond["PATIENT"] == pid].sort_values("START", ascending=False)
if st.checkbox("Active conditions only", value=False):
    cond = cond[cond["STOP"].isna()]

if cond.empty:
    st.caption("No conditions to show.")
else:
    view = pd.DataFrame({
        "Condition": cond["CONDITION_DESC"].values,
        "Onset": cond["START"].dt.strftime("%Y-%m-%d").values,
        "Resolved": cond["STOP"].dt.strftime("%Y-%m-%d").fillna("Ongoing").values,
    })
    st.dataframe(view, hide_index=True, use_container_width=True, height=300)

# ---------- Lab history ----------
st.subheader("Labs and vitals")
obs = load_numeric_obs()
obs = obs[obs["PATIENT"] == pid]

if obs.empty:
    st.caption("No lab or vital records for this patient.")
else:
    choice = st.selectbox("Lab / vital", sorted(obs["DESCRIPTION"].unique()))
    s = obs[obs["DESCRIPTION"] == choice].sort_values("DATE")
    unit = s["UNITS"].dropna().iloc[0] if s["UNITS"].notna().any() else ""
    title = choice if len(choice) <= 70 else choice[:67] + "..."

    fig = px.line(
        s, x="DATE", y="VALUE", markers=True,
        color_discrete_sequence=[SERIES[0]],
        title=f"{title} ({unit})" if unit else title,
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=unit or None)
    st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

    with st.expander("Latest value of every lab (table view)"):
        latest = obs.sort_values("DATE").groupby("DESCRIPTION").tail(1)
        table = pd.DataFrame({
            "Lab / vital": latest["DESCRIPTION"].values,
            "Value": latest["VALUE"].round(2).values,
            "Units": latest["UNITS"].fillna("").values,
            "Recorded": latest["DATE"].dt.strftime("%Y-%m-%d").values,
        })
        st.dataframe(table, hide_index=True, use_container_width=True)