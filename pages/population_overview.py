# pages/population_overview.py
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import DISEASES, DISEASE_LABELS, load_labels, load_patients
from utils.predict import DISEASE_LABELS_COL
from utils.theme import BG, SERIES, inject_css, style_fig

st.set_page_config(page_title="Population Overview", layout="wide")
inject_css()
st.title("Population Overview")

patients = load_patients().copy()
labels = load_labels()

patients["Gender"] = patients["GENDER"].map({"M": "Male", "F": "Female"}).fillna(patients["GENDER"])
patients["Race"] = patients["RACE"].astype(str).str.title()
patients["Ethnicity"] = patients["ETHNICITY"].astype(str).str.title()

# ---------- Summary cards ----------
c = st.columns(4)
c[0].metric("Patients", f"{len(patients):,}")
c[1].metric("Median age", f"{patients['AGE'].median():.0f}")
c[2].metric("Female", f"{(patients['Gender'] == 'Female').mean() * 100:.0f}%")
c[3].metric("Deceased", f"{patients['DEATHDATE'].notna().sum():,}")

# ---------- Demographics ----------
st.subheader("Who is in the dataset")
left, right = st.columns(2)

with left:
    fig = px.histogram(
        patients, x="AGE", color="Gender", nbins=20,
        color_discrete_map={"Male": SERIES[0], "Female": SERIES[1]},
        title="Age distribution",
    )
    fig.update_traces(marker_line_color=BG, marker_line_width=1.5)
    fig.update_layout(bargap=0.05)
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01), legend_title_text="")
    fig.update_xaxes(title="Age")
    fig.update_yaxes(title="Patients")
    st.plotly_chart(style_fig(fig), use_container_width=True)

with right:
    race = patients["Race"].value_counts().sort_values().reset_index()
    race.columns = ["Race", "Patients"]
    fig = px.bar(
        race, x="Patients", y="Race", orientation="h", text="Patients",
        color_discrete_sequence=[SERIES[0]], title="Patients by race",
    )
    fig.update_traces(textposition="outside", marker_cornerradius=4)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    st.plotly_chart(style_fig(fig), use_container_width=True)

# ---------- Diagnosis rates ----------
st.subheader("Diagnosis rates by group")
f1, f2 = st.columns(2)
disease = f1.selectbox("Condition", DISEASES, format_func=lambda d: DISEASE_LABELS[d])
group_by = f2.selectbox("Group by", ["Age band", "Gender", "Race", "Ethnicity"])

label_col = f"{DISEASE_LABELS_COL[disease]}_LABEL"
df = patients.merge(labels[["PATIENT", label_col]], on="PATIENT", how="inner")
df["Age band"] = pd.cut(
    df["AGE"], bins=[-1, 17, 34, 49, 64, 79, 200],
    labels=["0-17", "18-34", "35-49", "50-64", "65-79", "80+"],
)

summary = (
    df.groupby(group_by, observed=True)[label_col]
    .agg(Rate="mean", Diagnosed="sum", Patients="count")
    .reset_index()
)
summary["Rate (%)"] = summary["Rate"] * 100
summary["Text"] = summary.apply(lambda r: f"{r['Rate (%)']:.1f}%  (n={int(r['Patients'])})", axis=1)

fig = px.bar(
    summary, x=group_by, y="Rate (%)", text="Text",
    hover_data={"Diagnosed": True, "Patients": True, "Rate (%)": ":.1f", "Text": False},
    color_discrete_sequence=[SERIES[0]],
    title=f"{DISEASE_LABELS[disease]}: share of patients diagnosed, by {group_by.lower()}",
)
fig.update_traces(textposition="outside", marker_cornerradius=4)
fig.update_xaxes(title=None)
fig.update_yaxes(title="Diagnosed (%)")
st.plotly_chart(style_fig(fig, height=400), use_container_width=True)

st.caption(
    f"Rates use the {len(labels):,} patients with usable labels. "
    "Groups with a small n give unstable percentages, so read them with caution. "
    "Age is each patient's current (or at-death) age."
)

with st.expander("Table view"):
    st.dataframe(
        summary[["Rate (%)", "Diagnosed", "Patients"]].assign(**{group_by: summary[group_by]})
        [[group_by, "Diagnosed", "Patients", "Rate (%)"]].round(1),
        hide_index=True, use_container_width=True,
    )