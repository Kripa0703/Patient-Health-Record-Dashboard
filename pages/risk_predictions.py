# pages/risk_predictions.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import DISEASES, DISEASE_LABELS, patient_selector
from utils.predict import actual_outcome, feature_importances, is_holdout, predict_patient
from utils.theme import RISK, SERIES, inject_css, risk_badge, style_fig

st.set_page_config(page_title="Risk Predictions", layout="wide")
inject_css()
st.title("Risk Predictions")

pid = patient_selector()
results = {d: predict_patient(pid, d) for d in DISEASES}

if any(r is None for r in results.values()):
    st.warning("No model features exist for this patient (excluded earlier for invalid dates). Pick another patient.")
    st.stop()


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"


# ---------- Risk cards ----------
st.subheader("Predicted risk of future diagnosis")
cols = st.columns(len(DISEASES))
for col, d in zip(cols, DISEASES):
    r = results[d]
    with col:
        st.metric(DISEASE_LABELS[d], pct(r["probability"]))
        st.markdown(risk_badge(r["level"]), unsafe_allow_html=True)

# ---------- Detail table ----------
rows = []
for d in DISEASES:
    r = results[d]
    out = actual_outcome(pid, d)
    if out is None:
        actual, snap = "—", "—"
    else:
        snap = out["snapshot"].strftime("%Y-%m-%d")
        actual = (
            f"Diagnosed {out['diag_date'].strftime('%Y-%m-%d')}" if out["diagnosed"] else "Not diagnosed"
        )
    rows.append({
        "Condition": DISEASE_LABELS[d],
        "Risk": f"{RISK[r['level']]['mark']} {r['level']}",
        "Probability": r["probability"] * 100,
        "Model": r["primary_model"],
        "Data up to": snap,
        "Actual outcome": actual,
        "Evaluation": "Unseen test patient" if is_holdout(pid, d) else "Seen in training",
    })

st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Probability": st.column_config.ProgressColumn(
            "Probability", min_value=0, max_value=100, format="%.1f%%"
        ),
    },
)
st.caption(
    "Each prediction uses only data recorded up to the 'Data up to' date, which is two years "
    "before the diagnosis (or before the last visit if never diagnosed). 'Seen in training' "
    "patients were part of model fitting, so their scores are optimistic; 'Unseen test patient' "
    "rows are an honest preview. These are risk estimates, not diagnoses."
)

# ---------- Explanation ----------
st.subheader("What drives the prediction")
d = st.selectbox("Condition to explain", DISEASES, format_func=lambda x: DISEASE_LABELS[x])
r = results[d]

imp = feature_importances(d, top=10)
imp["Patient value"] = imp["Feature"].map(r["features"]).round(2)
imp["Note"] = np.where(imp["Patient value"] == -1, "not tested", "")
imp["Short name"] = imp["Feature"].str.slice(0, 55)

left, right = st.columns([3, 2])
with left:
    plot = imp.sort_values("Importance")
    fig = px.bar(
        plot, x="Importance", y="Short name", orientation="h",
        color_discrete_sequence=[SERIES[0]],
        title=f"Top factors for {DISEASE_LABELS[d]} (random forest importance)",
    )
    fig.update_traces(marker_cornerradius=4)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
with right:
    st.dataframe(
        imp[["Short name", "Patient value", "Note"]].rename(columns={"Short name": "Factor"}),
        hide_index=True, use_container_width=True, height=420,
    )

with st.expander("Compare the three models"):
    names = {"logreg": "Logistic regression", "rf": "Random forest", "hgb": "Gradient boosting"}
    comp = pd.DataFrame({
        "Model": [names.get(k, k) for k in r["model_probs"]],
        "Probability": [pct(v) for v in r["model_probs"].values()],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)