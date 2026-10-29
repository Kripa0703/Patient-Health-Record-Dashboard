# Patient Health Record Dashboard

A staff-facing analytics dashboard built on synthetic patient records (Synthea). It shows
individual patient records, population-level statistics, and a predicted risk of being
diagnosed with one of five chronic conditions: Diabetes, Hypertension, Chronic Kidney
Disease, Hyperlipidemia and Anemia.

> Prototype for academic use on synthetic data. Not for clinical use.

## How the prediction works

- For each patient and condition, features are built only from data recorded **before a
  snapshot date**: two years before the diagnosis, or two years before the last encounter
  for patients who were never diagnosed. This prevents data leakage.
- Features: labs and vitals (last value before the snapshot), flags for related past
  conditions, visit counts, and demographics.
- Sparse labs (glucose, HbA1c, GFR, creatinine, hemoglobin) use a sentinel value (-1) when
  never tested, so "not tested" stays distinguishable from "tested and normal".
- Models per condition: logistic regression, random forest and gradient boosting, with
  class weighting for imbalanced outcomes. The dashboard uses gradient boosting.
- The 124 patients with invalid dates were excluded from modelling.

## Pages

| Page | What it shows |
|---|---|
| Home (`app.py`) | Patient count and diagnosis counts per condition |
| Patient Record | Demographics, tracked diagnoses, condition history, lab trends |
| Population Overview | Age and race distribution, diagnosis rates by group |
| Risk Predictions | Five-condition risk, actual outcome, top factors, model comparison |

## Project structure

```
app.py                  Home page
pages/                  Patient Record, Population Overview, Risk Predictions
utils/data_loader.py    Cached loading of data and models
utils/predict.py        Feature lookup, prediction, risk bands
utils/theme.py          Colours, CSS, chart styling
data/                   Processed parquet files (patients, labels, features, ...)
models/                 Saved models, one .pkl bundle per condition
.streamlit/config.toml  Light theme
```

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501.

## Limitations

- Synthetic data: relationships are simplified compared with real patients.
- Patients marked "Seen in training" on the Risk Predictions page were part of model
  fitting, so their scores are optimistic. "Unseen test patient" rows are the honest view.
- Risk levels (Low / Monitor / Elevated / High) are fixed probability cut-offs for display,
  not clinical thresholds.
- Predictions estimate the likelihood of a diagnosis, not a diagnosis itself.