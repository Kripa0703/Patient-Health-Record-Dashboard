# Patient Health Record Dashboard

A staff-facing analytics dashboard built on synthetic patient records (Synthea). It shows
individual patient records and population statistics, and predicts the risk that a patient
will be diagnosed with one of five chronic conditions: Diabetes, Hypertension, Chronic
Kidney Disease, Hyperlipidemia and Anemia. Staff can also enter a new patient manually to
get a risk estimate.

> Prototype for academic use on synthetic data. Not for clinical use.

## How the prediction works

- For each patient and condition, features are built only from data recorded **before a
  snapshot date**: two years before the diagnosis, or two years before the last encounter
  for patients who were never diagnosed. This prevents data leakage.
- Features: labs and vitals (last value before the snapshot), flags for related past
  conditions and medications, visit counts, and demographics.
- Tests are ordered when a condition is already suspected, so a missing lab carries
  information. For key sparse labs (for example glucose and HbA1c for diabetes) a sentinel
  value (-1) marks "never tested", and "tested" flags are kept, instead of filling with the
  median.
- Models per condition: logistic regression, random forest and gradient boosting, with
  class weighting for imbalanced outcomes. The dashboard uses gradient boosting.
- Patients with invalid dates (124) were excluded, leaving 5,628 labelled patients.

## Results

Evaluated on a stratified 20% held-out test set (1,126 patients), gradient boosting, default
0.5 threshold:

| Condition | ROC-AUC | F1 (diagnosed class) |
|---|---|---|
| Diabetes | 0.942 | 0.676 |
| Hypertension | 0.929 | 0.772 |
| Chronic Kidney Disease | 0.950 | 0.718 |
| Hyperlipidemia | 0.910 | 0.615 |
| Anemia | 0.947 | 0.856 |

Removing the visit-count features lowered ROC-AUC by 0.044 to 0.098, so they carry
substantial signal (see Limitations).

## Pages

| Page | What it shows |
|---|---|
| Home (`app.py`) | Patient count and diagnosis counts per condition |
| Patient Record | Demographics, tracked diagnoses, condition history, lab trends |
| Population Overview | Age and race distribution, diagnosis rates by group |
| Risk Predictions | Five-condition risk, actual outcome, top factors, model comparison |
| New Patient | Enter a patient's details and labs to get a five-condition risk estimate |

## Project structure

```
app.py                    Home page
pages/                    Patient Record, Population Overview, Risk Predictions, New Patient
utils/data_loader.py      Cached loading of data and models
utils/predict.py          Feature lookup, prediction, risk bands
utils/manual_entry.py     Builds model-ready features from the New Patient form
utils/theme.py            Colours, CSS, chart styling
check_manual_entry.py     Test: the form logic reproduces the stored training features
data/                     Processed parquet files (patients, labels, features, ...)
models/                   Saved models, one .pkl bundle per condition
.streamlit/config.toml    Light theme
```

## Run it

Requires Python 3.11 or newer.

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. To verify the New Patient logic, run
`python check_manual_entry.py`; every condition should report 0 mismatches.

The saved models were trained with scikit-learn 1.6.1, so keep that version pinned.

## Limitations

- Synthetic data: relationships are simpler than in real patients.
- Patients marked "Seen in training" on the Risk Predictions page were part of model
  fitting, so their scores are optimistic. "Unseen test patient" rows are the honest view.
- Visit counts and the number of labs on record both grow with the length of a patient's
  record, and record length differs between diagnosed and undiagnosed patients, so these
  features mix clinical activity with record length. A visits-per-year feature would
  separate the two and is future work.
- Very few patients in the training data had unusually low visit counts, and most of them
  were later diagnosed. The New Patient form warns when a count is unusually low for the
  patient's age, because scores there are unreliable.
- Risk levels (Low / Monitor / Elevated / High) are fixed probability cut-offs for display,
  not clinical thresholds, and are not adjusted for each condition's base rate.
- Predictions estimate the likelihood of a diagnosis, not a diagnosis itself.

**Live demo:** https://patient-health-record-dashboard.streamlit.app/