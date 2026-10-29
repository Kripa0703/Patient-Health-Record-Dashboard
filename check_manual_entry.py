# check_manual_entry.py
import numpy as np

from utils.data_loader import DISEASES, load_features, load_model_bundle
from utils.manual_entry import HIST_ALL, LABS, RACE_COLS, build_rows

for d in DISEASES:
    feats = load_features(d)
    cols = load_model_bundle(d)["feature_columns"]
    sample = feats.sample(300, random_state=0)
    bad, first = 0, None
    for _, r in sample.iterrows():
        inputs = {
            "age": r["AGE_AT_SNAPSHOT"],
            "male": bool(r["GENDER_M"]),
            "nonhispanic": bool(r["ETHNICITY_nonhispanic"]),
            "race": next((n for n, c in RACE_COLS.items() if c and r.get(c, 0) == 1), "Asian"),
            "labs": {col: (r[col] if r.get(flag, 0) == 1 else None) for col, flag, _ in LABS if col in r.index},
            "history": {c for c in HIST_ALL if r.get(c, 0) == 1},
            "total_visits": r["TOTAL_VISITS"],
            "emergency_visits": r["EMERGENCY_VISITS"],
        }
        built = build_rows(inputs)[d].iloc[0]
        stored = r[cols].astype(float)
        if not np.allclose(built.values, stored.values, atol=1e-6):
            bad += 1
            if first is None:
                first = [c for c in cols if abs(built[c] - stored[c]) > 1e-6]
    print(f"{d}: {bad} mismatches out of {len(sample)}", "" if first is None else f"-> e.g. {first}")