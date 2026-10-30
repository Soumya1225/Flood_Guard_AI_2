"""GIS risk map: combine the v2 flood model with district geography.

For every Odisha district it writes:
  * flood_freq_per_year   - how often floods started there (training years, India Flood Inventory)
  * latest_probability    - model probability of a flood starting in the next 3 days on the latest data day
  * peak_probability_30d  - highest probability in the last 30 days of data
  * static indicators from the existing spatial layer (elevation, river distance, ...)
  * risk_level            - Low / Moderate / High / Very High from the combined score
Outputs: gis/odisha_flood_risk_v2.gpkg (open in QGIS / served by the backend) and gis/odisha_flood_risk_v2.csv

    python gis/06_build_risk_map.py          (after pipeline_v2 steps 1-3)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_v2.common import DATASET, HORIZON, MODELS, load_districts  # noqa: E402

LEVELS = [(0.75, "Very High"), (0.5, "High"), (0.25, "Moderate"), (0.0, "Low")]


def main():
    bundle = joblib.load(MODELS / "flood_onset_model_v2.pkl")
    feats, model = bundle["features"], bundle["model"]
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df[feats] = df[feats].fillna(pd.Series(bundle["medians"]))
    last = df["date"].max()
    recent = df[df["date"] > last - pd.Timedelta(days=30)].copy()
    recent["probability"] = model.predict_proba(recent[feats])[:, 1]
    latest = recent[recent["date"] == last].set_index("district")["probability"]
    peak = recent.groupby("district")["probability"].max()
    freq = df.groupby("district")["district_flood_freq"].first()

    g = load_districts().set_index("district")
    g["flood_freq_per_year"] = freq
    g["latest_probability"] = latest.round(4)
    g["peak_probability_30d"] = peak.round(4)
    static = [c for c in df.columns if c.startswith("static_")]
    for c in static:
        g[c] = df.groupby("district")[c].first()

    def rank(s):
        return s.rank(pct=True).fillna(0)
    score = 0.5 * rank(g["flood_freq_per_year"]) + 0.5 * rank(g["peak_probability_30d"])
    g["risk_score"] = score.round(3)
    g["risk_level"] = [next(name for cut, name in LEVELS if v >= cut) for v in score]
    g["data_date"] = str(last.date())
    g["model"] = bundle["name"]
    g = g.reset_index()
    g.to_file(ROOT / "gis" / "odisha_flood_risk_v2.gpkg", driver="GPKG")
    g.drop(columns="geometry").to_csv(ROOT / "gis" / "odisha_flood_risk_v2.csv", index=False)
    print(f"Risk map for {len(g)} districts, data up to {last.date()} (model: {bundle['name']}, {HORIZON}-day horizon)")
    print(g["risk_level"].value_counts().to_string())
    print(g.sort_values("risk_score", ascending=False)[["district_name", "flood_freq_per_year",
                                                        "peak_probability_30d", "risk_level"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
