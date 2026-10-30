"""Flood Early Warning v2 API: district flood warnings from the model trained on REAL flood events.

Model: models/flood_onset_model_v2.pkl (pipeline_v2), trained on India Flood Inventory events 2000-2016,
tested on 2017-2023. Endpoints:
  GET  /api/v2/warnings              latest warning for every district (probability, alert, main drivers)
  GET  /api/v2/district/{name}       last 60 days of rainfall and warning probability for one district
  POST /api/v2/what-if               the SAME model re-run with changed rainfall (true ML what-if)
  GET  /api/v2/model-report          test-year results and the comparison with the rainfall-rule baseline
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "flood_onset_model_v2.pkl"
DATASET = ROOT / "Datasets" / "processed_data" / "district_daily_v2.csv.gz"
REPORTS = ROOT / "models" / "v2_reports"
router = APIRouter(prefix="/api/v2", tags=["flood-warning-v2"])

LABELS = {"rain_mm": "Rain today", "rain_3d": "Rain, last 3 days", "rain_7d": "Rain, last 7 days",
          "rain_15d": "Rain, last 15 days", "rain_30d": "Rain, last 30 days", "rain_max_3d": "Heaviest day (3 days)",
          "rain_api": "Soil wetness index", "rain_7d_anomaly": "Rain above normal (7 days)",
          "rain_7d_normal": "Normal 7-day rain", "district_flood_freq": "Past flood frequency",
          "humidity": "Humidity", "humidity_3d": "Humidity (3 days)", "temp_c": "Temperature", "wind_ms": "Wind",
          "doy_sin": "Season", "doy_cos": "Season", "monsoon": "Monsoon season",
          "static_elevation_m": "Elevation", "static_distance_to_river_m": "Distance to river"}


@lru_cache(maxsize=1)
def bundle():
    if not MODEL.exists():
        raise HTTPException(503, "v2 model not trained yet: run  python -m pipeline_v2.run_all")
    return joblib.load(MODEL)


@lru_cache(maxsize=1)
def data():
    df = pd.read_csv(DATASET, parse_dates=["date"])
    b = bundle()
    df[b["features"]] = df[b["features"]].fillna(pd.Series(b["medians"]))
    return df


def level(p, thr):
    lv = bundle().get("levels")
    if lv:
        return "Severe" if p >= lv["Severe"] else "Warning" if p >= lv["Warning"] else "Watch" if p >= lv["Watch"] else "Normal"
    if p >= thr * 2:
        return "Severe"
    if p >= thr:
        return "Warning"
    if p >= thr * 0.5:
        return "Watch"
    return "Normal"


def drivers(row, b, top=3):
    """Main reasons for this district's probability (logistic regression: coefficient x standardised value)."""
    m, feats = b["model"], b["features"]
    x = row[feats].astype(float).to_numpy()[None, :]
    if hasattr(m, "named_steps"):
        z = m[0].transform(pd.DataFrame(x, columns=feats))[0]
        contrib = z * m[-1].coef_[0]
    elif hasattr(m, "feature_importances_"):
        med = np.array([b["medians"][f] for f in feats])
        contrib = m.feature_importances_ * np.sign(x[0] - med)
    else:
        return []
    order = np.argsort(-contrib)[:top]
    return [{"feature": LABELS.get(feats[i], feats[i]), "value": round(float(x[0][i]), 1),
             "effect": "raises" if contrib[i] > 0 else "lowers"} for i in order if contrib[i] > 0]


@router.get("/warnings")
def warnings(date: str | None = None):
    b, df = bundle(), data()
    day = pd.Timestamp(date) if date else df["date"].max()
    today = df[df["date"] == day].copy()
    if today.empty:
        raise HTTPException(404, f"No data for {day.date()}")
    today["probability"] = b["model"].predict_proba(today[b["features"]])[:, 1]
    out = []
    for _, r in today.sort_values("probability", ascending=False).iterrows():
        out.append({"district": r["district_name"], "probability": round(float(r["probability"]), 4),
                    "level": level(r["probability"], b["threshold"]), "alert": bool(r["probability"] >= b["threshold"]),
                    "rain_today_mm": round(float(r["rain_mm"]), 1), "rain_3d_mm": round(float(r["rain_3d"]), 1),
                    "drivers": drivers(r, b)})
    return {"date": str(day.date()), "model": b["name"], "question": b.get("question", ""),
            "threshold": round(float(b["threshold"]), 4), "levels": b.get("levels"), "districts": out,
            "note": "Data ends where the downloaded IMD archive ends; pick any past date with ?date=YYYY-MM-DD."}


@router.get("/district/{name}")
def district(name: str, days: int = 60, date: str | None = None):
    b, df = bundle(), data()
    d = df[df["district_name"].str.lower() == name.lower()]
    if d.empty:
        raise HTTPException(404, f"Unknown district {name}")
    end = pd.Timestamp(date) if date else d["date"].max()
    d = d[(d["date"] <= end) & (d["date"] > end - pd.Timedelta(days=days))].copy()
    d["probability"] = b["model"].predict_proba(d[b["features"]])[:, 1]
    return {"district": name, "threshold": float(b["threshold"]),
            "series": [{"date": str(r.date.date()), "rain_mm": round(float(r.rain_mm), 1),
                        "probability": round(float(r.probability), 4), "flood_reported": bool(r.flood_active)}
                       for r in d.itertuples()]}


class WhatIf(BaseModel):
    district: str
    date: str | None = None
    rain_change_pct: float = 0.0          # e.g. 50 = 50% more rain in the last 30 days
    extra_rain_today_mm: float = 0.0


@router.post("/what-if")
def what_if(q: WhatIf):
    b, df = bundle(), data()
    d = df[df["district_name"].str.lower() == q.district.lower()]
    if d.empty:
        raise HTTPException(404, f"Unknown district {q.district}")
    day = pd.Timestamp(q.date) if q.date else d["date"].max()
    row = d[d["date"] == day]
    if row.empty:
        raise HTTPException(404, f"No data for {day.date()}")
    base = row.iloc[0].copy()
    new = base.copy()
    f = 1 + q.rain_change_pct / 100.0
    for c in ["rain_mm", "rain_3d", "rain_7d", "rain_15d", "rain_30d", "rain_max_3d", "rain_api"]:
        if c in new:
            new[c] = base[c] * f
    for c in ["rain_mm", "rain_3d", "rain_7d", "rain_15d", "rain_30d", "rain_api"]:
        if c in new:
            new[c] += q.extra_rain_today_mm
    if "rain_max_3d" in new:
        new["rain_max_3d"] = max(new["rain_max_3d"], new["rain_mm"])
    if "rain_7d_anomaly" in new:
        new["rain_7d_anomaly"] = new["rain_7d"] - base["rain_7d_normal"]
    X = pd.DataFrame([base[b["features"]], new[b["features"]]]).astype(float)
    p0, p1 = b["model"].predict_proba(X)[:, 1]
    return {"district": q.district, "date": str(day.date()), "baseline_probability": round(float(p0), 4),
            "scenario_probability": round(float(p1), 4), "baseline_level": level(p0, b["threshold"]),
            "scenario_level": level(p1, b["threshold"]),
            "scenario": {"rain_change_pct": q.rain_change_pct, "extra_rain_today_mm": q.extra_rain_today_mm},
            "method": "The trained ML model is re-run with the changed rainfall features."}


@router.get("/model-report")
def model_report():
    out = {}
    for name in ["metrics_v2.json", "experiments_best.json"]:
        p = REPORTS / name
        if p.exists():
            out[name.replace(".json", "")] = json.loads(p.read_text())
    p = REPORTS / "experiments.csv"
    if p.exists():
        out["experiments"] = pd.read_csv(p).to_dict("records")
    if not out:
        raise HTTPException(404, "No v2 reports yet")
    return out

@router.get("/flood-dates")
def flood_dates(limit: int = 8):
    """Past dates with the most districts reporting a flood (for quick replay in the dashboard)."""
    df = data()
    if "flood_active" not in df.columns:
        return {"dates": []}
    f = df[df["flood_active"].astype(bool)]
    if f.empty:
        return {"dates": []}
    counts = f.groupby("date")["district_name"].agg(["count", lambda s: ", ".join(sorted(s.astype(str))[:6])])
    counts.columns = ["districts", "names"]
    counts = counts.sort_values("districts", ascending=False)
    picked = []
    for day, row in counts.iterrows():
        if all(abs((day - p).days) > 20 for p, _ in picked):   # one date per flood event
            picked.append((day, row))
        if len(picked) >= limit:
            break
    picked.sort(key=lambda x: x[0], reverse=True)
    return {"dates": [{"date": str(d.date()), "districts": int(r["districts"]), "names": r["names"]} for d, r in picked]}
