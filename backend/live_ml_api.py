"""Live ML flood prediction: the trained v2 model (models/flood_onset_model_v2.pkl) run on TODAY's data.

The v2 model was trained on IMD rainfall + NASA POWER weather (2000-2016) and tested on 2017-2023.
Its inputs are rainfall and weather features only, so it can be fed with live data:
  * Open-Meteo (free): last 60 days of rainfall/weather + 8-day forecast, for each district centre
  * the same feature formulas as pipeline_v2 (rain totals over 3/7/15/30 days, heaviest day,
    soil-wetness index, rain vs the district's normal for this time of year, season, weather)
  * district constants from the training data (past flood frequency, elevation, distance to river)
Question answered (same as training): "will a flood start in this district within the next 7 days?"

Endpoints:
  GET /api/live-ml/warnings          today's probability + 7-day peak for all districts
  GET /api/live-ml/district/{name}   last 30 days + next 7 days of probability and rain for one district
"""
import time
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from urllib.parse import urlencode

router = APIRouter(prefix="/api/live-ml", tags=["live-ml-prediction"])

WEATHER_URL = "https://api.open-meteo.com/v1/forecast?"
PAST_DAYS, FORECAST_DAYS = 60, 8
TTL = 3 * 3600
WIND_10M_TO_2M = 0.748          # NASA POWER (training) uses 2 m wind, Open-Meteo gives 10 m wind
_cache = {"t": 0, "result": None}


def _v2():
    try:
        from backend import flood_v2_api as v2
    except ImportError:
        import flood_v2_api as v2
    return v2


def _lf():
    try:
        from backend import live_forecast_api as lf
    except ImportError:
        import live_forecast_api as lf
    return lf


def _district_constants():
    """Per district: last known constant features + normal 7-day rain for every day of the year."""
    v2 = _v2()
    df = v2.data()
    b = v2.bundle()
    feats = b["features"]
    const_cols = [f for f in feats if f.startswith("static_") or f == "district_flood_freq"]
    df = df.copy()
    df["key"] = df["district_name"].astype(str).str.strip().str.lower()
    df["doy"] = df["date"].dt.dayofyear
    last = df.sort_values("date").groupby("key").tail(1).set_index("key")
    consts = {k: {c: float(last.loc[k, c]) for c in const_cols if c in last.columns} for k in last.index}
    normals = {}
    if "rain_7d_normal" in df.columns:
        nm = df.groupby(["key", "doy"])["rain_7d_normal"].first()
        normals = {k: v for k, v in nm.items()}
    return consts, normals


def _fetch_weather(rows):
    lf = _lf()
    q = urlencode({
        "latitude": ",".join(str(r[1]) for r in rows),
        "longitude": ",".join(str(r[2]) for r in rows),
        "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean",
        "wind_speed_unit": "ms", "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
        "timezone": "Asia/Kolkata",
    })
    data = lf._get_json(WEATHER_URL + q, timeout=60)
    return data if isinstance(data, list) else [data]


def _features_for(name, daily, consts, normals, medians):
    """Feature rows (same formulas as pipeline_v2.s02_build_dataset) for every day in the series."""
    dates = pd.to_datetime(daily.get("time", []))
    rain = np.array([v if v is not None else 0.0 for v in daily.get("precipitation_sum", [])], dtype=float)
    hum = pd.Series(daily.get("relative_humidity_2m_mean", []), dtype=float)
    temp = pd.Series(daily.get("temperature_2m_mean", []), dtype=float)
    wind = pd.Series(daily.get("wind_speed_10m_mean", []), dtype=float) * WIND_10M_TO_2M
    s = pd.Series(rain)
    api, acc = [], 0.0
    for v in rain:
        acc = acc * 0.9 + v
        api.append(acc)
    key = name.strip().lower()
    rows = pd.DataFrame({
        "date": dates, "rain_mm": rain,
        "rain_3d": s.rolling(3, min_periods=1).sum(), "rain_7d": s.rolling(7, min_periods=1).sum(),
        "rain_15d": s.rolling(15, min_periods=1).sum(), "rain_30d": s.rolling(30, min_periods=1).sum(),
        "rain_max_3d": s.rolling(3, min_periods=1).max(), "rain_api": api,
        "humidity": hum, "humidity_3d": hum.rolling(3, min_periods=1).mean(), "temp_c": temp, "wind_ms": wind,
    })
    doy = rows["date"].dt.dayofyear
    rows["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    rows["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    rows["monsoon"] = rows["date"].dt.month.between(6, 10).astype(int)
    rows["rain_7d_normal"] = [normals.get((key, int(d)), np.nan) for d in doy]
    rows["rain_7d_anomaly"] = rows["rain_7d"] - rows["rain_7d_normal"]
    for c, v in consts.get(key, {}).items():
        rows[c] = v
    return rows


def _run():
    if _cache["result"] and time.time() - _cache["t"] < TTL:
        return _cache["result"]
    v2, lf = _v2(), _lf()
    b = v2.bundle()
    feats, medians = b["features"], pd.Series(b["medians"])
    rows = lf._district_rows()
    consts, normals = _district_constants()
    weather = _fetch_weather(rows)
    today = pd.Timestamp(date.today())
    out = {}
    for (name, lat, lon, risk), item in zip(rows, weather):
        f = _features_for(name, item.get("daily", {}), consts, normals, medians)
        for c in feats:
            if c not in f.columns:
                f[c] = np.nan
        X = f[feats].astype(float).fillna(medians)
        f["probability"] = b["model"].predict_proba(X)[:, 1]
        f["level"] = [v2.level(p, b["threshold"]) for p in f["probability"]]
        f["is_today"] = f["date"] == today
        f["is_past"] = f["date"] < today
        future = f[~f["is_past"]]
        today_row = f[f["is_today"]]
        t = today_row.iloc[0] if not today_row.empty else (future.iloc[0] if not future.empty else f.iloc[-1])
        peak = future.loc[future["probability"].idxmax()] if not future.empty else t
        X_t = X.loc[t.name].copy()
        out[name] = {
            "district": name, "spatial_risk_score": risk,
            "today_probability": round(float(t["probability"]), 4), "today_level": t["level"],
            "peak_probability": round(float(peak["probability"]), 4), "peak_level": peak["level"],
            "peak_date": str(peak["date"].date()),
            "rain_next7_mm": round(float(future["rain_mm"].sum()), 1),
            "rain_last7_mm": round(float(f[f["is_past"]]["rain_mm"].tail(7).sum()), 1),
            "drivers": v2.drivers(X_t, b),
            "series": [{"date": str(r.date.date()), "rain_mm": round(float(r.rain_mm), 1),
                        "probability": round(float(r.probability), 4), "level": r.level,
                        "is_today": bool(r.is_today), "is_past": bool(r.is_past)}
                       for r in f.tail(30 + FORECAST_DAYS).itertuples()],
        }
    result = {
        "today": str(today.date()), "model": b.get("name", "v2 model"),
        "question": "Will a flood start in this district within the next 7 days?",
        "threshold": round(float(b["threshold"]), 4), "levels": b.get("levels"),
        "districts": out,
        "source": "Open-Meteo live rainfall/weather (last 60 days + 8-day forecast) fed into the trained v2 model",
        "note": ("The model was trained on IMD district-average rainfall; live values are forecasts for the "
                 "district centre, so probabilities are an early-warning screening, not an official warning."),
    }
    _cache.update(t=time.time(), result=result)
    return result


def get_live_ml(name=None):
    """Used by other modules (response plan, area alert)."""
    res = _run()
    if name is None:
        return res
    for k, v in res["districts"].items():
        if k.lower() == str(name).strip().lower():
            return v
    return None


@router.get("/warnings")
def live_ml_warnings():
    try:
        res = _run()
    except HTTPException:
        raise
    except Exception as exc:
        code = 429 if "429" in str(exc) or "limit" in str(exc).lower() else 503
        raise HTTPException(code, f"Live ML prediction unavailable: {exc}")
    rows = [{k: v for k, v in d.items() if k != "series"} for d in res["districts"].values()]
    rows.sort(key=lambda r: -r["peak_probability"])
    return {**{k: v for k, v in res.items() if k != "districts"}, "districts": rows}


@router.get("/district/{district_name}")
def live_ml_district(district_name: str):
    try:
        d = get_live_ml(district_name)
    except Exception as exc:
        raise HTTPException(503, f"Live ML prediction unavailable: {exc}")
    if d is None:
        raise HTTPException(404, f"District not found: {district_name}")
    res = _cache["result"]
    return {"today": res["today"], "model": res["model"], "threshold": res["threshold"],
            "question": res["question"], "source": res["source"], "note": res["note"], **d}
