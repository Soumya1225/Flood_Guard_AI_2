"""Live 7-day flood forecast for every Odisha district (current dates).

Sources (free, no login):
  * Open-Meteo Flood API  -> GloFAS v4 river discharge (m3/s), past 7 days + next 7 days, ~5 km grid
  * Open-Meteo Forecast API -> daily rainfall forecast for the district centre

Method (simple and explainable):
  1. River point: for each district, 9 points around its centre are checked and the grid cell with the
     largest recent discharge (the main river) is used.
  2. River thresholds: that cell's own daily discharge history (Aug 2019 - Jul 2022, GloFAS) gives
     percentiles P80 / P95 / P99.
       discharge < P80 -> Low, P80-P95 -> Moderate, P95-P99 -> High, >= P99 -> Very High
  3. Rain level (IMD categories): < 64.5 mm Low, 64.5-115.5 heavy -> Moderate,
     115.6-204.4 very heavy -> High, >= 204.5 extremely heavy -> Very High
  4. Daily flood risk = the higher of the river level and the rain level; raised one step when the
     district's spatial risk score is >= 0.70 (low-lying, near rivers, many past floods).

River points and thresholds are computed once per district and cached in
Datasets/processed_data/glofas_district_reference.json. To build the cache for all districts in advance:
    python -m backend.live_forecast_api
"""
import json
import time
from datetime import date
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / "Datasets" / "processed_data" / "glofas_district_reference.json"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood?"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast?"
UA = {"User-Agent": "FloodGuardAI/2.0 (academic-project)"}
REF_START, REF_END = "2019-08-01", "2022-07-31"   # 3 monsoon seasons (kept short: the free API counts long ranges as many calls)
BUILD_PAUSE = 12                                    # seconds between districts when building references
OFFSETS = [(0, 0), (0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15), (0.15, 0.15), (0.15, -0.15), (-0.15, 0.15), (-0.15, -0.15)]
LEVELS = ["Low", "Moderate", "High", "Very High"]
LIVE_TTL = 3 * 3600          # live forecast is refreshed every 3 hours

router = APIRouter(prefix="/api/live", tags=["live-forecast"])
_lock = Lock()
_live_cache = {}             # district -> (time, result)
_build_state = {"running": False, "done": 0, "total": 0, "error": None}


# ------------------------------------------------------------------ helpers
class RateLimited(RuntimeError):
    pass


def _get_json(url, timeout=30, tries=2):
    last = None
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # network / timeout / rate limit
            last = exc
            if "429" in str(exc):
                raise RateLimited("Open-Meteo free limit reached (HTTP 429). Please try again in a few minutes.")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed: {last}")


def _main():
    """The main backend module (imported lazily to avoid a circular import)."""
    try:
        from backend import main as m
    except ImportError:
        import main as m
    return m


def _district_rows():
    """[(name, lat, lon, spatial_risk)] for all districts, from the backend's loaded data."""
    m = _main()
    gdf = m.district_geometry_gdf
    if gdf is None or gdf.empty:
        raise HTTPException(503, "District boundaries are not loaded in the backend.")
    risk = {}
    dd = m.district_data
    if dd is not None and not dd.empty and "NAME_2" in dd.columns:
        col = next((c for c in ["spatial_risk_score", "risk_score"] if c in dd.columns), None)
        if col:
            risk = {str(n).strip().lower(): float(v) for n, v in zip(dd["NAME_2"], dd[col]) if v == v}
    pts = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
    rows = []
    for name, p in zip(gdf["NAME_2"].astype(str), pts):
        rows.append((name.strip(), round(float(p.y), 4), round(float(p.x), 4), risk.get(name.strip().lower())))
    return rows


def _load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def level_from_discharge(q, th):
    if q is None or th is None:
        return None
    if q >= th["p99"]:
        return "Very High"
    if q >= th["p95"]:
        return "High"
    if q >= th["p80"]:
        return "Moderate"
    return "Low"


def level_from_rain(mm):
    if mm is None:
        return None
    if mm >= 204.5:
        return "Very High"
    if mm >= 115.6:
        return "High"
    if mm >= 64.5:
        return "Moderate"
    return "Low"


SMALL_STREAM_M3S = 50.0   # below this, a river cell is a small stream: at most "Moderate"
SMALL_RIVER_P95_M3S = 100.0   # river cell whose usual high flow (P95) is below this is a small stream too


def combine(river, rain, spatial_risk, small_stream=False):
    capped = False
    if small_stream and river in ("High", "Very High"):
        river, capped = "Moderate", True
    known = [x for x in (river, rain) if x]
    if not known:
        return None, "No live data"
    lvl = max(known, key=LEVELS.index)
    reason = []
    if river and river == lvl and river != "Low":
        reason.append("high river discharge" if not capped else "small stream above normal (capped at Moderate)")
    if rain and rain == lvl and rain != "Low":
        reason.append("heavy rain")
    river_only_small = small_stream and rain in (None, "Low")
    if (spatial_risk is not None and spatial_risk >= 0.70 and lvl in ("Moderate", "High")
            and not river_only_small):
        lvl = LEVELS[LEVELS.index(lvl) + 1]
        reason.append("high-risk district (spatial risk ≥ 0.70)")
    return lvl, (", ".join(reason) if reason else "normal river and rain")


# ------------------------------------------------------------------ reference (once per district)
def build_reference(name, lat, lon):
    """Choose the main-river grid cell near the district centre and its discharge thresholds."""
    lats = ",".join(str(round(lat + dy, 4)) for dy, dx in OFFSETS)
    lons = ",".join(str(round(lon + dx, 4)) for dy, dx in OFFSETS)
    # step 1 (cheap): last 14 days at 9 points -> the cell with the most water is the main river
    q = urlencode({"latitude": lats, "longitude": lons, "daily": "river_discharge",
                   "past_days": 14, "forecast_days": 1})
    data = _get_json(FLOOD_URL + q)
    items = data if isinstance(data, list) else [data]
    best = None
    for it in items:
        vals = [v for v in (it.get("daily", {}).get("river_discharge") or []) if v is not None]
        if not vals:
            continue
        mean = float(np.mean(vals))
        if best is None or mean > best[0]:
            best = (mean, it)
    if best is None:
        raise RuntimeError(f"No river discharge found near {name}")
    cell = best[1]
    # step 2: 3 years of daily history for that one cell -> percentiles
    hq = urlencode({"latitude": cell.get("latitude"), "longitude": cell.get("longitude"),
                    "daily": "river_discharge", "start_date": REF_START, "end_date": REF_END})
    hist = _get_json(FLOOD_URL + hq, timeout=60).get("daily", {})
    vals = np.array([v for v in (hist.get("river_discharge") or []) if v is not None], dtype=float)
    if vals.size < 365:
        raise RuntimeError(f"Not enough river history near {name}")
    return {
        "lat": cell.get("latitude"), "lon": cell.get("longitude"),
        "mean_discharge": round(float(vals.mean()), 2),
        "thresholds": {k: round(float(np.percentile(vals, p)), 2) for k, p in (("p80", 80), ("p95", 95), ("p99", 99))},
        "reference_period": f"{REF_START} to {REF_END}",
        "days_used": int(vals.size),
    }


def get_reference(name, lat, lon):
    cache = _load_cache()
    key = name.lower()
    if key in cache:
        return cache[key]
    ref = build_reference(name, lat, lon)
    with _lock:
        cache = _load_cache()
        cache[key] = ref
        _save_cache(cache)
    return ref


# ------------------------------------------------------------------ live forecast
def live_for(name, lat, lon, spatial_risk):
    key = name.lower()
    hit = _live_cache.get(key)
    if hit and time.time() - hit[0] < LIVE_TTL:
        return hit[1]

    ref = get_reference(name, lat, lon)
    th = ref["thresholds"]
    fq = urlencode({"latitude": ref["lat"], "longitude": ref["lon"],
                    "daily": "river_discharge,river_discharge_max", "past_days": 7, "forecast_days": 8})
    flood = _get_json(FLOOD_URL + fq).get("daily", {})
    wq = urlencode({"latitude": lat, "longitude": lon, "daily": "precipitation_sum",
                    "past_days": 7, "forecast_days": 8, "timezone": "Asia/Kolkata"})
    try:
        rain = _get_json(WEATHER_URL + wq).get("daily", {})
    except Exception:
        rain = {}
    rain_by_day = dict(zip(rain.get("time", []), rain.get("precipitation_sum", [])))

    today = date.today().isoformat()
    days = []
    times = flood.get("time", [])
    qs = flood.get("river_discharge", [])
    qmax = flood.get("river_discharge_max", [])
    for i, d in enumerate(times):
        q = qs[i] if i < len(qs) else None
        qm = qmax[i] if i < len(qmax) else None
        mm = rain_by_day.get(d)
        rl, wl = level_from_discharge(q, th), level_from_rain(mm)
        small = (q is not None and q < SMALL_STREAM_M3S) or float(th.get("p95") or 0) < SMALL_RIVER_P95_M3S
        fl, why = combine(rl, wl, spatial_risk, small_stream=small)
        if small and rl in ("High", "Very High"):
            rl = "Moderate"
        days.append({
            "date": d, "is_past": d < today, "is_today": d == today,
            "discharge_m3s": None if q is None else round(q, 1),
            "discharge_max_m3s": None if qm is None else round(qm, 1),
            "rain_mm": None if mm is None else round(mm, 1),
            "river_level": rl, "rain_level": wl, "risk_level": fl, "reason": why,
        })
    future = [d for d in days if not d["is_past"] and d["risk_level"]]
    today_row = next((d for d in days if d["is_today"]), future[0] if future else None)
    peak = max(future, key=lambda d: (LEVELS.index(d["risk_level"]), d["discharge_m3s"] or 0)) if future else None
    result = {
        "district": name, "today": today,
        "spatial_risk_score": spatial_risk,
        "river_point": {"lat": ref["lat"], "lon": ref["lon"], "mean_discharge_m3s": ref["mean_discharge"]},
        "thresholds_m3s": th, "reference_period": ref["reference_period"],
        "today_level": today_row["risk_level"] if today_row else None,
        "today_reason": today_row["reason"] if today_row else None,
        "peak": peak,
        "days": days,
        "source": "Open-Meteo Flood API (GloFAS v4 river discharge) + Open-Meteo rainfall forecast",
        "method": ("River level from the district's own discharge percentiles (P80/P95/P99, 2010-2022); "
                   "rain level from IMD heavy-rain categories; daily risk = the higher of the two, "
                   "raised one step for districts with spatial risk ≥ 0.70. Small streams (< 50 m³/s) "
                   "are capped at Moderate."),
    }
    _live_cache[key] = (time.time(), result)
    return result


def _find(name):
    for row in _district_rows():
        if row[0].lower() == str(name).strip().lower():
            return row
    raise HTTPException(404, f"District not found: {name}")


@router.get("/forecast/{district_name}")
def live_forecast(district_name: str):
    name, lat, lon, risk = _find(district_name)
    try:
        return {"status": "success", **live_for(name, lat, lon, risk)}
    except HTTPException:
        raise
    except RateLimited as exc:
        raise HTTPException(429, str(exc))
    except Exception as exc:
        raise HTTPException(503, f"Live flood data is not reachable right now: {exc}")


def _build_missing(rows):
    """Background: build river references one district at a time (gentle on the free API)."""
    if _build_state["running"]:
        return
    _build_state.update(running=True, error=None)
    try:
        cache = _load_cache()
        todo = [r for r in rows if r[0].lower() not in cache]
        _build_state.update(total=len(rows), done=len(rows) - len(todo))
        for n, la, lo, _ in todo:
            for attempt in range(3):
                try:
                    get_reference(n, la, lo)
                    _build_state["done"] += 1
                    break
                except RateLimited as exc:
                    _build_state["error"] = str(exc)
                    time.sleep(65)
                except Exception as exc:
                    _build_state["error"] = f"{n}: {exc}"
                    break
            time.sleep(BUILD_PAUSE)
    finally:
        _build_state["running"] = False


@router.get("/summary")
def live_summary():
    """Today's live risk and 7-day peak for every district whose river reference is ready.
    Missing references are prepared in the background, one district at a time."""
    rows = _district_rows()
    cache = _load_cache()
    ready = [r for r in rows if r[0].lower() in cache]
    missing = [r[0] for r in rows if r[0].lower() not in cache]
    if missing and not _build_state["running"]:
        Thread(target=_build_missing, args=(rows,), daemon=True).start()

    out, errors = [], []
    for r in ready:
        try:
            res = live_for(*r)
            pk = res["peak"] or {}
            out.append({"district": r[0], "today_level": res["today_level"], "today_reason": res["today_reason"],
                        "peak_level": pk.get("risk_level"), "peak_date": pk.get("date"),
                        "peak_discharge_m3s": pk.get("discharge_m3s"), "p95_m3s": res["thresholds_m3s"]["p95"],
                        "max_rain_mm": max([d["rain_mm"] or 0 for d in res["days"] if not d["is_past"]] or [0]),
                        "spatial_risk_score": r[3]})
        except RateLimited as exc:
            errors.append(f"{r[0]}: {exc}")
            break
        except Exception as exc:
            errors.append(f"{r[0]}: {exc}")
    order = {lv: i for i, lv in enumerate(LEVELS)}
    out.sort(key=lambda x: (-order.get(x["peak_level"] or "Low", 0), -(x["peak_discharge_m3s"] or 0)))
    return {"status": "success", "today": date.today().isoformat(), "districts": out,
            "preparing": missing, "ready_count": len(ready), "total": len(rows),
            "build_status": dict(_build_state), "unavailable": errors, "levels": LEVELS}


if __name__ == "__main__":
    # Build the river reference cache for all districts (run once, ~1-3 minutes).
    import sys
    sys.path.insert(0, str(ROOT))
    for n, la, lo, rk in _district_rows():
        try:
            ref = get_reference(n, la, lo)
            print(f"{n:16s} river cell {ref['lat']},{ref['lon']}  mean {ref['mean_discharge']} m3/s  P95 {ref['thresholds']['p95']}")
            time.sleep(BUILD_PAUSE)
        except Exception as exc:
            print(f"{n:16s} FAILED: {exc}")
    print("Saved:", CACHE_FILE)
