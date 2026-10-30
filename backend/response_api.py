"""Flood response plan for one district: what to do, for how many people, who to call, what to tell people.

Combines (all already in Flood Guard AI):
  * live 7-day river + rain outlook (live_forecast_api: GloFAS discharge + Open-Meteo rain)
  * live ML flood-start probability (live_ml_api: trained v2 model on live rainfall)
  * long-term spatial risk score and WorldPop 2020 population (existing datasets)
  * Census 2011 towns of the district

Endpoints:
  GET  /api/response/{district}                      response tier, people, relief needs, priorities, contacts,
                                                     warning messages (English / Hindi / Odia), checklist
  POST /api/response/{district}/task/{task_id}?done=true|false   tick a checklist item (saved in SQLite)
"""
import math
import os
from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/response", tags=["response-plan"])

TIERS = [
    {"tier": 0, "name": "Normal", "action": "Monitor", "color": "#16a34a"},
    {"tier": 1, "name": "Watch", "action": "Prepare", "color": "#ca8a04"},
    {"tier": 2, "name": "Warning", "action": "Act", "color": "#ea580c"},
    {"tier": 3, "name": "Severe", "action": "Evacuate", "color": "#dc2626"},
]
RIVER_TIER = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
ML_TIER = {"Normal": 0, "Watch": 1, "Warning": 2, "Severe": 3}
# share of the population-at-risk to shelter at each tier (planning assumption)
EVACUATE_SHARE = {0: 0.0, 1: 0.01, 2: 0.05, 3: 0.15}

# NDRF 3rd Battalion headquarters, Mundali (Cuttack district)
NDRF_BASE = {"name": "NDRF 3rd Battalion, Mundali (Cuttack)", "lat": 20.44, "lon": 85.76}

CONTACTS = [
    {"name": "Emergency (police / fire / ambulance)", "number": "112"},
    {"name": "State Emergency Operation Centre, Odisha", "number": "1070"},
    {"name": "District Emergency Operation Centre (control room)", "number": "1077"},
    {"name": "Ambulance", "number": "108"},
]

TASKS = [
    (0, "monitor", "Check the live river and rain outlook twice a day"),
    (0, "contacts", "Verify control-room (1077) staff roster and phone list"),
    (1, "inform", "Alert block officers, sarpanches and Gram Panchayat volunteers"),
    (1, "shelters", "Identify and open-check cyclone/flood shelters and schools for use as relief camps"),
    (1, "stock", "Check stock of dry rations, drinking water, medicines and tarpaulins"),
    (1, "boats", "Keep boats, life jackets and Fire Service / ODRAF teams on standby"),
    (2, "warn", "Send the public warning message (Odia / Hindi / English) through SMS, radio and loudspeakers"),
    (2, "evacuate_vulnerable", "Move elderly, pregnant women, children and disabled people from low-lying areas"),
    (2, "camps", "Open relief camps with water, food, toilets and a medical team"),
    (2, "ndrf", "Request NDRF / ODRAF deployment through the State EOC (1070)"),
    (2, "roads", "Close flooded roads and causeways; plan alternate routes"),
    (3, "evacuate_all", "Evacuate all people from river banks and low-lying villages"),
    (3, "rescue", "Deploy boats and rescue teams to the highest-priority areas"),
    (3, "health", "Start health camps, chlorinate drinking water, prevent disease outbreaks"),
    (3, "report", "Send a situation report to the State EOC every 6 hours"),
]

LEVEL_WORDS = {
    "en": {0: "LOW", 1: "MODERATE", 2: "HIGH", 3: "VERY HIGH"},
    "hi": {0: "कम", 1: "मध्यम", 2: "अधिक", 3: "बहुत अधिक"},
    "or": {0: "କମ୍", 1: "ମଧ୍ୟମ", 2: "ଅଧିକ", 3: "ଅତ୍ୟଧିକ"},
}


def _m():
    try:
        from backend import main as m
    except ImportError:
        import main as m
    return m


def _mod(name):
    try:
        return __import__(f"backend.{name}", fromlist=[name])
    except ImportError:
        return __import__(name)


def _km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def _population(name):
    m = _m()
    path = getattr(m, "POPULATION_DISTRICT_FILE", None)
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["district"].astype(str).str.strip().str.lower() == name.lower()]
    if row.empty:
        return None
    return float(row.iloc[0]["population_worldpop_2020"])


def _towns(name, limit=10):
    m = _m()
    path = getattr(m, "CITY_FILE", None)
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    d = df[df["district"].astype(str).str.strip().str.lower() == name.lower()].copy()
    if d.empty:
        return []
    d["population_2011"] = pd.to_numeric(d["population_2011"], errors="coerce").fillna(0)
    d = d.sort_values("population_2011", ascending=False).head(limit)
    return [{"name": str(r.city_name), "population_2011": int(r.population_2011)} for r in d.itertuples()]


def _messages(district, tier, peak_date, reasons):
    when = peak_date or "the coming days"
    if tier == 0:
        return {
            "en": (f"FLOOD UPDATE – {district}: No flood is expected in the next 7 days. "
                   "Keep following weather updates and instructions from the district administration. "
                   "Emergency: call 112 or 1070."),
            "hi": (f"बाढ़ सूचना – {district}: अगले 7 दिनों में बाढ़ की आशंका नहीं है। "
                   "मौसम की जानकारी और ज़िला प्रशासन के निर्देशों पर ध्यान देते रहें। आपात स्थिति में 112 या 1070 पर कॉल करें।"),
            "or": (f"ବନ୍ୟା ସୂଚନା – {district}: ଆଗାମୀ 7 ଦିନରେ ବନ୍ୟାର ଆଶଙ୍କା ନାହିଁ। "
                   "ପାଣିପାଗ ସୂଚନା ଓ ଜିଲ୍ଲା ପ୍ରଶାସନର ନିର୍ଦ୍ଦେଶ ଉପରେ ନଜର ରଖନ୍ତୁ। ଜରୁରୀକାଳୀନ ସହାୟତା ପାଇଁ 112 କିମ୍ବା 1070 କୁ ଫୋନ କରନ୍ତୁ।"),
        }
    en_w, hi_w, or_w = LEVEL_WORDS["en"][tier], LEVEL_WORDS["hi"][tier], LEVEL_WORDS["or"][tier]
    msgs = {
        "en": (f"FLOOD ALERT – {district}: Flood risk is {en_w} (highest around {when}). "
               f"{'Reason: ' + reasons + '. ' if reasons else ''}"
               "Stay away from river banks and low-lying areas, keep drinking water, food, medicines and documents ready, "
               "and follow instructions from the district administration. Emergency: call 112 or 1070."),
        "hi": (f"बाढ़ चेतावनी – {district}: बाढ़ का खतरा {hi_w} है ({when} के आसपास सबसे अधिक)। "
               "नदी किनारे और निचले इलाकों से दूर रहें, पीने का पानी, खाना, दवाइयाँ और ज़रूरी कागज़ात तैयार रखें, "
               "और ज़िला प्रशासन के निर्देशों का पालन करें। आपात स्थिति में 112 या 1070 पर कॉल करें।"),
        "or": (f"ବନ୍ୟା ସତର୍କତା – {district}: ବନ୍ୟା ବିପଦ {or_w} ({when} ନିକଟରେ ସର୍ବାଧିକ)। "
               "ନଦୀ କୂଳ ଓ ନିମ୍ନାଞ୍ଚଳରୁ ଦୂରେଇ ରୁହନ୍ତୁ, ପିଇବା ପାଣି, ଖାଦ୍ୟ, ଔଷଧ ଓ ଜରୁରୀ କାଗଜପତ୍ର ପ୍ରସ୍ତୁତ ରଖନ୍ତୁ, "
               "ଏବଂ ଜିଲ୍ଲା ପ୍ରଶାସନର ନିର୍ଦ୍ଦେଶ ମାନନ୍ତୁ। ଜରୁରୀକାଳୀନ ସହାୟତା ପାଇଁ 112 କିମ୍ବା 1070 କୁ ଫୋନ କରନ୍ତୁ।"),
    }
    return msgs


def _ensure_table():
    with _m().get_db_connection() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS response_tasks (
            district TEXT, task_id TEXT, done INTEGER, updated_at TEXT,
            PRIMARY KEY (district, task_id))""")


def _task_status(district):
    _ensure_table()
    with _m().get_db_connection() as con:
        rows = con.execute("SELECT task_id, done, updated_at FROM response_tasks WHERE district = ?",
                           (district.lower(),)).fetchall()
    return {r["task_id"]: {"done": bool(r["done"]), "updated_at": r["updated_at"]} for r in rows}


def build_plan(district_name):
    lf, ml = _mod("live_forecast_api"), _mod("live_ml_api")
    name, lat, lon, spatial = lf._find(district_name)

    river, ml_row, notes = None, None, []
    try:
        river = lf.live_for(name, lat, lon, spatial)
    except Exception as exc:
        notes.append(f"Live river/rain outlook unavailable: {exc}")
    try:
        ml_row = ml.get_live_ml(name)
    except Exception as exc:
        notes.append(f"Live ML prediction unavailable: {exc}")

    river_peak = (river or {}).get("peak") or {}
    river_t = RIVER_TIER.get(river_peak.get("risk_level"), 0) if river else None
    ml_t = ML_TIER.get((ml_row or {}).get("peak_level"), 0) if ml_row else None
    known = [t for t in (river_t, ml_t) if t is not None]
    if known:
        tier = max(known)
    else:  # no live data: fall back to the long-term spatial risk
        tier = 2 if (spatial or 0) >= 0.7 else 1 if (spatial or 0) >= 0.5 else 0
        notes.append("Tier based on long-term spatial risk only (no live data).")

    reasons = []
    if river_t:
        reasons.append(f"river/rain outlook {river_peak.get('risk_level')} ({river_peak.get('reason')})")
    if ml_t:
        reasons.append(f"ML flood probability {ml_row['peak_probability'] * 100:.0f}% ({ml_row['peak_level']})")
    peak_date = river_peak.get("date") if river_t and river_t >= (ml_t or 0) else (ml_row or {}).get("peak_date")

    population = _population(name)
    at_risk = population * float(spatial or 0) if population else None
    to_shelter = int(round(at_risk * EVACUATE_SHARE[tier])) if at_risk else 0
    days = 3
    relief = {
        "people_to_shelter": to_shelter,
        "relief_camps": math.ceil(to_shelter / 500) if to_shelter else 0,
        "drinking_water_litres_per_day": to_shelter * 15,
        "dry_rations_kg_per_day": round(to_shelter * 0.5),
        "shelter_space_m2": round(to_shelter * 3.5),
        "toilets": math.ceil(to_shelter / 20) if to_shelter else 0,
        "rescue_boats": math.ceil(to_shelter / 1000) if to_shelter else 0,
        "planning_days": days,
        "basis": ("People to shelter = WorldPop 2020 population × spatial risk × tier share "
                  "(Watch 1%, Warning 5%, Severe 15%). Water 15 L, covered space 3.5 m² and 1 toilet per 20 people "
                  "follow Sphere minimum standards; 0.5 kg dry ration per person per day, 1 camp per 500 people and "
                  "1 rescue boat per 1,000 people are planning assumptions."),
    }

    dist_ndrf = _km(lat, lon, NDRF_BASE["lat"], NDRF_BASE["lon"])
    resources = {
        "ndrf": {**NDRF_BASE, "distance_km": round(dist_ndrf), "travel_hours_by_road": round(dist_ndrf / 40, 1)},
        "district_teams": ("District Emergency Operation Centre, Odisha Fire & Emergency Service and ODRAF "
                           "(request through the State EOC, 1070). Nearest hospitals, fire stations and shelters "
                           "are listed in Emergency GIS Intelligence below."),
        "contacts": CONTACTS,
    }

    status = _task_status(name)
    checklist = [{"id": tid, "tier": t, "text": txt, "required": t <= tier,
                  "done": status.get(tid, {}).get("done", False),
                  "updated_at": status.get(tid, {}).get("updated_at")} for t, tid, txt in TASKS]

    return {
        "status": "success", "district": name, "today": date.today().isoformat(),
        "tier": TIERS[tier], "reasons": reasons, "peak_date": peak_date, "notes": notes,
        "live_outlook": {
            "river_peak_level": river_peak.get("risk_level"), "river_peak_date": river_peak.get("date"),
            "ml_peak_probability": (ml_row or {}).get("peak_probability"),
            "ml_peak_level": (ml_row or {}).get("peak_level"), "ml_peak_date": (ml_row or {}).get("peak_date"),
        },
        "spatial_risk_score": spatial,
        "population_worldpop_2020": round(population) if population else None,
        "population_at_risk": round(at_risk) if at_risk else None,
        "relief": relief,
        "evacuation_priority": _towns(name),
        "resources": resources,
        "messages": _messages(name, tier, peak_date, "; ".join(reasons)),
        "checklist": checklist,
        "note": "Decision-support plan for academic use; final decisions rest with the district administration.",
    }


@router.get("/{district_name}")
def response_plan(district_name: str):
    try:
        return build_plan(district_name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Response plan failed: {exc}")


@router.post("/{district_name}/task/{task_id}")
def set_task(district_name: str, task_id: str, done: bool = True):
    if task_id not in {t[1] for t in TASKS}:
        raise HTTPException(404, "Unknown task")
    _ensure_table()
    with _m().get_db_connection() as con:
        con.execute("INSERT OR REPLACE INTO response_tasks (district, task_id, done, updated_at) VALUES (?, ?, ?, ?)",
                    (district_name.strip().lower(), task_id, int(bool(done)), datetime.now().isoformat(timespec="seconds")))
    return {"status": "success", "district": district_name, "task_id": task_id, "done": bool(done)}
