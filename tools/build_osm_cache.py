"""Saves OpenStreetMap emergency data (hospitals, shelters, roads, routes) for all 30 districts.
Run once while the backend is running; afterwards Emergency GIS loads instantly, even offline.
Usage:  python tools\\build_osm_cache.py
"""
import json, time
from urllib.parse import quote
from urllib.request import urlopen

API = "http://127.0.0.1:8001"
DISTRICTS = ["Anugul", "Balangir", "Baleshwar", "Bargarh", "Bauda", "Bhadrak", "Cuttack", "Debagarh", "Dhenkanal",
             "Gajapati", "Ganjam", "Jagatsinghapur", "Jajapur", "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara",
             "Kendujhar", "Khordha", "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangapur", "Nayagarh", "Nuapada",
             "Puri", "Rayagada", "Sambalpur", "Subarnapur", "Sundargarh"]
PARTS = [("infrastructure", "asset_count"), ("shelters", None), ("roads", None), ("city-risk", "city_count"), ("evacuation-route", "distance_km")]

for n, d in enumerate(DISTRICTS, 1):
    out = []
    for path, key in PARTS:
        try:
            with urlopen(f"{API}/api/{path}?district={quote(d)}", timeout=300) as r:
                j = json.loads(r.read().decode("utf-8"))
            out.append(f"{path}: {j.get(key) if key else j.get('status', 'ok')}")
        except Exception as exc:
            out.append(f"{path}: FAILED ({exc})")
        time.sleep(2)
    print(f"{n:2d}/30 {d:15s} " + " | ".join(out), flush=True)
print("Done. Saved in Datasets\\processed_data\\osm_cache")
