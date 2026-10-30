"""Shared paths, settings and Odisha district-name matching."""
import difflib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets" / "data"
V2 = DATA / "v2"
IMD_DIR = V2 / "imd"
POWER_DIR = V2 / "nasa_power"
IFI_CSV = V2 / "India_Flood_Inventory_v3.csv"
DATASET = ROOT / "Datasets" / "processed_data" / "district_daily_v2.csv.gz"
MODELS = ROOT / "models"
REPORTS = ROOT / "models" / "v2_reports"

START_YEAR, END_YEAR = 2000, 2023     # IMD rainfall + flood inventory overlap
TRAIN_END = 2016                       # train 2000-2016, test 2017-2023 (time split, no leakage)
VALID_FROM = 2013                      # inside training: 2013-2016 used to choose the alert threshold
HORIZON = 3                            # predict: will a flood start in this district within the next 3 days?
BBOX = (81.0, 17.5, 87.6, 22.7)        # Odisha lon/lat box

# common spellings in flood reports -> GADM 4.1 district names
ALIASES = {
    "angul": "anugul", "bolangir": "balangir", "balasore": "baleshwar", "baleswar": "baleshwar",
    "balasor": "baleshwar", "boudh": "baudh", "bauda": "baudh", "deogarh": "debagarh", "deoghar": "debagarh",
    "jajpur": "jajapur", "jagatsingpur": "jagatsinghapur", "jagatsinghpur": "jagatsinghapur",
    "keonjhar": "kendujhar", "khurda": "khordha", "khorda": "khordha", "nawarangpur": "nabarangapur",
    "nabarangpur": "nabarangapur", "sonepur": "subarnapur", "sonapur": "subarnapur", "malkanagiri": "malkangiri",
    "phulbani": "kandhamal", "sundergarh": "sundargarh", "nawapara": "nuapada", "baragarh": "bargarh",
    "kendrapada": "kendrapara", "mayurbhanja": "mayurbhanj", "jharsugda": "jharsuguda", "dhenkenal": "dhenkanal",
}


def norm(name):
    s = re.sub(r"[^a-z]", "", str(name).lower().replace("district", ""))
    return ALIASES.get(s, s)


def match_district(name, known):
    """Map a district name from any source to one of the known (normalised) GADM names, or None."""
    k = norm(name)
    if k in known:
        return k
    m = difflib.get_close_matches(k, list(known), n=1, cutoff=0.82)
    return m[0] if m else None


def split_districts(text):
    """'Cuttack, Puri and Khordha' -> ['Cuttack', 'Puri', 'Khordha']"""
    parts = re.split(r",|;|/|\band\b|&", str(text))
    return [p.strip() for p in parts if p.strip() and p.strip().lower() not in ("nan", "none")]


def load_districts():
    """Odisha district polygons from the GADM file already in the project (data/boundaries/gadm41_IND.gpkg)."""
    import geopandas as gpd
    path = DATA / "boundaries" / "gadm41_IND.gpkg"
    layer = "ADM_ADM_2"
    try:
        import pyogrio
        layers = [l[0] for l in pyogrio.list_layers(path)]
        layer = next((l for l in layers if l.endswith("_2")), layer)
    except Exception:
        pass
    g = gpd.read_file(path, layer=layer)
    g = g[g["NAME_1"].str.lower().isin(["odisha", "orissa"])].copy()
    g["district"] = g["NAME_2"].map(norm)
    g["district_name"] = g["NAME_2"]
    return g[["district", "district_name", "geometry"]].to_crs(4326).reset_index(drop=True)
