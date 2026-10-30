from pathlib import Path
import pandas as pd
import geopandas as gpd


# ============================================================
# FLOOD GUARD AI
# STEP 2A - DATA PREPROCESSING
# ============================================================

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "Datasets" / "data"
OUTPUT = BASE / "Datasets" / "processed_data"

OUTPUT.mkdir(parents=True, exist_ok=True)


def save_csv(df, filename):
    path = OUTPUT / filename
    df.to_csv(path, index=False)
    print(f"   ✅ Saved: {path}")
    print(f"      Shape: {df.shape}")


print("=" * 70)
print("        FLOOD GUARD AI - DATA PREPROCESSING")
print("=" * 70)


# ============================================================
# 1. RAINFALL
# ============================================================

print("\n[1] Processing rainfall...")

rainfall_path = (
    DATA / "rainfall" / "rainfall_districtwise_daily_imd.csv"
)

rainfall = pd.read_csv(rainfall_path)

rainfall["Date"] = pd.to_datetime(
    rainfall["Date"],
    errors="coerce",
    dayfirst=True
)

rainfall = rainfall[
    rainfall["State"]
    .astype(str)
    .str.upper()
    .str.contains("ODISHA", na=False)
].copy()

rainfall = rainfall.dropna(subset=["Date"])

rainfall.columns = [
    str(c).strip().lower().replace(" ", "_")
    for c in rainfall.columns
]

save_csv(rainfall, "rainfall_odisha_processed.csv")


# ============================================================
# 2. WATER LEVEL
# ============================================================

print("\n[2] Processing water level...")

water_path = (
    DATA / "water_level" / "rwl_tele_hr_cwc_004_1991_2020.csv"
)

water = pd.read_csv(water_path)

water["data_acquisition_time"] = pd.to_datetime(
    water["Data Acquisition Time"],
    errors="coerce"
)

water = water[
    water["State"]
    .astype(str)
    .str.upper()
    .str.contains("ODISHA", na=False)
].copy()

water["water_level_m"] = pd.to_numeric(
    water["River Water Level Telemetry Hourly (meter)"],
    errors="coerce"
)

water = water.dropna(
    subset=["data_acquisition_time", "water_level_m"]
)

water.columns = [
    str(c).strip().lower().replace(" ", "_")
    for c in water.columns
]

save_csv(water, "water_level_odisha_processed.csv")


# ============================================================
# 3. WEATHER
# ============================================================

print("\n[3] Processing weather...")

weather_path = DATA / "weather" / "weather_odisha.csv.csv"

weather = pd.read_csv(
    weather_path,
    skiprows=13
)

weather = weather.replace(-999, pd.NA)

weather["date"] = pd.to_datetime(
    weather["YEAR"].astype(str)
    + weather["DOY"].astype(str),
    format="%Y%j",
    errors="coerce"
)

weather = weather.dropna(subset=["date"])

weather.columns = [
    str(c).strip().lower().replace(" ", "_")
    for c in weather.columns
]

save_csv(weather, "weather_odisha_processed.csv")


# ============================================================
# 4. HISTORICAL FLOOD
# ============================================================

print("\n[4] Processing historical flood inventory...")

flood_path = (
    DATA
    / "historical_flood"
    / "INDIA_FLOOD_INVENTORY_V3.shp"
)

flood = gpd.read_file(flood_path)

# Keep Odisha events only
flood = flood[
    flood["State"]
    .astype(str)
    .str.upper()
    .str.contains("ODISHA", na=False)
].copy()

flood["start_date"] = pd.to_datetime(
    flood["StartDate"],
    errors="coerce",
    dayfirst=True
)

flood["end_date"] = pd.to_datetime(
    flood["EndDate"],
    errors="coerce",
    dayfirst=True
)

flood["duration_days"] = (
    flood["end_date"] - flood["start_date"]
).dt.days + 1

# Keep useful attributes
flood_clean = flood[
    [
        "State",
        "Districts",
        "MainCause",
        "EventSour",
        "start_date",
        "end_date",
        "duration_days",
        "geometry",
    ]
].copy()

# Save GIS version
flood_clean.to_file(
    OUTPUT / "historical_flood_odisha.gpkg",
    layer="historical_flood",
    driver="GPKG"
)

# Save tabular version
flood_table = flood_clean.drop(columns="geometry")

save_csv(
    flood_table,
    "historical_flood_odisha_processed.csv"
)


# ============================================================
# 5. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING STEP 2A COMPLETE")
print("=" * 70)

print("\nProcessed files:")
for file in sorted(OUTPUT.iterdir()):
    print(f"   ✅ {file.name}")

print("\n" + "=" * 70)