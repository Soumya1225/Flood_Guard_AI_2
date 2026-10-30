from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio

BASE = Path(r"D:\Flood_Ai")
DATA = BASE / "Datasets" / "data"


def check_file(path):
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✅ {path.name} | {size_mb:.2f} MB")
        return True

    print(f"❌ MISSING: {path}")
    return False


print("=" * 70)
print("        FLOOD GUARD AI - DATASET VALIDATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. RAINFALL
# ---------------------------------------------------------
print("\n[1] RAINFALL")

rainfall_files = [
    DATA / "rainfall" / "rainfall_districtwise_daily_imd.csv",
    DATA / "rainfall" / "rainfall_statewise_daily_imd.csv",
]

for file in rainfall_files:
    if check_file(file):
        df = pd.read_csv(file)
        print(f"   Rows    : {len(df):,}")
        print(f"   Columns : {len(df.columns)}")


# ---------------------------------------------------------
# 2. WATER LEVEL
# ---------------------------------------------------------
print("\n[2] WATER LEVEL")

water_file = DATA / "water_level" / "rwl_tele_hr_cwc_004_1991_2020.csv"

if check_file(water_file):
    df = pd.read_csv(water_file)

    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {len(df.columns)}")
    print(f"   Stations: {df['Station'].nunique() if 'Station' in df.columns else 'N/A'}")


# ---------------------------------------------------------
# 3. WEATHER
# ---------------------------------------------------------
print("\n[3] WEATHER")

weather_file = DATA / "weather" / "weather_odisha.csv.csv"

if check_file(weather_file):
    df = pd.read_csv(weather_file, skiprows=13)

    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {len(df.columns)}")
    print(f"   Columns : {list(df.columns)}")


# ---------------------------------------------------------
# 4. ADMINISTRATIVE BOUNDARIES
# ---------------------------------------------------------
print("\n[4] ADMINISTRATIVE BOUNDARIES")

boundary_file = DATA / "boundaries" / "gadm41_IND.gpkg"

if check_file(boundary_file):
    layers = gpd.list_layers(boundary_file)

    print("   Layers:")

    for layer in layers["name"]:
        print(f"      - {layer}")


# ---------------------------------------------------------
# 5. LULC
# ---------------------------------------------------------
print("\n[5] LULC")

lulc_dir = DATA / "lulc"
lulc_files = list(lulc_dir.glob("*.tif"))

print(f"   GeoTIFF tiles: {len(lulc_files)}")

for file in lulc_files:
    try:
        with rasterio.open(file) as src:
            print(
                f"   ✅ {file.name} | "
                f"{src.width}x{src.height} | "
                f"CRS={src.crs}"
            )
    except Exception as e:
        print(f"   ❌ {file.name} | {e}")


# ---------------------------------------------------------
# 6. DEM
# ---------------------------------------------------------
print("\n[6] DEM")

dem_dir = DATA / "dem"
dem_files = list(dem_dir.glob("*.tif"))

print(f"   DEM tiles: {len(dem_files)}")

for file in dem_files[:5]:
    try:
        with rasterio.open(file) as src:
            print(
                f"   ✅ {file.name} | "
                f"{src.width}x{src.height} | "
                f"CRS={src.crs}"
            )
    except Exception as e:
        print(f"   ❌ {file.name} | {e}")

if len(dem_files) > 5:
    print(f"   ... and {len(dem_files) - 5} more tiles")


# ---------------------------------------------------------
# 7. DRAINAGE
# ---------------------------------------------------------
print("\n[7] DRAINAGE")

drainage_dir = DATA / "drainage"

drainage_files = list(drainage_dir.iterdir()) if drainage_dir.exists() else []

print(f"   Files found: {len(drainage_files)}")

for file in drainage_files:
    print(f"   - {file.name}")


# ---------------------------------------------------------
# 8. HISTORICAL FLOOD
# ---------------------------------------------------------
print("\n[8] HISTORICAL FLOOD")

historical_dir = DATA / "historical_flood"

historical_files = (
    list(historical_dir.iterdir())
    if historical_dir.exists()
    else []
)

print(f"   Files found: {len(historical_files)}")

for file in historical_files:
    print(f"   - {file.name}")


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print("All available datasets have been inspected.")
print("=" * 70)