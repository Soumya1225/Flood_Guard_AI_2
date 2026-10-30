import geopandas as gpd
import requests
from pathlib import Path
from tqdm.auto import tqdm
import math
from shapely.geometry import box

# ---------------------------------------------------------
# Copernicus GLO-90 public AWS bucket
# ---------------------------------------------------------
bucket = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"

# ---------------------------------------------------------
# 1. Load Odisha boundary
# ---------------------------------------------------------
states = gpd.read_file(
    r"D:\Flood_Ai\gadm41_IND.gpkg",
    layer="ADM_ADM_1"
)

odisha = states[
    states["NAME_1"].str.strip().str.lower() == "odisha"
].copy()

print("✅ Odisha boundary found.")

# ---------------------------------------------------------
# 2. Get Odisha geometry
# ---------------------------------------------------------
odisha_geom = odisha.geometry.union_all()

minx, miny, maxx, maxy = odisha.total_bounds

print("\nOdisha extent:")
print(f"Longitude: {minx:.2f} to {maxx:.2f}")
print(f"Latitude : {miny:.2f} to {maxy:.2f}")

# ---------------------------------------------------------
# 3. Find 1° x 1° tiles that intersect Odisha
# ---------------------------------------------------------
tiles = []

for lat in range(math.floor(miny), math.floor(maxy) + 1):
    for lon in range(math.floor(minx), math.floor(maxx) + 1):

        tile_geom = box(lon, lat, lon + 1, lat + 1)

        if tile_geom.intersects(odisha_geom):

            tiles.append((lat, lon))

print(f"\n✅ Tiles intersecting Odisha: {len(tiles)}")

# ---------------------------------------------------------
# 4. Create DEM folder
# ---------------------------------------------------------
output_folder = Path(r"D:\Flood_Ai\DEM")
output_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 5. Download GLO-90 DEM tiles
# ---------------------------------------------------------
print("\nStarting DEM downloads...\n")

success = 0
failed = 0

for lat, lon in tqdm(tiles, desc="Downloading DEM"):

    # Copernicus naming convention
    lat_code = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_code = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"

    tile_code = f"{lat_code}_00_{lon_code}_00"

    folder_name = (
        f"Copernicus_DSM_COG_30_{tile_code}_DEM"
    )

    filename = (
        f"Copernicus_DSM_COG_30_{tile_code}_DEM.tif"
    )

    url = f"{bucket}/{folder_name}/{filename}"

    output_file = output_folder / filename

    # Skip existing files
    if output_file.exists():
        print(f"Already exists: {filename}")
        success += 1
        continue

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=120
        )

        if response.status_code != 200:
            print(
                f"❌ Failed: {tile_code} "
                f"(HTTP {response.status_code})"
            )
            failed += 1
            continue

        with open(output_file, "wb") as f:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    f.write(chunk)

        print(f"✅ Downloaded: {filename}")
        success += 1

    except Exception as e:
        print(f"❌ Error: {tile_code} -> {e}")
        failed += 1

# ---------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------
print("\n====================================")
print("DEM DOWNLOAD SUMMARY")
print("====================================")
print(f"Successful: {success}")
print(f"Failed    : {failed}")
print(f"Location  : {output_folder}")
print("====================================")