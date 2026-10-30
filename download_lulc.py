import geopandas as gpd
import requests
from pathlib import Path
from tqdm.auto import tqdm

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------
# ESA WorldCover
# ---------------------------------------------------------
s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# ---------------------------------------------------------
# 1. Load the WorldCover grid
# ---------------------------------------------------------
grid_url = f"{s3_url_prefix}/esa_worldcover_grid.geojson"
grid = gpd.read_file(grid_url)

print("WorldCover grid loaded.")

# ---------------------------------------------------------
# 2. Load India's state boundaries
# ---------------------------------------------------------
states = gpd.read_file(
    PROJECT_ROOT / "gadm41_IND.gpkg",
    layer="ADM_ADM_1"
)

# ---------------------------------------------------------
# 3. Select Odisha
# ---------------------------------------------------------
odisha = states[
    states["NAME_1"].str.strip().str.lower() == "odisha"
].copy()

print("Odisha boundary found.")

# ---------------------------------------------------------
# 4. Match coordinate systems
# ---------------------------------------------------------
odisha = odisha.to_crs(grid.crs)

# ---------------------------------------------------------
# 5. Find WorldCover tiles covering Odisha
# ---------------------------------------------------------
tiles = grid[
    grid.intersects(odisha.geometry.union_all())
]

tile_names = tiles["ll_tile"].tolist()

print("\nTiles required for Odisha:")
print(tile_names)

# ---------------------------------------------------------
# 6. Create output folder
# ---------------------------------------------------------
output_folder = PROJECT_ROOT / "Datasets" / "LULC"
output_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 7. Download WorldCover 2021 v200 tiles
# ---------------------------------------------------------
year = 2021
version = "v200"

for tile in tqdm(tile_names, desc="Downloading LULC tiles"):

    url = (
        f"{s3_url_prefix}/{version}/{year}/map/"
        f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    )

    output_file = output_folder / Path(url).name

    if output_file.exists():
        print(f"Already downloaded: {output_file.name}")
        continue

    response = requests.get(
        url,
        stream=True,
        timeout=120
    )

    if response.status_code != 200:
        print(
            f"Failed to download {tile} "
            f"(HTTP {response.status_code})"
        )
        continue

    with open(output_file, "wb") as f:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {output_file.name}")

print("\nLULC download completed.")
print(f"Files saved in: {output_folder}")