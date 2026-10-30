from pathlib import Path
import pandas as pd
import geopandas as gpd
import pyogrio

BASE_DIR = Path(r"D:\Flood_Ai")

POPULATION_FILE = (
    BASE_DIR
    / "Datasets" / "data"
    / "population"
    / "ppp_IND_2020_1km_Aggregated_UNadj.csv"
)

BOUNDARY_FILE = (
    BASE_DIR
    / "Datasets" / "data"
    / "boundaries"
    / "gadm41_IND.gpkg"
)

OUTPUT_DIR = (
    BASE_DIR
    / "Datasets" / "processed_data"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "population_district_worldpop_2020.csv"
)

CHUNK_SIZE = 250_000


def choose_layer(path):
    layers = pyogrio.list_layers(path)

    if len(layers) == 0:
        raise ValueError(
            "No vector layers found in GADM GeoPackage."
        )

    layer_names = [
        str(item[0])
        for item in layers
    ]

    preferred_layers = [
        "ADM_ADM_2",
        "ADM_ADM2",
        "gadm41_IND_2",
        "ADM2",
    ]

    for preferred in preferred_layers:
        if preferred in layer_names:
            return preferred

    raise ValueError(
        "District-level GADM layer not found. "
        f"Available layers: {layer_names}"
    )


def find_column(columns, candidates):
    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def main():

    print("=" * 72)
    print("FLOOD GUARD AI - WORLDPOP DISTRICT POPULATION PREPROCESSING")
    print("=" * 72)

    print(
        f"Population grid: {POPULATION_FILE}"
    )

    print(
        f"Boundary file:   {BOUNDARY_FILE}"
    )

    if not POPULATION_FILE.exists():
        raise FileNotFoundError(
            f"Population CSV not found: {POPULATION_FILE}"
        )

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {BOUNDARY_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Read district boundaries
    # -------------------------------------------------------------

    layer = choose_layer(
        str(BOUNDARY_FILE)
    )

    print()
    print(
        f"GADM layer selected: {layer}"
    )

    districts = gpd.read_file(
        BOUNDARY_FILE,
        layer=layer
    )

    if districts.empty:
        raise ValueError(
            "GADM district layer is empty."
        )

    state_column = find_column(
        districts.columns,
        [
            "NAME_1",
            "name_1",
            "STATE",
            "state",
        ]
    )

    district_column = find_column(
        districts.columns,
        [
            "NAME_2",
            "name_2",
            "DISTRICT",
            "district",
        ]
    )

    if state_column is None:
        raise ValueError(
            "Could not identify state-name column in GADM."
        )

    if district_column is None:
        raise ValueError(
            "Could not identify district-name column in GADM."
        )

    print(
        f"State column:    {state_column}"
    )

    print(
        f"District column: {district_column}"
    )

    odisha = districts[
        districts[state_column]
        .astype(str)
        .str.strip()
        .str.lower()
        == "odisha"
    ].copy()

    if odisha.empty:
        raise ValueError(
            "No Odisha districts found in GADM."
        )

    odisha = odisha[
        [
            district_column,
            "geometry",
        ]
    ].copy()

    odisha.columns = [
        "district",
        "geometry",
    ]

    odisha["district"] = (
        odisha["district"]
        .astype(str)
        .str.strip()
    )

    odisha = odisha[
        odisha["district"] != ""
    ].copy()

    odisha = odisha[
        odisha.geometry.notna()
        & ~odisha.geometry.is_empty
    ].copy()

    if odisha.crs is None:
        raise ValueError(
            "GADM layer has no CRS."
        )

    # WorldPop X/Y coordinates are longitude/latitude.
    odisha = odisha.to_crs(
        "EPSG:4326"
    )

    print(
        f"Odisha districts found: {len(odisha)}"
    )

    if len(odisha) != 30:
        print(
            "WARNING: Expected 30 Odisha districts "
            f"but found {len(odisha)}."
        )

    # -------------------------------------------------------------
    # Create spatial index once
    # -------------------------------------------------------------

    print()
    print("Preparing district spatial index...")

    spatial_index = odisha.sindex

    # Odisha bounding box.
    minx, miny, maxx, maxy = (
        odisha.total_bounds
    )

    print(
        "Odisha boundary extent:"
    )

    print(
        f"Longitude: {minx:.6f} to {maxx:.6f}"
    )

    print(
        f"Latitude:  {miny:.6f} to {maxy:.6f}"
    )

    # -------------------------------------------------------------
    # Process WorldPop CSV in chunks
    # -------------------------------------------------------------

    totals = {
        district: 0.0
        for district in odisha["district"]
    }

    matched_points = {
        district: 0
        for district in odisha["district"]
    }

    total_rows_read = 0
    total_points_in_odisha_bbox = 0
    total_positive_population_rows = 0

    print()
    print(
        f"Reading WorldPop CSV in chunks of {CHUNK_SIZE:,} rows..."
    )

    try:

        reader = pd.read_csv(
            POPULATION_FILE,
            usecols=["X", "Y", "Z"],
            chunksize=CHUNK_SIZE,
            dtype={
                "X": "float64",
                "Y": "float64",
                "Z": "float64",
            },
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not read the WorldPop CSV with expected "
            "columns X,Y,Z."
        ) from exc

    chunk_number = 0

    for chunk in reader:

        chunk_number += 1

        total_rows_read += len(chunk)

        # Remove invalid coordinate/population rows.
        chunk = chunk.dropna(
            subset=[
                "X",
                "Y",
                "Z",
            ]
        )

        # Only positive population contributes to exposure.
        chunk = chunk[
            chunk["Z"] > 0
        ].copy()

        total_positive_population_rows += len(chunk)

        if chunk.empty:
            continue

        # Fast Odisha bounding-box filter.
        chunk = chunk[
            (chunk["X"] >= minx)
            & (chunk["X"] <= maxx)
            & (chunk["Y"] >= miny)
            & (chunk["Y"] <= maxy)
        ].copy()

        total_points_in_odisha_bbox += len(chunk)

        if chunk.empty:
            continue

        points = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(
                chunk["X"],
                chunk["Y"]
            ),
            crs="EPSG:4326",
        )

        # Spatial join to district polygons.
        joined = gpd.sjoin(
            points,
            odisha,
            how="inner",
            predicate="within",
        )

        if not joined.empty:

            grouped = (
                joined
                .groupby("district")
                .agg(
                    population_sum=("Z", "sum"),
                    point_count=("Z", "size"),
                )
            )

            for district, row in grouped.iterrows():

                totals[district] += float(
                    row["population_sum"]
                )

                matched_points[district] += int(
                    row["point_count"]
                )

        if (
            chunk_number == 1
            or chunk_number % 5 == 0
        ):
            print(
                f"Processed chunks: {chunk_number:,} | "
                f"rows read: {total_rows_read:,} | "
                f"Odisha-bbox points: "
                f"{total_points_in_odisha_bbox:,}"
            )

    # -------------------------------------------------------------
    # Build final district table
    # -------------------------------------------------------------

    result = odisha[
        ["district"]
    ].copy()

    result["population_worldpop_2020"] = (
        result["district"]
        .map(totals)
        .fillna(0.0)
    )

    result["grid_points_matched"] = (
        result["district"]
        .map(matched_points)
        .fillna(0)
        .astype(int)
    )

    result["data_source"] = (
        "WorldPop 2020 1km Aggregated UNadjusted"
    )

    result["population_year"] = 2020

    result = result.sort_values(
        "population_worldpop_2020",
        ascending=False,
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    districts_with_population = int(
        (
            result["population_worldpop_2020"]
            > 0
        ).sum()
    )

    total_population = float(
        result["population_worldpop_2020"]
        .sum()
    )

    print()
    print("=" * 72)
    print("WORLDPOP DISTRICT POPULATION DATASET CREATED")
    print("=" * 72)

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Rows read from WorldPop CSV: "
        f"{total_rows_read:,}"
    )

    print(
        f"Positive population rows: "
        f"{total_positive_population_rows:,}"
    )

    print(
        f"Points inside Odisha bounding box: "
        f"{total_points_in_odisha_bbox:,}"
    )

    print(
        f"Districts processed: "
        f"{len(result)}"
    )

    print(
        f"Districts with population > 0: "
        f"{districts_with_population}"
    )

    print(
        f"Total WorldPop 2020 population in Odisha: "
        f"{total_population:,.2f}"
    )

    print()
    print("Top 10 districts by WorldPop population:")

    print(
        result[
            [
                "district",
                "population_worldpop_2020",
                "grid_points_matched",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("District coverage:")

    print(
        result[
            [
                "district",
                "population_worldpop_2020",
                "grid_points_matched",
            ]
        ]
        .to_string(index=False)
    )

    print()
    print(
        "NOTE: WorldPop values represent population estimates "
        "for 2020 and are not current 2026 population figures."
    )


if __name__ == "__main__":
    main()
