from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(r"D:\Flood_Ai")

INPUT_FILE = (
    BASE_DIR
    / "Datasets" / "data"
    / "population"
    / "census_2011_towns.xlsx"
)

OUTPUT_DIR = (
    BASE_DIR
    / "Datasets" / "data"
    / "cities"
)

OUTPUT_FILE = OUTPUT_DIR / "odisha_cities.csv"


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_code(value, width):
    text = clean_text(value)

    if text == "":
        return ""

    if text.isdigit():
        return text.zfill(width)

    return text


def normalize_city_name(name):
    name = clean_text(name)

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    return name


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Census file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reading Census workbook...")

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="Data",
        dtype=object,
        engine="openpyxl"
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    required_columns = [
        "State",
        "District",
        "Subdistt",
        "Town/Village",
        "Name",
        "TRU",
        "TOT_P",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------
    # Odisha = Census state code 21
    # ---------------------------------------------------------

    df["State"] = (
        df["State"]
        .map(lambda x: clean_code(x, 2))
    )

    odisha = df[
        df["State"] == "21"
    ].copy()

    print(
        f"Odisha rows: {len(odisha):,}"
    )

    # ---------------------------------------------------------
    # Build district-code -> district-name mapping
    #
    # District aggregate rows use Town/Village = 000000
    # and contain the district name in the Name column.
    # ---------------------------------------------------------

    district_rows = odisha[
        (odisha["District"].map(
            lambda x: clean_code(x, 3)
        ) != "000")
        &
        (odisha["Town/Village"].map(
            lambda x: clean_code(x, 6)
        ) == "000000")
        &
        (
            odisha["Name"].notna()
        )
    ].copy()

    district_map = {}

    for _, row in district_rows.iterrows():

        district_code = clean_code(
            row["District"],
            3
        )

        name = normalize_city_name(
            row["Name"]
        )

        tru = clean_text(
            row["TRU"]
        ).lower()

        if (
            district_code
            and name
            and name.upper() != "ODISHA"
            and tru in {"total", "urban", "rural", ""}
        ):

            district_map.setdefault(
                district_code,
                name
            )

    print(
        f"District codes identified: "
        f"{len(district_map)}"
    )

    # ---------------------------------------------------------
    # Select actual urban town/city records.
    #
    # Town/Village != 000000 prevents us from accidentally
    # treating district urban totals as individual cities.
    # ---------------------------------------------------------

    odisha["district_code"] = (
        odisha["District"]
        .map(lambda x: clean_code(x, 3))
    )

    odisha["town_code"] = (
        odisha["Town/Village"]
        .map(lambda x: clean_code(x, 6))
    )

    odisha["urban_rural"] = (
        odisha["TRU"]
        .map(clean_text)
    )

    odisha["city_name"] = (
        odisha["Name"]
        .map(normalize_city_name)
    )

    odisha["population_2011"] = pd.to_numeric(
        odisha["TOT_P"],
        errors="coerce"
    )

    cities = odisha[
        (odisha["urban_rural"].str.lower() == "urban")
        &
        (odisha["town_code"] != "000000")
        &
        (odisha["city_name"] != "")
        &
        (odisha["population_2011"].notna())
        &
        (odisha["population_2011"] > 0)
    ].copy()

    # ---------------------------------------------------------
    # Attach district names
    # ---------------------------------------------------------

    cities["district"] = (
        cities["district_code"]
        .map(district_map)
    )

    cities = cities[
        cities["district"].notna()
    ].copy()

    # ---------------------------------------------------------
    # Keep only useful fields
    # ---------------------------------------------------------

    cities = cities[
        [
            "city_name",
            "district",
            "district_code",
            "town_code",
            "population_2011",
            "urban_rural",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Remove exact duplicates
    # ---------------------------------------------------------

    cities = cities.drop_duplicates(
        subset=[
            "district_code",
            "town_code",
            "city_name",
        ]
    )

    # ---------------------------------------------------------
    # Sort for predictable output
    # ---------------------------------------------------------

    cities = cities.sort_values(
        [
            "district",
            "population_2011",
            "city_name",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    cities["population_2011"] = (
        cities["population_2011"]
        .astype(int)
    )

    cities.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("CITY DATASET CREATED")
    print("=" * 60)
    print(
        f"Output: {OUTPUT_FILE}"
    )
    print(
        f"City/town records: {len(cities):,}"
    )
    print(
        f"Districts represented: "
        f"{cities['district'].nunique()}"
    )
    print(
        f"Total 2011 urban population: "
        f"{cities['population_2011'].sum():,}"
    )

    print()
    print("Sample records:")
    print(
        cities.head(15).to_string(
            index=False
        )
    )

    print()
    print("Records per district:")
    print(
        cities.groupby("district")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()
