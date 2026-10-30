from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "Datasets" / "processed_data"

print("=" * 70)
print("       FLOOD GUARD AI - FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# 1. LOAD WATER LEVEL DATA
# ============================================================

print("\n[1] Loading water level data...")

water = pd.read_csv(
    PROCESSED / "water_level_odisha_processed.csv"
)

water["data_acquisition_time"] = pd.to_datetime(
    water["data_acquisition_time"],
    errors="coerce",
    dayfirst=True
)

water["water_level_m"] = pd.to_numeric(
    water["water_level_m"],
    errors="coerce"
)

water["rl_of_zerogauge"] = pd.to_numeric(
    water["rl_of_zerogauge"],
    errors="coerce"
)

water["station"] = (
    water["station"]
    .astype(str)
    .str.strip()
)

water = water.dropna(
    subset=[
        "data_acquisition_time",
        "water_level_m"
    ]
)

water = water[
    water["station"].notna()
    & (water["station"] != "")
    & (water["station"].str.lower() != "nan")
].copy()

print("   Valid records:", len(water))
print("   Stations:", water["station"].nunique())


# ============================================================
# 2. DETERMINE STATION SCALE
# ============================================================

print("\n[2] Determining station water-level representation...")

station_stats = (
    water.groupby("station")
    .agg(
        raw_median=("water_level_m", "median"),
        raw_q1=("water_level_m", lambda s: s.quantile(0.25)),
        raw_q3=("water_level_m", lambda s: s.quantile(0.75)),
        raw_q95=("water_level_m", lambda s: s.quantile(0.95)),
        gauge_zero=("rl_of_zerogauge", "first")
    )
    .reset_index()
)

print("\n   Station scale diagnostics:")

print(
    station_stats[
        [
            "station",
            "raw_median",
            "raw_q3",
            "gauge_zero"
        ]
    ]
    .sort_values("station")
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 3. REMOVE INVALID LOW/SENTINEL CLUSTERS
# ============================================================

print("\n[3] Removing invalid low/sentinel readings...")

water = water.merge(
    station_stats,
    on="station",
    how="left"
)

# Stations with a usable positive gauge-zero reference.
has_gauge_zero = (
    water["gauge_zero"].notna()
    & (water["gauge_zero"] > 0)
)

# When a station has a positive gauge-zero reference,
# readings hundreds of metres below that reference are not
# treated as valid stage measurements.
gauge_lower_bound = (
    water["gauge_zero"] - 20.0
)

invalid_gauge_low = (
    has_gauge_zero
    & (
        water["water_level_m"]
        < gauge_lower_bound
    )
)

# Some stations have no gauge-zero value, but their distribution
# clearly contains a low sentinel cluster and a much higher
# operational cluster.
bimodal_high_scale = (
    (~has_gauge_zero)
    & (water["raw_q3"] > 20.0)
    & (
        water["raw_q1"]
        < (water["raw_q3"] * 0.10)
    )
)

# For such stations, use the upper operating cluster.
# q95-q3 is small when the upper cluster is stable.
cluster_spread = (
    water["raw_q95"]
    - water["raw_q3"]
).clip(lower=0.5)

cluster_lower_bound = (
    water["raw_q3"]
    - 5.0 * cluster_spread
)

invalid_bimodal_low = (
    bimodal_high_scale
    & (
        water["water_level_m"]
        < cluster_lower_bound
    )
)

invalid_low = (
    invalid_gauge_low
    | invalid_bimodal_low
)

print(
    "   Invalid low/sentinel records removed:",
    int(invalid_low.sum())
)

water = water.loc[
    ~invalid_low
].copy()


# ============================================================
# 4. CREATE COMMON STATION STAGE
# ============================================================

print("\n[4] Creating station-relative water stage...")

# Recalculate scale flags after filtering.

has_gauge_zero = (
    water["gauge_zero"].notna()
    & (water["gauge_zero"] > 0)
)

water["water_stage_m"] = np.where(
    has_gauge_zero,
    water["water_level_m"]
    - water["gauge_zero"],
    water["water_level_m"]
)


# ============================================================
# 5. REMOVE EXTREME STATION-LOCAL OUTLIERS
# ============================================================

print("\n[5] Removing station-local extreme readings...")

stage_stats = (
    water.groupby("station")["water_stage_m"]
    .agg(
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75)
    )
    .reset_index()
)

stage_stats["iqr"] = (
    stage_stats["q3"]
    - stage_stats["q1"]
)

water = water.merge(
    stage_stats,
    on="station",
    how="left"
)

# Generous IQR rule after the invalid low cluster has already
# been removed.
water["lower_bound"] = (
    water["q1"]
    - 5.0 * water["iqr"]
)

water["upper_bound"] = (
    water["q3"]
    + 5.0 * water["iqr"]
)

flat_station = (
    water["iqr"].fillna(0) < 0.001
)

water.loc[
    flat_station,
    "lower_bound"
] = (
    water.loc[
        flat_station,
        "q1"
    ] - 20.0
)

water.loc[
    flat_station,
    "upper_bound"
] = (
    water.loc[
        flat_station,
        "q3"
    ] + 20.0
)

outlier_mask = (
    (water["water_stage_m"] < water["lower_bound"])
    | (water["water_stage_m"] > water["upper_bound"])
)

print(
    "   Extreme records removed:",
    int(outlier_mask.sum())
)

water = water.loc[
    ~outlier_mask
].copy()


# ============================================================
# 5. DAILY WATER FEATURES
# ============================================================

print("\n[5] Building daily water-stage features...")

water["date"] = (
    water["data_acquisition_time"]
    .dt.floor("D")
)

water_daily = (
    water.groupby("date", as_index=False)
    .agg(
        water_level_mean_m=(
            "water_stage_m",
            "mean"
        ),
        water_level_max_m=(
            "water_stage_m",
            "max"
        ),
        water_level_min_m=(
            "water_stage_m",
            "min"
        ),
        station_count=(
            "station",
            "nunique"
        )
    )
)

water_daily = (
    water_daily
    .sort_values("date")
    .reset_index(drop=True)
)

print(
    "   Water dates:",
    water_daily["date"].min(),
    "to",
    water_daily["date"].max()
)

print(
    "   Daily rows:",
    len(water_daily)
)


# ============================================================
# 6. TEMPORAL FEATURES
# ============================================================

print("\n[6] Building temporal water features...")

water_daily["water_level_change_m"] = (
    water_daily["water_level_mean_m"]
    .diff()
    .fillna(0.0)
)

water_daily["water_level_3day_mean_m"] = (
    water_daily["water_level_mean_m"]
    .rolling(
        3,
        min_periods=1
    )
    .mean()
)

water_daily["water_level_7day_mean_m"] = (
    water_daily["water_level_mean_m"]
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


# ============================================================
# 7. WEATHER
# ============================================================

print("\n[7] Loading weather...")

weather = pd.read_csv(
    PROCESSED / "weather_odisha_processed.csv"
)

weather["date"] = pd.to_datetime(
    weather["date"],
    errors="coerce"
)

for col in [
    "t2m",
    "rh2m",
    "ws10m",
    "ps"
]:
    weather[col] = pd.to_numeric(
        weather[col],
        errors="coerce"
    )

weather = weather.dropna(
    subset=["date"]
)

weather_daily = (
    weather.groupby("date", as_index=False)
    .agg(
        temperature=("t2m", "mean"),
        humidity=("rh2m", "mean"),
        wind_speed=("ws10m", "mean"),
        pressure=("ps", "mean")
    )
)

print(
    "   Weather dates:",
    weather_daily["date"].min(),
    "to",
    weather_daily["date"].max()
)

print(
    "   Weather rows:",
    len(weather_daily)
)


# ============================================================
# 8. MERGE WATER + WEATHER
# ============================================================

print("\n[8] Combining water + weather...")

dynamic = pd.merge(
    water_daily,
    weather_daily,
    on="date",
    how="inner"
)

dynamic = (
    dynamic
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 9. CALENDAR FEATURES
# ============================================================

dynamic["month"] = (
    dynamic["date"].dt.month
)

dynamic["day_of_year"] = (
    dynamic["date"].dt.dayofyear
)


# ============================================================
# 10. FINAL CLEANING
# ============================================================

feature_columns = [
    "water_level_mean_m",
    "water_level_max_m",
    "water_level_min_m",
    "water_level_change_m",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "water_level_3day_mean_m",
    "water_level_7day_mean_m",
    "month",
    "day_of_year"
]

dynamic = dynamic.replace(
    [np.inf, -np.inf],
    np.nan
)

dynamic = dynamic.dropna(
    subset=feature_columns
).reset_index(drop=True)


# ============================================================
# 11. SAVE DYNAMIC DATA
# ============================================================

dynamic.to_csv(
    PROCESSED / "dynamic_environment_2020.csv",
    index=False
)

print("\n   Dynamic dataset saved.")
print("   Shape:", dynamic.shape)


# ============================================================
# 12. RAINFALL FEATURES
# ============================================================

print("\n[9] Loading rainfall...")

rainfall = pd.read_csv(
    PROCESSED / "rainfall_odisha_processed.csv"
)

rainfall["date"] = pd.to_datetime(
    rainfall["date"],
    errors="coerce"
)

rainfall["daily_actual"] = pd.to_numeric(
    rainfall["daily_actual"],
    errors="coerce"
)

rainfall["district"] = (
    rainfall["district"]
    .astype(str)
    .str.strip()
)

rainfall = rainfall.dropna(
    subset=[
        "date",
        "daily_actual"
    ]
)

rainfall_daily = (
    rainfall
    .groupby("date", as_index=False)
    .agg(
        rainfall_mean_mm=(
            "daily_actual",
            "mean"
        ),
        rainfall_max_mm=(
            "daily_actual",
            "max"
        ),
        rainfall_min_mm=(
            "daily_actual",
            "min"
        )
    )
)

rainfall_daily = (
    rainfall_daily
    .sort_values("date")
)

rainfall_daily["rainfall_3day_mm"] = (
    rainfall_daily["rainfall_mean_mm"]
    .rolling(
        3,
        min_periods=1
    )
    .sum()
)

rainfall_daily["rainfall_7day_mm"] = (
    rainfall_daily["rainfall_mean_mm"]
    .rolling(
        7,
        min_periods=1
    )
    .sum()
)

rainfall_daily.to_csv(
    PROCESSED / "current_rainfall_features.csv",
    index=False
)

print(
    "   Rainfall dates:",
    rainfall_daily["date"].min(),
    "to",
    rainfall_daily["date"].max()
)

print(
    "   Rainfall rows:",
    len(rainfall_daily)
)


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("\nCreated:")
print(
    PROCESSED / "dynamic_environment_2020.csv"
)
print(
    PROCESSED / "current_rainfall_features.csv"
)

print(
    "\nFinal dynamic shape:",
    dynamic.shape
)

print("\nWater-stage feature summary:")

print(
    dynamic[
        [
            "water_level_mean_m",
            "water_level_max_m",
            "water_level_min_m",
            "water_level_change_m"
        ]
    ]
    .describe()
    .round(3)
    .to_string()
)

print("\nNext step:")
print("ML target construction + model training")
