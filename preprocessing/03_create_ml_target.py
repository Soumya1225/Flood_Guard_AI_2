from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "Datasets" / "processed_data"

INPUT = PROCESSED / "dynamic_environment_2020.csv"
OUTPUT = PROCESSED / "ml_training_data.csv"

print("=" * 70)
print("       FLOOD GUARD AI - ML TARGET CREATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading dynamic environment data...")

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values("date").reset_index(drop=True)

print("   Rows:", len(df))
print("   Columns:", len(df.columns))


# ============================================================
# 2. CALCULATE HIGH-WATER THRESHOLD
# ============================================================

print("\n[2] Calculating high-water threshold...")

water_threshold = df["water_level_max_m"].quantile(0.95)

print(
    f"   95th percentile water level: "
    f"{water_threshold:.3f} m"
)


# ============================================================
# 3. CREATE TARGET
# ============================================================

print("\n[3] Creating ML target...")

df["high_water_event"] = (
    df["water_level_max_m"] >= water_threshold
).astype(int)

print(
    "   High-water events:",
    int(df["high_water_event"].sum())
)

print(
    "   Normal observations:",
    int((df["high_water_event"] == 0).sum())
)


# ============================================================
# 4. CREATE RISK LEVEL
# ============================================================

print("\n[4] Creating risk categories...")

df["water_level_percentile"] = (
    df["water_level_max_m"]
    .rank(pct=True)
)

df["water_risk_level"] = pd.cut(
    df["water_level_percentile"],
    bins=[0, 0.50, 0.75, 0.95, 1.0],
    labels=[
        "Low",
        "Moderate",
        "High",
        "Very High"
    ],
    include_lowest=True
)


# ============================================================
# 5. REMOVE INVALID VALUES
# ============================================================

print("\n[5] Cleaning ML dataset...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

required_columns = [
    "water_level_max_m",
    "water_level_mean_m",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "high_water_event"
]

df = df.dropna(
    subset=required_columns
)

print("   Final rows:", len(df))


# ============================================================
# 6. SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

print("\n[6] Dataset saved:")
print(OUTPUT)

print("\nFinal shape:", df.shape)

print("\nTarget distribution:")
print(
    df["high_water_event"]
    .value_counts()
    .sort_index()
)

print("\nRisk distribution:")
print(
    df["water_risk_level"]
    .value_counts()
)

print("\n" + "=" * 70)
print("ML TARGET CREATION COMPLETE")
print("=" * 70)
