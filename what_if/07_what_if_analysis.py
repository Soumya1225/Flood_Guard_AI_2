import os
import pandas as pd
import numpy as np


# ============================================================
# FLOOD GUARD AI - WHAT-IF EMERGENCY ANALYSIS
# ============================================================

print("=" * 70)
print("       FLOOD GUARD AI - WHAT-IF EMERGENCY ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

BASE_DIR = r"D:\Flood_Ai"

GIS_FILE = os.path.join(
    BASE_DIR,
    "gis",
    "odisha_spatial_risk.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "what_if"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Load GIS spatial risk data
# ------------------------------------------------------------

print("\n[1] Loading GIS spatial risk data...")

df = pd.read_csv(GIS_FILE)

print(f"   Districts loaded: {len(df)}")


# ------------------------------------------------------------
# 3. Check required columns
# ------------------------------------------------------------

required_columns = [
    "spatial_risk_score",
    "elevation_risk",
    "drainage_risk",
    "historical_risk",
    "lulc_risk"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("   Required columns verified.")


# ------------------------------------------------------------
# 4. Base risk score
# ------------------------------------------------------------

print("\n[2] Preparing base spatial risk...")

df["base_risk_score"] = df["spatial_risk_score"]


# ------------------------------------------------------------
# 5. Scenario calculation function
# ------------------------------------------------------------

def calculate_scenario(
    data,
    scenario_name,
    rainfall_change=0.0,
    water_level_change=0.0,
    drainage_change=0.0
):

    result = data.copy()

    # --------------------------------------------------------
    # Project-defined scenario effects
    #
    # Rainfall increase:
    # Higher rainfall increases flood pressure.
    #
    # Water-level increase:
    # Higher river/water level increases flood risk.
    #
    # Drainage effectiveness decrease:
    # Poorer drainage increases flood risk.
    #
    # These are scenario assumptions, NOT ground-truth
    # physical flood equations.
    # --------------------------------------------------------

    rainfall_effect = rainfall_change * 0.20

    water_effect = (
        water_level_change / 0.5
    ) * 0.10

    drainage_effect = (
        abs(drainage_change) * 0.20
    )

    total_adjustment = (
        rainfall_effect
        + water_effect
        + drainage_effect
    )

    # Apply scenario adjustment
    result["scenario_risk_score"] = (
        result["base_risk_score"]
        + total_adjustment
    )

    # Keep score between 0 and 1
    result["scenario_risk_score"] = (
        result["scenario_risk_score"]
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # Risk categories
    # --------------------------------------------------------

    result["scenario_risk_level"] = pd.cut(
        result["scenario_risk_score"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.0],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
    )

    result["scenario"] = scenario_name

    result["rainfall_change_percent"] = (
        rainfall_change * 100
    )

    result["water_level_change_m"] = (
        water_level_change
    )

    result["drainage_effectiveness_change_percent"] = (
        drainage_change * 100
    )

    return result


# ------------------------------------------------------------
# 6. Base scenario
# ------------------------------------------------------------

print("\n[3] Calculating BASE scenario...")

base = calculate_scenario(
    df,
    "Base",
    rainfall_change=0.0,
    water_level_change=0.0,
    drainage_change=0.0
)


# ------------------------------------------------------------
# 7. Rainfall +25%
# ------------------------------------------------------------

print("\n[4] Calculating RAINFALL +25% scenario...")

rainfall_scenario = calculate_scenario(
    df,
    "Rainfall +25%",
    rainfall_change=0.25,
    water_level_change=0.0,
    drainage_change=0.0
)


# ------------------------------------------------------------
# 8. Water Level +0.5 m
# ------------------------------------------------------------

print("\n[5] Calculating WATER LEVEL +0.5 m scenario...")

water_scenario = calculate_scenario(
    df,
    "Water Level +0.5m",
    rainfall_change=0.0,
    water_level_change=0.5,
    drainage_change=0.0
)


# ------------------------------------------------------------
# 9. Drainage effectiveness -15%
# ------------------------------------------------------------

print("\n[6] Calculating DRAINAGE -15% scenario...")

drainage_scenario = calculate_scenario(
    df,
    "Drainage Effectiveness -15%",
    rainfall_change=0.0,
    water_level_change=0.0,
    drainage_change=-0.15
)


# ------------------------------------------------------------
# 10. Combined worst-case scenario
# ------------------------------------------------------------

print("\n[7] Calculating COMBINED scenario...")

combined_scenario = calculate_scenario(
    df,
    "Combined Stress Scenario",
    rainfall_change=0.25,
    water_level_change=0.5,
    drainage_change=-0.15
)


# ------------------------------------------------------------
# 11. Combine all scenarios
# ------------------------------------------------------------

print("\n[8] Combining scenario results...")

all_scenarios = pd.concat(
    [
        base,
        rainfall_scenario,
        water_scenario,
        drainage_scenario,
        combined_scenario
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# 12. Save detailed results
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "what_if_results.csv"
)

all_scenarios.to_csv(
    output_file,
    index=False
)

print(f"\n   Results saved:")
print(f"   {output_file}")


# ------------------------------------------------------------
# 13. Create scenario summary
# ------------------------------------------------------------

print("\n[9] Creating scenario summary...")

summary_rows = []

for scenario_name, scenario_df in [
    ("Base", base),
    ("Rainfall +25%", rainfall_scenario),
    ("Water Level +0.5m", water_scenario),
    ("Drainage Effectiveness -15%", drainage_scenario),
    ("Combined Stress Scenario", combined_scenario)
]:

    counts = (
        scenario_df["scenario_risk_level"]
        .value_counts()
    )

    summary_rows.append({
        "scenario": scenario_name,
        "average_risk_score":
            scenario_df["scenario_risk_score"].mean(),
        "minimum_risk_score":
            scenario_df["scenario_risk_score"].min(),
        "maximum_risk_score":
            scenario_df["scenario_risk_score"].max(),
        "low_districts":
            counts.get("Low", 0),
        "moderate_districts":
            counts.get("Moderate", 0),
        "high_districts":
            counts.get("High", 0),
        "very_high_districts":
            counts.get("Very High", 0)
    })


summary = pd.DataFrame(summary_rows)

summary_file = os.path.join(
    OUTPUT_DIR,
    "what_if_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# 14. Display results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT-IF SCENARIO SUMMARY")
print("=" * 70)

print(
    summary.to_string(index=False)
)

print("\n" + "=" * 70)
print("WHAT-IF ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nDetailed results:")
print(output_file)

print(f"\nScenario summary:")
print(summary_file)

print("\nNOTE:")
print(
    "These scenario results use project-defined risk adjustments "
    "for decision-support demonstration. They are not calibrated "
    "physical flood predictions or ground-truth probabilities."
)

print("=" * 70)