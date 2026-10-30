from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# FLOOD GUARD AI - WATER LEVEL FORECASTING
# 7-DAY FORECASTING PIPELINE
# ============================================================

BASE = Path(__file__).resolve().parents[1]

PROCESSED = BASE / "Datasets" / "processed_data"
OUTPUT_DIR = BASE / "forecasting"

INPUT = PROCESSED / "dynamic_environment_2020.csv"

FORECAST_OUTPUT = OUTPUT_DIR / "water_level_forecast.csv"
NEXT_FORECAST_OUTPUT = OUTPUT_DIR / "next_water_level_forecast.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "water_level_mean_m"

LAG_DAYS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7]

TEST_RATIO = 0.20
RANDOM_STATE = 42

FORECAST_HORIZON = 7

ANOMALY_WINDOW = 7
ANOMALY_SIGMA = 6.0
MIN_ANOMALY_THRESHOLD = 15.0


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "rolling_3",
    "rolling_7",
    "change_1d",
    "change_3d",
    "month",
    "day_of_year",
    "day_of_week",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("       FLOOD GUARD AI - WATER LEVEL FORECAST")
print("=" * 70)

print("\n[1] Loading dynamic environmental data...")

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print("   Original shape:", df.shape)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    TARGET,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CLEAN DATE AND TARGET
# ============================================================

print("\n[2] Cleaning date and water-level values...")

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

df = df[
    ["date", TARGET]
].dropna()


# ============================================================
# DAILY AGGREGATION
# ============================================================

print("\n[3] Aggregating water-level observations by day...")

df["date"] = df["date"].dt.floor("D")

df = (
    df.groupby(
        "date",
        as_index=False,
    )[TARGET]
    .mean()
    .sort_values("date")
    .reset_index(drop=True)
)

print(
    "   Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date(),
)

print(
    "   Available daily observations:",
    len(df),
)


# ============================================================
# KEEP ORIGINAL OBSERVATIONS
# ============================================================

df["observed_water_level_m"] = df[TARGET]


# ============================================================
# CREATE COMPLETE DAILY TIMELINE
# ============================================================

print("\n[4] Creating continuous daily timeline...")

full_date_range = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="D",
)

df = (
    df.set_index("date")
    .reindex(full_date_range)
    .rename_axis("date")
    .reset_index()
)


# ============================================================
# MARK MISSING OBSERVATIONS
# ============================================================

df["observation_available"] = (
    df["observed_water_level_m"].notna()
)

missing_before = int(
    df["observed_water_level_m"].isna().sum()
)

print(
    "   Missing daily observations:",
    missing_before,
)


# ============================================================
# MODELING SERIES
# ============================================================

df["model_water_level_m"] = (
    df["observed_water_level_m"]
)


# ============================================================
# INTERPOLATE MISSING VALUES FOR MODELING
# ============================================================

if missing_before > 0:

    print(
        "   Interpolating missing values for model features..."
    )

    df["model_water_level_m"] = (
        df["model_water_level_m"]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )


# ============================================================
# EXTREME OBSERVATION DETECTION
# ============================================================

print("\n[5] Detecting extreme observed anomalies...")

series = df["model_water_level_m"].copy()

rolling_median = (
    series
    .rolling(
        window=ANOMALY_WINDOW,
        center=True,
        min_periods=3,
    )
    .median()
)

absolute_deviation = (
    series - rolling_median
).abs()

rolling_mad = (
    absolute_deviation
    .rolling(
        window=ANOMALY_WINDOW,
        center=True,
        min_periods=3,
    )
    .median()
)

robust_sigma = (
    1.4826 * rolling_mad
)

dynamic_threshold = np.maximum(
    ANOMALY_SIGMA * robust_sigma,
    MIN_ANOMALY_THRESHOLD,
)

anomaly_mask = (
    df["observation_available"]
    &
    rolling_median.notna()
    &
    (
        absolute_deviation
        >
        dynamic_threshold
    )
)

anomaly_count = int(
    anomaly_mask.sum()
)

df["anomaly_detected"] = (
    anomaly_mask.astype(bool)
)

print(
    "   Detected observed anomalies:",
    anomaly_count,
)


# ============================================================
# REPLACE DETECTED EXTREME OBSERVATIONS
# FOR MODELING ONLY
# ============================================================

if anomaly_count > 0:

    print(
        "   Replacing detected extreme observations "
        "for modeling..."
    )

    df.loc[
        anomaly_mask,
        "model_water_level_m",
    ] = np.nan

    df["model_water_level_m"] = (
        df["model_water_level_m"]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

    print("\n   Corrected observations:")

    corrected_rows = df[
        anomaly_mask
    ][
        [
            "date",
            "observed_water_level_m",
            "model_water_level_m",
        ]
    ]

    for _, row in corrected_rows.iterrows():

        print(
            f"   {row['date'].date()} : "
            f"{row['observed_water_level_m']:.3f} m -> "
            f"{row['model_water_level_m']:.3f} m"
        )

else:

    print(
        "   No extreme observed anomalies detected."
    )


# ============================================================
# CREATE TIME FEATURES
# ============================================================

print("\n[6] Creating time-series features...")

df["month"] = (
    df["date"].dt.month
)

df["day_of_year"] = (
    df["date"].dt.dayofyear
)

df["day_of_week"] = (
    df["date"].dt.dayofweek
)


# ============================================================
# CREATE LAG FEATURES
# ============================================================

for lag in LAG_DAYS:

    df[f"lag_{lag}"] = (
        df["model_water_level_m"]
        .shift(lag)
    )


# ============================================================
# CREATE ROLLING FEATURES
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"rolling_{window}"] = (
        df["model_water_level_m"]
        .shift(1)
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )


# ============================================================
# CREATE CHANGE FEATURES
# ============================================================

df["change_1d"] = (
    df["model_water_level_m"].shift(1)
    -
    df["model_water_level_m"].shift(2)
)

df["change_3d"] = (
    df["model_water_level_m"].shift(1)
    -
    df["model_water_level_m"].shift(4)
)


# ============================================================
# REMOVE ROWS WITHOUT MODEL HISTORY
# ============================================================

df = (
    df.dropna(
        subset=FEATURE_COLUMNS
    )
    .reset_index(drop=True)
)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print(
    "\n[7] Creating chronological train/test split..."
)

split_index = int(
    len(df) * (1 - TEST_RATIO)
)

train = df.iloc[
    :split_index
].copy()

test = df.iloc[
    split_index:
].copy()


X_train = train[
    FEATURE_COLUMNS
]

X_test = test[
    FEATURE_COLUMNS
]

y_train = train[
    "model_water_level_m"
]

y_test_model = test[
    "model_water_level_m"
]

observed_mask_test = (
    test["observation_available"]
    &
    test["observed_water_level_m"].notna()
)

y_test_observed = test.loc[
    observed_mask_test,
    "observed_water_level_m"
]


print(
    "   Training rows:",
    len(train),
)

print(
    "   Testing rows:",
    len(test),
)

print(
    "   Test rows with genuine observations:",
    int(observed_mask_test.sum()),
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print(
    "\n[8] Training Random Forest Regressor..."
)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)

print(
    "   Training complete."
)


# ============================================================
# HISTORICAL FORECAST
# ============================================================

print(
    "\n[9] Generating historical forecasts..."
)

predictions = model.predict(
    X_test
)


# ============================================================
# ROBUST DAILY CHANGE LIMIT
# ============================================================

historical_changes = (
    train["model_water_level_m"]
    .diff()
    .dropna()
    .abs()
)

median_change = float(
    historical_changes.median()
)

mad_change = float(
    (
        historical_changes
        -
        median_change
    )
    .abs()
    .median()
)

robust_limit = (
    median_change
    +
    6.0
    * 1.4826
    * mad_change
)

quantile_limit = float(
    historical_changes.quantile(
        0.99
    )
)

max_reasonable_change = min(
    robust_limit,
    quantile_limit,
)

max_reasonable_change = max(
    max_reasonable_change,
    10.0,
)


# ============================================================
# BOUND HISTORICAL PREDICTIONS
# ============================================================

previous_levels = (
    X_test[
        "lag_1"
    ].to_numpy()
)

lower_bound = (
    previous_levels
    -
    max_reasonable_change
)

upper_bound = (
    previous_levels
    +
    max_reasonable_change
)

bounded_predictions = np.clip(
    predictions,
    lower_bound,
    upper_bound,
)


# ============================================================
# MODEL METRICS
# ============================================================

mae_model = mean_absolute_error(
    y_test_model,
    bounded_predictions,
)

rmse_model = np.sqrt(
    mean_squared_error(
        y_test_model,
        bounded_predictions,
    )
)

if observed_mask_test.any():

    observed_predictions = (
        bounded_predictions[
            observed_mask_test.to_numpy()
        ]
    )

    mae_observed = (
        mean_absolute_error(
            y_test_observed,
            observed_predictions,
        )
    )

    rmse_observed = np.sqrt(
        mean_squared_error(
            y_test_observed,
            observed_predictions,
        )
    )

else:

    mae_observed = float("nan")
    rmse_observed = float("nan")


# ============================================================
# ESTIMATE FORECAST ERROR RANGE
# ============================================================

test_residuals = (
    y_test_model.to_numpy()
    -
    bounded_predictions
)

residual_std = float(
    np.std(
        test_residuals
    )
)

# Keep the uncertainty estimate bounded so that it does not
# produce an unusably wide dashboard range.
residual_std = min(
    residual_std,
    max_reasonable_change * 2.0,
)


# ============================================================
# SAVE HISTORICAL FORECAST
# ============================================================

forecast = test[
    [
        "date",
        "observed_water_level_m",
        "model_water_level_m",
        "observation_available",
        "anomaly_detected",
    ]
].copy()

forecast[
    "water_level_mean_m"
] = forecast[
    "observed_water_level_m"
]

forecast[
    "predicted_water_level_m"
] = bounded_predictions

forecast[
    "absolute_error_m"
] = (
    forecast[
        "observed_water_level_m"
    ]
    -
    forecast[
        "predicted_water_level_m"
    ]
).abs()

forecast[
    "absolute_error_m"
] = (
    forecast[
        "absolute_error_m"
    ].where(
        forecast[
            "observed_water_level_m"
        ].notna()
    )
)

forecast[
    "prediction_change_m"
] = (
    forecast[
        "predicted_water_level_m"
    ]
    -
    forecast[
        "model_water_level_m"
    ].shift(1)
).abs()

forecast = forecast[
    [
        "date",
        "water_level_mean_m",
        "predicted_water_level_m",
        "absolute_error_m",
        "prediction_change_m",
        "model_water_level_m",
        "observation_available",
        "anomaly_detected",
    ]
]


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

forecast.to_csv(
    FORECAST_OUTPUT,
    index=False,
)


# ============================================================
# 7-DAY RECURSIVE FORECAST
# ============================================================

print(
    "\n[10] Generating 7-day recursive forecast..."
)

history_dates = (
    df["date"]
    .tolist()
)

history_levels = (
    df["model_water_level_m"]
    .astype(float)
    .tolist()
)

latest_date = (
    history_dates[-1]
)

forecast_rows = []

previous_forecast_level = (
    history_levels[-1]
)


for forecast_day in range(
    1,
    FORECAST_HORIZON + 1,
):

    forecast_date = (
        latest_date
        +
        pd.Timedelta(
            days=forecast_day
        )
    )

    values = list(
        history_levels
    )

    if len(values) < 7:

        raise ValueError(
            "Not enough historical observations "
            "for 7-day recursive forecasting."
        )

    lag_1 = values[-1]
    lag_2 = values[-2]
    lag_3 = values[-3]
    lag_7 = values[-7]

    rolling_3 = float(
        np.mean(
            values[-3:]
        )
    )

    rolling_7 = float(
        np.mean(
            values[-7:]
        )
    )

    change_1d = (
        values[-1]
        -
        values[-2]
    )

    change_3d = (
        values[-1]
        -
        values[-4]
    )

    next_features = pd.DataFrame(
        [{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_7": lag_7,
            "rolling_3": rolling_3,
            "rolling_7": rolling_7,
            "change_1d": change_1d,
            "change_3d": change_3d,
            "month": int(
                forecast_date.month
            ),
            "day_of_year": int(
                forecast_date.dayofyear
            ),
            "day_of_week": int(
                forecast_date.dayofweek
            ),
        }]
    )

    raw_prediction = float(
        model.predict(
            next_features[
                FEATURE_COLUMNS
            ]
        )[0]
    )

    # Limit recursive one-step movement.
    bounded_prediction = float(
        np.clip(
            raw_prediction,
            previous_forecast_level
            -
            max_reasonable_change,
            previous_forecast_level
            +
            max_reasonable_change,
        )
    )

    prediction_change = float(
        bounded_prediction
        -
        previous_forecast_level
    )

    if prediction_change > 2.0:

        trend = "Rising"

    elif prediction_change < -2.0:

        trend = "Falling"

    else:

        trend = "Stable"

    lower_estimate = max(
        0.0,
        bounded_prediction
        -
        residual_std,
    )

    upper_estimate = (
        bounded_prediction
        +
        residual_std
    )

    forecast_rows.append(
        {
            "forecast_day": forecast_day,
            "forecast_date": (
                forecast_date.strftime(
                    "%Y-%m-%d"
                )
            ),
            "predicted_water_level_m": (
                bounded_prediction
            ),
            "prediction_change_m": (
                abs(
                    prediction_change
                )
            ),
            "trend": trend,
            "lower_estimate_m": (
                lower_estimate
            ),
            "upper_estimate_m": (
                upper_estimate
            ),
        }
    )

    # Add the forecast back into the recursive history.
    history_levels.append(
        bounded_prediction
    )

    history_dates.append(
        forecast_date
    )

    previous_forecast_level = (
        bounded_prediction
    )


next_forecast = pd.DataFrame(
    forecast_rows
)


# ============================================================
# SAVE 7-DAY FORECAST
# ============================================================

next_forecast.to_csv(
    NEXT_FORECAST_OUTPUT,
    index=False,
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("FORECAST MODEL PERFORMANCE")
print("-" * 70)

print(
    f"Model MAE       : {mae_model:.4f} m"
)

print(
    f"Model RMSE      : {rmse_model:.4f} m"
)

print(
    f"Observed MAE    : {mae_observed:.4f} m"
)

print(
    f"Observed RMSE   : {rmse_observed:.4f} m"
)

print(
    f"\nRobust maximum daily change: "
    f"{max_reasonable_change:.4f} m"
)

print(
    f"Observed anomalies detected: "
    f"{anomaly_count}"
)

print(
    f"Missing observations retained: "
    f"{missing_before}"
)

print(
    f"Forecast horizon: "
    f"{FORECAST_HORIZON} days"
)

print(
    f"Estimated forecast residual spread: "
    f"{residual_std:.4f} m"
)

print(
    "\nHistorical forecast saved:"
)

print(
    FORECAST_OUTPUT
)

print(
    "\n7-day forecast:"
)

for row in forecast_rows:

    print(
        f"   Day {row['forecast_day']}: "
        f"{row['forecast_date']} -> "
        f"{row['predicted_water_level_m']:.3f} m "
        f"({row['trend']})"
    )

print(
    "\n7-day forecast saved:"
)

print(
    NEXT_FORECAST_OUTPUT
)

print("\n" + "=" * 70)
print(
    "7-DAY WATER LEVEL FORECASTING COMPLETE"
)
print("=" * 70)