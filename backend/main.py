from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd
import geopandas as gpd
import joblib
import numpy as np

import os
import math
import json
import re
import sqlite3
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Flood Guard AI API",
    description=(
        "AI-Based Urban Flood Forecasting, "
        "Impact Mapping and Emergency Decision Support System"
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# FILE PATHS
# ============================================================

GIS_FILE = os.path.join(
    BASE_DIR, "gis", "odisha_spatial_risk.csv"
)

GIS_GPKG_FILE = os.path.join(
    BASE_DIR, "gis", "odisha_spatial_risk.gpkg"
)

BOUNDARY_FILE = os.path.join(
    BASE_DIR, "Datasets", "data", "boundaries", "gadm41_IND.gpkg"
)

HISTORICAL_FLOOD_FILE = os.path.join(
    BASE_DIR,
    "Datasets", "data",
    "historical_flood",
    "INDIA_FLOOD_INVENTORY_V3.shp",
)

RIVER_FILE = os.path.join(
    BASE_DIR, "Datasets", "data", "drainage", "River_Network.shp"
)

RAINFALL_DISTRICT_FILE = os.path.join(
    BASE_DIR,
    "Datasets", "data",
    "rainfall",
    "rainfall_districtwise_daily_imd.csv",
)

RAINFALL_STATE_FILE = os.path.join(
    BASE_DIR,
    "Datasets", "data",
    "rainfall",
    "rainfall_statewise_daily_imd.csv",
)

WEATHER_FILE = os.path.join(
    BASE_DIR, "Datasets", "data", "weather", "weather_odisha.csv.csv"
)

WHAT_IF_FILE = os.path.join(
    BASE_DIR, "what_if", "what_if_results.csv"
)

WHAT_IF_SUMMARY_FILE = os.path.join(
    BASE_DIR, "what_if", "what_if_summary.csv"
)

FORECAST_FILE = os.path.join(
    BASE_DIR, "forecasting", "water_level_forecast.csv"
)

NEXT_FORECAST_FILE = os.path.join(
    BASE_DIR, "forecasting", "next_water_level_forecast.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "flood_guard_random_forest.pkl"
)

ML_TRAINING_FILE = os.path.join(
    BASE_DIR, "Datasets", "processed_data", "ml_training_data.csv"
)

# Optional future city dataset. We do not invent city exposure data.
CITY_FILE = os.path.join(
    BASE_DIR, "Datasets", "data", "cities", "odisha_cities.csv"
)


# WorldPop 2020 district population exposure dataset.
POPULATION_DISTRICT_FILE = os.path.join(
    BASE_DIR,
    "Datasets", "processed_data",
    "population_district_worldpop_2020.csv",
)

DATABASE_DIR = os.path.join(
    BASE_DIR, "database"
)

DATABASE_FILE = os.path.join(
    DATABASE_DIR, "flood_guard.db"
)


# ============================================================
# MODEL FEATURE DEFINITIONS
# ============================================================

MODEL_FEATURES = [
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
    "day_of_year",
]


FEATURE_DISPLAY_NAMES = {
    "water_level_mean_m": "Water Level Mean",
    "water_level_max_m": "Water Level Maximum",
    "water_level_min_m": "Water Level Minimum",
    "water_level_change_m": "Water Level Change",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "wind_speed": "Wind Speed",
    "pressure": "Pressure",
    "water_level_3day_mean_m": "3-Day Water-Level Mean",
    "water_level_7day_mean_m": "7-Day Water-Level Mean",
    "month": "Month",
    "day_of_year": "Day of Year",
}


# ============================================================
# REQUEST MODELS
# ============================================================

class FloodPredictionRequest(BaseModel):
    water_level_mean_m: float = Field(...)
    water_level_max_m: float = Field(...)
    water_level_min_m: float = Field(...)
    water_level_change_m: float = Field(...)
    temperature: float = Field(...)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(...)
    pressure: float = Field(...)
    water_level_3day_mean_m: float = Field(...)
    water_level_7day_mean_m: float = Field(...)
    month: int = Field(..., ge=1, le=12)
    day_of_year: int = Field(..., ge=1, le=366)


class WhatIfAnalysisRequest(BaseModel):
    rainfall_change_percent: float = Field(
        0.0, ge=-100.0, le=500.0
    )
    water_level_change_m: float = Field(
        0.0, ge=-10.0, le=10.0
    )
    drainage_effectiveness_change_percent: float = Field(
        0.0, ge=-100.0, le=100.0
    )


# ============================================================
# LOAD PROJECT DATA
# ============================================================

district_data = (
    pd.read_csv(GIS_FILE)
    if os.path.exists(GIS_FILE)
    else pd.DataFrame()
)

what_if_data = (
    pd.read_csv(WHAT_IF_FILE)
    if os.path.exists(WHAT_IF_FILE)
    else pd.DataFrame()
)

what_if_summary = (
    pd.read_csv(WHAT_IF_SUMMARY_FILE)
    if os.path.exists(WHAT_IF_SUMMARY_FILE)
    else pd.DataFrame()
)

forecast_data = (
    pd.read_csv(FORECAST_FILE)
    if os.path.exists(FORECAST_FILE)
    else pd.DataFrame()
)

next_forecast_data = (
    pd.read_csv(NEXT_FORECAST_FILE)
    if os.path.exists(NEXT_FORECAST_FILE)
    else pd.DataFrame()
)

ml_training_data = pd.DataFrame()
if os.path.exists(ML_TRAINING_FILE):
    try:
        ml_training_data = pd.read_csv(ML_TRAINING_FILE)
    except Exception as exc:
        print(f"Failed to load ML training data: {exc}")


# ============================================================
# LOAD FLOOD MODEL
# ============================================================

flood_model = None

if os.path.exists(MODEL_FILE):
    try:
        flood_model = joblib.load(MODEL_FILE)
        print(f"Flood prediction model loaded: {MODEL_FILE}")
    except Exception as exc:
        print(f"Failed to load flood prediction model: {exc}")


# ============================================================
# OPTIONAL DISTRICT GEOMETRY
# ============================================================

boundary_gdf = None

district_geometry_gdf = None

try:
    if os.path.exists(BOUNDARY_FILE):
        boundary_gdf = gpd.read_file(
            BOUNDARY_FILE,
            layer="ADM_ADM_2",
        )

        if "NAME_1" in boundary_gdf.columns:
            boundary_gdf = boundary_gdf[
                boundary_gdf["NAME_1"].astype(str).str.lower().eq("odisha")
            ].copy()
        elif "STATE" in boundary_gdf.columns:
            boundary_gdf = boundary_gdf[
                boundary_gdf["STATE"].astype(str).str.lower().eq("odisha")
            ].copy()

        if not boundary_gdf.empty:
            if boundary_gdf.crs is None:
                boundary_gdf = boundary_gdf.set_crs(epsg=4326)
            district_geometry_gdf = boundary_gdf.to_crs(epsg=4326)

except Exception as exc:
    print(f"District geometry preload failed: {exc}")


# ============================================================
# DATABASE / HISTORY
# ============================================================

def get_db_connection():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_db_connection()
    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT,
                reference_date TEXT,
                prediction_label TEXT,
                risk_level TEXT,
                flood_probability_percent REAL,
                features_json TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                overall_status TEXT,
                alert_level TEXT,
                priority_district_count INTEGER,
                message TEXT,
                priority_districts_json TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scenario_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                rainfall_change_percent REAL,
                water_level_change_m REAL,
                drainage_effectiveness_change_percent REAL,
                average_risk_score REAL,
                maximum_risk_score REAL,
                very_high_districts INTEGER
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS report_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                report_type TEXT,
                summary_json TEXT
            )
            """
        )

        connection.commit()
    finally:
        connection.close()


initialize_database()


def save_prediction_history(result, source):
    try:
        connection = get_db_connection()

        # Avoid duplicate rows when the same latest prediction is repeatedly loaded.
        existing = connection.execute(
            """
            SELECT prediction_label, risk_level,
                   flood_probability_percent, reference_date
            FROM prediction_history
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        current_signature = (
            result.get("prediction_label"),
            result.get("risk_level"),
            result.get("flood_probability_percent"),
            result.get("reference_date"),
        )

        previous_signature = (
            (
                existing["prediction_label"],
                existing["risk_level"],
                existing["flood_probability_percent"],
                existing["reference_date"],
            )
            if existing
            else None
        )

        if current_signature != previous_signature:
            connection.execute(
                """
                INSERT INTO prediction_history (
                    created_at, source, reference_date,
                    prediction_label, risk_level,
                    flood_probability_percent, features_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    source,
                    result.get("reference_date"),
                    result.get("prediction_label"),
                    result.get("risk_level"),
                    result.get("flood_probability_percent"),
                    json.dumps(
                        result.get("features_used", {}),
                        default=str,
                    ),
                ),
            )
            connection.commit()

        connection.close()
    except Exception as exc:
        print(f"Prediction history save failed: {exc}")


def save_scenario_history(result):
    try:
        connection = get_db_connection()
        scenario = result.get("scenario", {})
        summary = result.get("summary", {})

        connection.execute(
            """
            INSERT INTO scenario_history (
                created_at,
                rainfall_change_percent,
                water_level_change_m,
                drainage_effectiveness_change_percent,
                average_risk_score,
                maximum_risk_score,
                very_high_districts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                scenario.get("rainfall_change_percent"),
                scenario.get("water_level_change_m"),
                scenario.get("drainage_effectiveness_change_percent"),
                summary.get("average_risk_score"),
                summary.get("maximum_risk_score"),
                summary.get("very_high_districts"),
            ),
        )

        connection.commit()
        connection.close()
    except Exception as exc:
        print(f"Scenario history save failed: {exc}")


def save_report_history(report_type, summary):
    try:
        connection = get_db_connection()
        connection.execute(
            """
            INSERT INTO report_history (
                created_at, report_type, summary_json
            )
            VALUES (?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                report_type,
                json.dumps(summary, default=str),
            ),
        )
        connection.commit()
        connection.close()
    except Exception as exc:
        print(f"Report history save failed: {exc}")


def save_alert_history_if_changed(unified):
    try:
        connection = get_db_connection()

        last = connection.execute(
            """
            SELECT overall_status, alert_level
            FROM alert_history
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        current_pair = (
            unified.get("overall_status"),
            unified.get("alert_level"),
        )

        last_pair = (
            (last["overall_status"], last["alert_level"])
            if last
            else None
        )

        if current_pair != last_pair:
            connection.execute(
                """
                INSERT INTO alert_history (
                    created_at,
                    overall_status,
                    alert_level,
                    priority_district_count,
                    message,
                    priority_districts_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    unified.get("overall_status"),
                    unified.get("alert_level"),
                    unified.get("priority_district_count", 0),
                    unified.get("message"),
                    json.dumps(
                        unified.get("priority_districts", []),
                        default=str,
                    ),
                ),
            )
            connection.commit()

        connection.close()
    except Exception as exc:
        print(f"Alert history save failed: {exc}")


# ============================================================
# JSON-SAFE HELPERS
# ============================================================

def clean_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return value


def clean_records(records):
    return [
        {
            key: clean_value(value)
            for key, value in record.items()
        }
        for record in records
    ]


def clean_nested(value):
    if isinstance(value, dict):
        return {
            key: clean_nested(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [clean_nested(item) for item in value]

    return clean_value(value)


# ============================================================
# RISK HELPERS
# ============================================================

def probability_to_risk_level(probability_percent):
    if probability_percent < 35:
        return "Low"
    if probability_percent < 65:
        return "Moderate"
    if probability_percent < 85:
        return "High"
    return "Very High"


def spatial_score_to_risk_level(score):
    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Moderate"
    if score < 0.75:
        return "High"
    return "Very High"


def risk_level_rank(level):
    return {
        "Low": 0,
        "Moderate": 1,
        "High": 2,
        "Very High": 3,
    }.get(str(level), -1)


def get_date_column(dataframe):
    for candidate in [
        "date",
        "Date",
        "datetime",
        "timestamp",
        "forecast_date",
    ]:
        if candidate in dataframe.columns:
            return candidate
    return None


# ============================================================
# MODEL INPUT / PREDICTION
# ============================================================

def build_model_input_from_request(request):
    return pd.DataFrame(
        [[
            request.water_level_mean_m,
            request.water_level_max_m,
            request.water_level_min_m,
            request.water_level_change_m,
            request.temperature,
            request.humidity,
            request.wind_speed,
            request.pressure,
            request.water_level_3day_mean_m,
            request.water_level_7day_mean_m,
            request.month,
            request.day_of_year,
        ]],
        columns=MODEL_FEATURES,
    )


def generate_model_prediction(input_data):
    if flood_model is None:
        raise RuntimeError(
            "Flood prediction model is not available"
        )

    prediction = flood_model.predict(input_data)[0]
    probabilities = flood_model.predict_proba(input_data)[0]
    model_classes = list(flood_model.classes_)

    if 1 in model_classes:
        high_class_index = model_classes.index(1)
    else:
        high_class_index = len(model_classes) - 1

    flood_probability = float(
        probabilities[high_class_index]
    )

    flood_probability_percent = flood_probability * 100.0
    risk_level = probability_to_risk_level(
        flood_probability_percent
    )

    prediction_label = (
        "Low Flood Risk"
        if risk_level == "Low"
        else "Moderate Flood Risk"
        if risk_level == "Moderate"
        else "High Flood Risk"
        if risk_level == "High"
        else "Very High Flood Risk"
    )

    return {
        "prediction": int(prediction),
        "prediction_label": prediction_label,
        "flood_probability": round(
            flood_probability,
            4,
        ),
        "flood_probability_percent": round(
            flood_probability_percent,
            2,
        ),
        "risk_level": risk_level,
    }


def get_latest_ml_prediction():
    if flood_model is None or ml_training_data.empty:
        return None

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in ml_training_data.columns
    ]

    if missing_features:
        return None

    try:
        data = ml_training_data.copy()
        reference_date = None

        date_column = get_date_column(data)

        if date_column:
            data[date_column] = pd.to_datetime(
                data[date_column],
                errors="coerce",
            )

            data = (
                data
                .sort_values(date_column)
                .reset_index(drop=True)
            )

            if not data.empty:
                latest_date_value = data.iloc[-1][date_column]

                if pd.notna(latest_date_value):
                    reference_date = (
                        pd.Timestamp(latest_date_value)
                        .strftime("%Y-%m-%d")
                    )

        if data.empty:
            return None

        latest_row = data.iloc[-1]

        input_data = pd.DataFrame(
            [[
                latest_row[feature]
                for feature in MODEL_FEATURES
            ]],
            columns=MODEL_FEATURES,
        )

        prediction_result = generate_model_prediction(
            input_data
        )

        features_used = {}

        for feature in MODEL_FEATURES:
            value = latest_row[feature]

            if feature in ["month", "day_of_year"]:
                features_used[feature] = int(value)
            else:
                features_used[feature] = float(value)

        return {
            **prediction_result,
            "reference_date": reference_date,
            "features_used": features_used,
        }

    except Exception as exc:
        print(f"Latest ML prediction failed: {exc}")
        return None


# ============================================================
# EXPLAINABILITY / FLOOD REASON
# ============================================================

def build_model_explainability(input_data, prediction_result):
    if flood_model is None:
        return {
            "available": False,
            "message": "Flood prediction model is not available.",
        }

    if not hasattr(flood_model, "feature_importances_"):
        return {
            "available": False,
            "message": (
                "This model does not expose feature importance values."
            ),
        }

    try:
        importances = np.asarray(
            flood_model.feature_importances_,
            dtype=float,
        )

        feature_names = MODEL_FEATURES.copy()

        if hasattr(flood_model, "feature_names_in_"):
            model_names = list(flood_model.feature_names_in_)
            if len(model_names) == len(importances):
                feature_names = model_names

        explanation = []

        for index, importance in enumerate(importances):
            feature = (
                feature_names[index]
                if index < len(feature_names)
                else f"feature_{index}"
            )

            current_value = None

            if feature in input_data.columns:
                current_value = clean_value(
                    input_data.iloc[0][feature]
                )

            training_median = None
            relative_difference_percent = None

            if (
                not ml_training_data.empty
                and feature in ml_training_data.columns
            ):
                training_values = pd.to_numeric(
                    ml_training_data[feature],
                    errors="coerce",
                ).dropna()

                if not training_values.empty:
                    training_median = float(
                        training_values.median()
                    )

                    if (
                        abs(training_median) > 1e-9
                        and current_value is not None
                    ):
                        relative_difference_percent = (
                            (
                                (
                                    float(current_value)
                                    - training_median
                                )
                                / abs(training_median)
                            )
                            * 100.0
                        )

            explanation.append(
                {
                    "feature": feature,
                    "display_name": FEATURE_DISPLAY_NAMES.get(
                        feature,
                        feature,
                    ),
                    "importance": float(importance),
                    "importance_percent": float(
                        importance * 100.0
                    ),
                    "current_value": current_value,
                    "training_median": clean_value(
                        training_median
                    ),
                    "relative_difference_percent": clean_value(
                        relative_difference_percent
                    ),
                }
            )

        explanation.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return {
            "available": True,
            "explanation_type": (
                "Global Random Forest feature importance "
                "with current-input context"
            ),
            "prediction": prediction_result,
            "top_features": explanation[:8],
            "all_features": explanation,
            "interpretation_note": (
                "Feature importance indicates how strongly "
                "features contribute to the trained Random "
                "Forest overall. It is not a causal explanation "
                "of an individual flood event."
            ),
        }

    except Exception as exc:
        return {
            "available": False,
            "message": (
                "Explainability generation failed: "
                + str(exc)
            ),
        }


def build_flood_reason(explainability):
    if not explainability:
        return {
            "available": False,
            "headline": "Flood reason is unavailable.",
            "summary": "Model explanation is unavailable.",
            "factors": [],
        }

    top_features = explainability.get(
        "top_features",
        [],
    )

    if not top_features:
        return {
            "available": False,
            "headline": "Flood reason is unavailable.",
            "summary": (
                "The current model explanation does not contain "
                "enough information for a contributing-factor summary."
            ),
            "factors": [],
        }

    factors = []

    for item in top_features[:5]:
        factors.append(
            {
                "feature": item.get("feature"),
                "display_name": item.get("display_name"),
                "importance_percent": item.get(
                    "importance_percent"
                ),
                "current_value": item.get("current_value"),
                "training_median": item.get("training_median"),
                "relative_difference_percent": item.get(
                    "relative_difference_percent"
                ),
            }
        )

    names = [
        item.get("display_name")
        for item in factors[:3]
        if item.get("display_name")
    ]

    summary = (
        "The current model assessment is mainly associated "
        f"with {', '.join(names)}."
        if names
        else "The model identifies important input features."
    )

    return {
        "available": True,
        "headline": "Main model-based contributing factors",
        "summary": summary,
        "factors": factors,
        "note": (
            "These are model-based contributing factors, not "
            "confirmed physical causes of flooding."
        ),
    }


# ============================================================
# FORECAST SUMMARY / RISK FORECAST
# ============================================================

def load_forecast_frames():
    historical = (
        pd.read_csv(FORECAST_FILE)
        if os.path.exists(FORECAST_FILE)
        else pd.DataFrame()
    )

    future = (
        pd.read_csv(NEXT_FORECAST_FILE)
        if os.path.exists(NEXT_FORECAST_FILE)
        else pd.DataFrame()
    )

    return historical, future


def get_forecast_summary():
    try:
        current_forecast, current_next_forecast = load_forecast_frames()

        result = {
            "available": False,
            "trend": "Unavailable",
            "trend_change_m": None,
            "trend_change_percent": None,
            "latest_model_water_level_m": None,
            "next_forecast_water_level_m": None,
            "next_forecast_date": None,
        }

        if current_forecast.empty:
            return result

        result["available"] = True
        current_forecast = current_forecast.copy()

        if "date" in current_forecast.columns:
            current_forecast["date"] = pd.to_datetime(
                current_forecast["date"],
                errors="coerce",
            )

            current_forecast = (
                current_forecast
                .dropna(subset=["date"])
                .sort_values("date")
            )

        if "model_water_level_m" in current_forecast.columns:
            valid_model = pd.to_numeric(
                current_forecast["model_water_level_m"],
                errors="coerce",
            ).dropna()

            if not valid_model.empty:
                result["latest_model_water_level_m"] = float(
                    valid_model.iloc[-1]
                )

        elif "water_level_mean_m" in current_forecast.columns:
            valid_observed = pd.to_numeric(
                current_forecast["water_level_mean_m"],
                errors="coerce",
            ).dropna()

            if not valid_observed.empty:
                result["latest_model_water_level_m"] = float(
                    valid_observed.iloc[-1]
                )

        if not current_next_forecast.empty:
            next_data = current_next_forecast.copy()

            if "forecast_day" in next_data.columns:
                next_data["forecast_day"] = pd.to_numeric(
                    next_data["forecast_day"],
                    errors="coerce",
                )

                next_data = (
                    next_data
                    .dropna(subset=["forecast_day"])
                    .sort_values("forecast_day")
                )

            elif "forecast_date" in next_data.columns:
                next_data["forecast_date"] = pd.to_datetime(
                    next_data["forecast_date"],
                    errors="coerce",
                )

                next_data = (
                    next_data
                    .dropna(subset=["forecast_date"])
                    .sort_values("forecast_date")
                )

            if not next_data.empty:
                next_row = next_data.iloc[0]

                if "forecast_date" in next_data.columns:
                    next_date = pd.to_datetime(
                        next_row["forecast_date"],
                        errors="coerce",
                    )

                    if pd.notna(next_date):
                        result["next_forecast_date"] = (
                            pd.Timestamp(next_date)
                            .strftime("%Y-%m-%d")
                        )

                if "predicted_water_level_m" in next_data.columns:
                    next_level = pd.to_numeric(
                        next_row["predicted_water_level_m"],
                        errors="coerce",
                    )

                    if pd.notna(next_level):
                        result["next_forecast_water_level_m"] = float(
                            next_level
                        )

        latest_level = result["latest_model_water_level_m"]
        next_level = result["next_forecast_water_level_m"]

        if latest_level is not None and next_level is not None:
            change = next_level - latest_level
            result["trend_change_m"] = float(change)

            if abs(latest_level) > 0.0001:
                result["trend_change_percent"] = float(
                    (change / abs(latest_level)) * 100.0
                )

            if change > 2.0:
                result["trend"] = "Rising"
            elif change < -2.0:
                result["trend"] = "Falling"
            else:
                result["trend"] = "Stable"

        return result

    except Exception as exc:
        print(f"Forecast summary failed: {exc}")
        return {
            "available": False,
            "trend": "Unavailable",
            "trend_change_m": None,
            "trend_change_percent": None,
            "latest_model_water_level_m": None,
            "next_forecast_water_level_m": None,
            "next_forecast_date": None,
        }


def build_forecast_risk_rows():
    historical, future = load_forecast_frames()

    if future.empty:
        return []

    data = future.copy()

    if "forecast_day" in data.columns:
        data["forecast_day"] = pd.to_numeric(
            data["forecast_day"],
            errors="coerce",
        )
        data = data.sort_values("forecast_day")
    elif "forecast_date" in data.columns:
        data["forecast_date"] = pd.to_datetime(
            data["forecast_date"],
            errors="coerce",
        )
        data = data.sort_values("forecast_date")

    level_column = None

    for candidate in [
        "predicted_water_level_m",
        "predicted_water_level",
        "forecast",
        "predicted",
    ]:
        if candidate in data.columns:
            level_column = candidate
            break

    if level_column is None:
        return []

    levels = pd.to_numeric(
        data[level_column],
        errors="coerce",
    )

    reference_values = pd.Series(dtype=float)

    for candidate in [
        "model_water_level_m",
        "water_level_mean_m",
        "actual_water_level",
    ]:
        if candidate in historical.columns:
            reference_values = pd.to_numeric(
                historical[candidate],
                errors="coerce",
            ).dropna()

            if not reference_values.empty:
                break

    if reference_values.empty:
        return []

    quantiles = reference_values.quantile([0.50, 0.70, 0.90])
    rows = []

    for index, row in data.iterrows():
        level = levels.loc[index]

        if pd.isna(level):
            continue

        if level <= quantiles.loc[0.50]:
            risk = "Low"
        elif level <= quantiles.loc[0.70]:
            risk = "Moderate"
        elif level <= quantiles.loc[0.90]:
            risk = "High"
        else:
            risk = "Very High"

        if "forecast_date" in row.index:
            date_value = row.get("forecast_date")
        else:
            date_value = row.get("date", "")

        try:
            date_value = pd.Timestamp(date_value).strftime(
                "%Y-%m-%d"
            )
        except Exception:
            date_value = str(date_value)[:10]

        rows.append(
            {
                "forecast_day": clean_value(
                    row.get("forecast_day")
                ),
                "forecast_date": date_value,
                "predicted_water_level_m": float(level),
                "risk_level": risk,
                "risk_method": (
                    "Empirical water-level percentile proxy"
                ),
            }
        )

    return clean_records(rows)


# ============================================================
# UNIFIED ALERT ENGINE
# ============================================================

def calculate_unified_alert(
    district_df,
    ml_result,
    forecast_result,
):
    if district_df.empty:
        return {
            "overall_status": "Unavailable",
            "alert_level": "Unavailable",
            "priority_district_count": 0,
            "priority_districts": [],
            "decision_basis": [],
            "message": "GIS risk data is unavailable.",
        }

    risk_series = pd.to_numeric(
        district_df["spatial_risk_score"],
        errors="coerce",
    ).fillna(0.0)

    risk_levels = (
        district_df["risk_level"].astype(str)
        if "risk_level" in district_df.columns
        else risk_series.apply(spatial_score_to_risk_level)
    )

    very_high_count = int(
        risk_levels.str.lower().eq("very high").sum()
    )
    high_count = int(
        risk_levels.str.lower().eq("high").sum()
    )
    moderate_count = int(
        risk_levels.str.lower().eq("moderate").sum()
    )
    low_count = int(
        risk_levels.str.lower().eq("low").sum()
    )

    if very_high_count >= 5:
        overall_status = "Very High"
        alert_level = "Critical"
    elif very_high_count >= 1 or high_count >= 15:
        overall_status = "High"
        alert_level = "High"
    elif high_count >= 5 or moderate_count >= 10:
        overall_status = "Moderate"
        alert_level = "Moderate"
    else:
        overall_status = "Low"
        alert_level = "Low"

    decision_basis = [
        f"{very_high_count} districts are in the Very High category.",
        f"{high_count} districts are in the High category.",
    ]

    if ml_result is not None:
        ml_risk = ml_result["risk_level"]

        if ml_risk == "Very High":
            overall_status = "Very High"
            alert_level = "Critical"
            decision_basis.append(
                "Latest Random Forest prediction is Very High risk."
            )
        elif ml_risk == "High" and overall_status == "Low":
            overall_status = "High"
            alert_level = "High"
            decision_basis.append(
                "Latest Random Forest prediction is High risk."
            )

    if forecast_result.get("available", False):
        if forecast_result.get("trend") == "Rising":
            decision_basis.append(
                "Water-level forecast indicates a rising trend."
            )

            if overall_status == "Low":
                overall_status = "Moderate"
                alert_level = "Moderate"
            elif overall_status == "Moderate":
                overall_status = "High"
                alert_level = "High"

    priority = district_df.copy()
    priority["risk_numeric"] = pd.to_numeric(
        priority["spatial_risk_score"],
        errors="coerce",
    ).fillna(0.0)

    priority["risk_level_display"] = (
        priority["risk_level"].astype(str)
        if "risk_level" in priority.columns
        else priority["risk_numeric"].apply(
            spatial_score_to_risk_level
        )
    )

    priority = priority.sort_values(
        "risk_numeric",
        ascending=False,
    ).head(5)

    priority_records = []

    for record in priority.to_dict(orient="records"):
        priority_records.append(
            {
                "district_id": record.get("district_id"),
                "district": record.get("NAME_2"),
                "risk_score": record.get(
                    "spatial_risk_score"
                ),
                "risk_level": record.get(
                    "risk_level_display"
                ),
                "historical_flood_events": record.get(
                    "historical_flood_events"
                ),
                "elevation_m": record.get("elevation_m"),
                "distance_to_river_m": record.get(
                    "distance_to_river_m"
                ),
                "drainage_risk": record.get("drainage_risk"),
                "historical_risk": record.get("historical_risk"),
                "lulc_risk": record.get("lulc_risk"),
            }
        )

    priority_records = clean_records(priority_records)

    messages = {
        "Very High": (
            "Multiple indicators require priority monitoring "
            "across the highest-risk districts."
        ),
        "High": (
            "High spatial risk is present across a significant "
            "number of districts."
        ),
        "Moderate": (
            "Moderate spatial risk is present; continue "
            "monitoring trends and scenarios."
        ),
        "Low": (
            "Current project indicators show relatively lower "
            "spatial risk."
        ),
    }

    return {
        "overall_status": overall_status,
        "alert_level": alert_level,
        "message": messages.get(
            overall_status,
            "Decision-support status unavailable.",
        ),
        "priority_district_count": len(priority_records),
        "priority_districts": priority_records,
        "risk_distribution": {
            "very_high": very_high_count,
            "high": high_count,
            "moderate": moderate_count,
            "low": low_count,
        },
        "decision_basis": decision_basis,
    }


# ============================================================
# EMERGENCY ACTION ENGINE
# ============================================================

def build_emergency_action_plan():
    if district_data.empty:
        return {
            "available": False,
            "message": "GIS risk data is not available.",
        }

    ml_result = get_latest_ml_prediction()
    forecast_result = get_forecast_summary()

    unified = calculate_unified_alert(
        district_data,
        ml_result,
        forecast_result,
    )

    overall_status = unified["overall_status"]

    action_plans = {
        "Very High": [
            {
                "priority": "Immediate",
                "action": (
                    "Prioritize monitoring and coordination for "
                    "Very High-risk districts."
                ),
            },
            {
                "priority": "Immediate",
                "action": (
                    "Review evacuation routes, shelters, and "
                    "emergency communication readiness."
                ),
            },
            {
                "priority": "High",
                "action": (
                    "Inspect drainage and river-adjacent areas "
                    "in priority districts."
                ),
            },
            {
                "priority": "High",
                "action": (
                    "Increase the frequency of water-level and "
                    "weather monitoring."
                ),
            },
        ],
        "High": [
            {
                "priority": "High",
                "action": (
                    "Closely monitor highest-risk districts and "
                    "changing water-level conditions."
                ),
            },
            {
                "priority": "High",
                "action": (
                    "Review local drainage capacity and known "
                    "flood-sensitive areas."
                ),
            },
            {
                "priority": "Medium",
                "action": (
                    "Prepare emergency communication and "
                    "response resources."
                ),
            },
        ],
        "Moderate": [
            {
                "priority": "Medium",
                "action": (
                    "Continue monitoring rainfall and "
                    "water-level trends."
                ),
            },
            {
                "priority": "Medium",
                "action": (
                    "Review district risk indicators and "
                    "What-If scenarios for planning."
                ),
            },
        ],
        "Low": [
            {
                "priority": "Routine",
                "action": (
                    "Continue routine monitoring of available "
                    "environmental and spatial indicators."
                ),
            },
            {
                "priority": "Routine",
                "action": (
                    "Reassess risk when new observations or "
                    "scenario conditions become available."
                ),
            },
        ],
    }

    return {
        "available": True,
        "engine": (
            "Project-defined rule-based emergency action "
            "recommendation engine"
        ),
        "overall_status": overall_status,
        "alert_level": unified["alert_level"],
        "message": unified["message"],
        "priority_districts": unified["priority_districts"],
        "recommended_actions": action_plans.get(
            overall_status,
            [],
        ),
        "decision_basis": unified["decision_basis"],
        "ml_prediction": (
            ml_result if ml_result is not None else {"available": False}
        ),
        "forecast": forecast_result,
        "note": (
            "These are project-defined decision-support "
            "recommendations for academic use. They are not "
            "official emergency instructions."
        ),
    }


# ============================================================
# DISTRICT / VULNERABILITY / IMPACT HELPERS
# ============================================================

def calculate_vulnerability_score(record):
    weights = {
        "elevation_risk": 0.30,
        "drainage_risk": 0.25,
        "historical_risk": 0.25,
        "lulc_risk": 0.20,
    }

    score = 0.0
    total_weight = 0.0

    for field, weight in weights.items():
        if field in record and pd.notna(record[field]):
            value = pd.to_numeric(
                record[field],
                errors="coerce",
            )

            if pd.notna(value):
                score += float(np.clip(value, 0.0, 1.0)) * weight
                total_weight += weight

    if total_weight == 0:
        return None

    return score / total_weight


def build_area_profile(record):
    if record is None:
        return None

    result = {
        key: clean_value(value)
        for key, value in record.items()
    }

    vulnerability_score = calculate_vulnerability_score(result)
    risk_score = result.get("spatial_risk_score")

    if vulnerability_score is not None:
        impact_score = (
            0.65 * float(risk_score or 0.0)
            + 0.35 * vulnerability_score
        )
    else:
        impact_score = risk_score

    result["vulnerability_score"] = clean_value(
        vulnerability_score
    )
    result["impact_score"] = clean_value(
        impact_score
    )
    result["impact_risk_level"] = (
        spatial_score_to_risk_level(impact_score)
        if impact_score is not None
        else "Unavailable"
    )

    return result


def find_district(name):
    if district_data.empty or "NAME_2" not in district_data.columns:
        return None

    target = str(name).strip().lower()

    matches = district_data[
        district_data["NAME_2"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(target)
    ]

    if matches.empty:
        partial = district_data[
            district_data["NAME_2"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(target, na=False)
        ]

        if partial.empty:
            return None

        matches = partial

    return build_area_profile(
        matches.iloc[0].to_dict()
    )


def get_priority_area():
    if district_data.empty or "spatial_risk_score" not in district_data.columns:
        return None

    data = district_data.copy()
    data["risk_numeric"] = pd.to_numeric(
        data["spatial_risk_score"],
        errors="coerce",
    ).fillna(0.0)

    row = data.sort_values(
        "risk_numeric",
        ascending=False,
    ).iloc[0]

    return build_area_profile(
        row.to_dict()
    )


def get_district_neighbors(district_name):
    if (
        district_geometry_gdf is None
        or district_geometry_gdf.empty
        or "NAME_2" not in district_geometry_gdf.columns
    ):
        return []

    target = district_geometry_gdf[
        district_geometry_gdf["NAME_2"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(str(district_name).strip().lower())
    ]

    if target.empty:
        return []

    source_geometry = target.iloc[0].geometry
    neighbors = []

    for _, row in district_geometry_gdf.iterrows():
        other_name = str(row.get("NAME_2", ""))

        if other_name.strip().lower() == str(district_name).strip().lower():
            continue

        try:
            if source_geometry.touches(row.geometry):
                neighbors.append(other_name)
        except Exception:
            continue

    return sorted(set(neighbors))


def build_potentially_affected_districts(
    source_district=None,
    limit=10,
):
    if district_data.empty:
        return []

    source_profile = (
        find_district(source_district)
        if source_district
        else get_priority_area()
    )

    if source_profile is None:
        return []

    source_name = source_profile.get("NAME_2")
    neighbor_names = get_district_neighbors(source_name)

    if not neighbor_names:
        return []

    candidates = district_data[
        district_data["NAME_2"].astype(str).isin(neighbor_names)
    ].copy()

    if candidates.empty:
        return []

    source_elevation = pd.to_numeric(
        source_profile.get("elevation_m"),
        errors="coerce",
    )

    candidates["risk_numeric"] = pd.to_numeric(
        candidates["spatial_risk_score"],
        errors="coerce",
    ).fillna(0.0)

    candidates["elevation_numeric"] = pd.to_numeric(
        candidates.get("elevation_m"),
        errors="coerce",
    )

    candidates["potential_impact_score"] = (
        0.70 * candidates["risk_numeric"]
    )

    if pd.notna(source_elevation):
        candidates["potential_impact_score"] += (
            0.30
            * (
                candidates["elevation_numeric"]
                <= float(source_elevation)
            ).astype(float)
        )

    candidates["potential_impact_score"] = candidates[
        "potential_impact_score"
    ].clip(0.0, 1.0)

    candidates = candidates.sort_values(
        "potential_impact_score",
        ascending=False,
    ).head(limit)

    rows = []

    for record in candidates.to_dict(orient="records"):
        rows.append(
            {
                "district": record.get("NAME_2"),
                "source_priority_area": source_name,
                "current_risk_score": record.get(
                    "spatial_risk_score"
                ),
                "current_risk_level": record.get(
                    "risk_level"
                ),
                "elevation_m": record.get("elevation_m"),
                "potential_impact_score": record.get(
                    "potential_impact_score"
                ),
                "potential_impact_level": spatial_score_to_risk_level(
                    record.get("potential_impact_score", 0.0)
                ),
                "reason": (
                    "Adjacent district with existing spatial "
                    "risk and relative-elevation screening."
                ),
            }
        )

    return clean_records(rows)


# ============================================================
# WEATHER INTEGRATION
# ============================================================

def fetch_open_meteo_weather(
    latitude,
    longitude,
    forecast_days=7,
):
    params = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "surface_pressure,"
                "wind_speed_10m"
            ),
            "daily": (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_sum,"
                "wind_speed_10m_max"
            ),
            "forecast_days": forecast_days,
            "timezone": "auto",
        }
    )

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + params
    )

    request = Request(
        url,
        headers={
            "User-Agent": "FloodGuardAI/2.0 (academic-project)"
        },
    )

    with urlopen(request, timeout=12) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def build_daily_weather_rows(payload):
    daily = payload.get("daily", {})

    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    rainfall = daily.get("precipitation_sum", [])
    wind_max = daily.get("wind_speed_10m_max", [])

    rows = []

    for index, date_value in enumerate(dates):
        rows.append(
            {
                "date": date_value,
                "temperature_max_c": (
                    temp_max[index]
                    if index < len(temp_max)
                    else None
                ),
                "temperature_min_c": (
                    temp_min[index]
                    if index < len(temp_min)
                    else None
                ),
                "rainfall_mm": (
                    rainfall[index]
                    if index < len(rainfall)
                    else None
                ),
                "wind_max_kmh": (
                    wind_max[index]
                    if index < len(wind_max)
                    else None
                ),
            }
        )

    return clean_records(rows)


def get_district_weather(district_name):
    district = find_district(district_name)

    if district is None:
        raise HTTPException(
            status_code=404,
            detail="District not found.",
        )

    if district_geometry_gdf is None or district_geometry_gdf.empty:
        return {
            "available": False,
            "message": (
                "District geometry is unavailable for weather lookup."
            ),
        }

    rows = district_geometry_gdf[
        district_geometry_gdf["NAME_2"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(str(district_name).strip().lower())
    ]

    if rows.empty:
        return {
            "available": False,
            "message": "District geometry is unavailable for weather lookup.",
        }

    projected = rows.to_crs(epsg=3857)
    centroid = projected.iloc[0].geometry.centroid

    point = gpd.GeoSeries(
        [centroid],
        crs=3857,
    ).to_crs(epsg=4326).iloc[0]

    try:
        payload = fetch_open_meteo_weather(
            float(point.y),
            float(point.x),
            forecast_days=7,
        )

        hourly = payload.get("hourly", {})

        return {
            "available": True,
            "district": district.get("NAME_2"),
            "latitude": float(point.y),
            "longitude": float(point.x),
            "timezone": payload.get("timezone"),
            "daily": build_daily_weather_rows(payload),
            "hourly": {
                key: value
                for key, value in hourly.items()
                if key in [
                    "time",
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "surface_pressure",
                    "wind_speed_10m",
                ]
            },
            "source": "Open-Meteo forecast API",
            "note": (
                "Weather conditions are forecast data for the "
                "district centroid and are not themselves a "
                "flood prediction."
            ),
        }

    except Exception as exc:
        return {
            "available": False,
            "district": district.get("NAME_2"),
            "message": (
                "Weather service could not be reached: "
                + str(exc)
            ),
        }


# ============================================================
# HISTORICAL FLOOD LAYER
# ============================================================

def get_historical_flood_geojson():
    if not os.path.exists(HISTORICAL_FLOOD_FILE):
        return {
            "available": False,
            "message": "Historical flood shapefile not found.",
        }

    try:
        historical = gpd.read_file(
            HISTORICAL_FLOOD_FILE
        )

        if historical.crs is None:
            historical = historical.set_crs(epsg=4326)

        historical = historical.to_crs(epsg=4326)

        return {
            "available": True,
            "feature_count": len(historical),
            "geojson": json.loads(
                historical.to_json()
            ),
        }

    except Exception as exc:
        return {
            "available": False,
            "message": (
                "Historical flood layer could not be loaded: "
                + str(exc)
            ),
        }


# ============================================================
# GIS LAYER STATUS / HEATMAP POINTS
# ============================================================

def build_gis_layer_status():
    return {
        "district_boundaries": {
            "available": os.path.exists(BOUNDARY_FILE),
            "file": BOUNDARY_FILE,
        },
        "spatial_risk": {
            "available": os.path.exists(GIS_FILE),
            "file": GIS_FILE,
        },
        "spatial_risk_geopackage": {
            "available": os.path.exists(GIS_GPKG_FILE),
            "file": GIS_GPKG_FILE,
        },
        "river_network": {
            "available": os.path.exists(RIVER_FILE),
            "file": RIVER_FILE,
        },
        "historical_flood": {
            "available": os.path.exists(HISTORICAL_FLOOD_FILE),
            "file": HISTORICAL_FLOOD_FILE,
        },
        "weather": {
            "available": os.path.exists(WEATHER_FILE),
            "file": WEATHER_FILE,
        },
        "rainfall": {
            "available": (
                os.path.exists(RAINFALL_DISTRICT_FILE)
                or os.path.exists(RAINFALL_STATE_FILE)
            ),
            "district_file": RAINFALL_DISTRICT_FILE,
            "state_file": RAINFALL_STATE_FILE,
        },
    }


def build_risk_points():
    if district_data.empty or district_geometry_gdf is None:
        return []

    rows = []

    for _, district in district_data.iterrows():
        name = str(district.get("NAME_2", ""))

        match = district_geometry_gdf[
            district_geometry_gdf["NAME_2"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(name.strip().lower())
        ]

        if match.empty:
            continue

        projected = match.to_crs(epsg=3857)
        centroid = projected.iloc[0].geometry.centroid

        point = gpd.GeoSeries(
            [centroid],
            crs=3857,
        ).to_crs(epsg=4326).iloc[0]

        rows.append(
            {
                "district": name,
                "latitude": float(point.y),
                "longitude": float(point.x),
                "risk_score": district.get("spatial_risk_score"),
                "risk_level": district.get("risk_level"),
            }
        )

    return clean_records(rows)


# ============================================================
# REPORT SUMMARY
# ============================================================

def build_report_summary():
    priority = district_data.copy()

    if not priority.empty and "spatial_risk_score" in priority.columns:
        priority["risk_numeric"] = pd.to_numeric(
            priority["spatial_risk_score"],
            errors="coerce",
        ).fillna(0.0)
        priority = priority.sort_values(
            "risk_numeric",
            ascending=False,
        ).head(5)

    ml_result = get_latest_ml_prediction()
    forecast_result = get_forecast_summary()

    unified = calculate_unified_alert(
        district_data,
        ml_result,
        forecast_result,
    )

    priority_records = [
        build_area_profile(record)
        for record in priority.to_dict(orient="records")
    ]

    summary = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "project": "Flood Guard AI",
        "district_count": (
            int(len(district_data))
            if not district_data.empty
            else 0
        ),
        "average_risk_score": (
            float(
                pd.to_numeric(
                    district_data["spatial_risk_score"],
                    errors="coerce",
                ).mean()
            )
            if (
                not district_data.empty
                and "spatial_risk_score" in district_data.columns
            )
            else None
        ),
        "maximum_risk_score": (
            float(
                pd.to_numeric(
                    district_data["spatial_risk_score"],
                    errors="coerce",
                ).max()
            )
            if (
                not district_data.empty
                and "spatial_risk_score" in district_data.columns
            )
            else None
        ),
        "overall_status": unified.get("overall_status"),
        "alert_level": unified.get("alert_level"),
        "ml_prediction": ml_result,
        "forecast": forecast_result,
        "priority_districts": priority_records,
    }

    return clean_nested(summary)


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "project": "Flood Guard AI",
        "status": "running",
        "version": "2.0.0",
        "message": "Flood Guard AI Backend API is working",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "database": os.path.exists(DATABASE_FILE),
        "model": flood_model is not None,
        "district_data": not district_data.empty,
        "forecast_data": (
            not forecast_data.empty
            or not next_forecast_data.empty
        ),
    }


# ============================================================
# PREDICTION APIS
# ============================================================

@app.post("/api/predict")
def predict_flood(request: FloodPredictionRequest):
    if flood_model is None:
        raise HTTPException(
            status_code=503,
            detail="Flood prediction model is not available",
        )

    try:
        input_data = build_model_input_from_request(request)
        prediction_result = generate_model_prediction(input_data)

        features_used = {
            feature: clean_value(
                input_data.iloc[0][feature]
            )
            for feature in MODEL_FEATURES
        }

        response = {
            "status": "success",
            **prediction_result,
            "features_used": features_used,
            "prediction_source": (
                "Random Forest flood-event classifier"
            ),
        }

        save_prediction_history(
            response,
            source="custom_prediction",
        )

        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Flood prediction failed: " + str(exc),
        )


@app.get("/api/predict/latest")
def predict_latest_available():
    result = get_latest_ml_prediction()

    if result is None:
        if flood_model is None:
            raise HTTPException(
                status_code=503,
                detail="Flood prediction model is not available",
            )

        if ml_training_data.empty:
            raise HTTPException(
                status_code=404,
                detail="ML training data is not available",
            )

        raise HTTPException(
            status_code=500,
            detail="Latest flood prediction could not be generated",
        )

    response = {
        "status": "success",
        **result,
        "flood_probability": round(
            result["flood_probability_percent"] / 100.0,
            4,
        ),
        "source": "ml_training_data.csv",
        "prediction_source": "Latest available project data",
    }

    save_prediction_history(
        response,
        source="latest_available",
    )

    return response


# ============================================================
# EXPLAINABILITY APIS
# ============================================================

@app.get("/api/explain/latest")
def explain_latest_prediction():
    latest_result = get_latest_ml_prediction()

    if latest_result is None:
        raise HTTPException(
            status_code=404,
            detail="Latest flood prediction is not available",
        )

    input_data = pd.DataFrame(
        [[
            latest_result["features_used"][feature]
            for feature in MODEL_FEATURES
        ]],
        columns=MODEL_FEATURES,
    )

    explanation = build_model_explainability(
        input_data,
        {
            "prediction": latest_result["prediction"],
            "prediction_label": latest_result["prediction_label"],
            "flood_probability_percent": latest_result[
                "flood_probability_percent"
            ],
            "risk_level": latest_result["risk_level"],
        },
    )

    return {
        "status": "success",
        "reference_date": latest_result["reference_date"],
        "features_used": latest_result["features_used"],
        "flood_reason": build_flood_reason(explanation),
        **explanation,
    }


@app.post("/api/explain")
def explain_prediction(request: FloodPredictionRequest):
    if flood_model is None:
        raise HTTPException(
            status_code=503,
            detail="Flood prediction model is not available",
        )

    try:
        input_data = build_model_input_from_request(request)
        prediction_result = generate_model_prediction(input_data)
        explanation = build_model_explainability(
            input_data,
            prediction_result,
        )

        return {
            "status": "success",
            "features_used": {
                feature: clean_value(
                    input_data.iloc[0][feature]
                )
                for feature in MODEL_FEATURES
            },
            "flood_reason": build_flood_reason(explanation),
            **explanation,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="AI explainability failed: " + str(exc),
        )


# ============================================================
# DECISION SUPPORT / ALERTS
# ============================================================

@app.get("/api/decision-support")
def get_decision_support():
    if district_data.empty:
        raise HTTPException(
            status_code=404,
            detail="GIS spatial risk data is not available",
        )

    if "spatial_risk_score" not in district_data.columns:
        raise HTTPException(
            status_code=500,
            detail="spatial_risk_score is missing from GIS risk data",
        )

    try:
        data = district_data.copy()
        data["spatial_risk_score"] = pd.to_numeric(
            data["spatial_risk_score"],
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0)

        if "risk_level" not in data.columns:
            data["risk_level"] = data[
                "spatial_risk_score"
            ].apply(spatial_score_to_risk_level)

        ml_result = get_latest_ml_prediction()
        forecast_result = get_forecast_summary()

        unified = calculate_unified_alert(
            data,
            ml_result,
            forecast_result,
        )

        save_alert_history_if_changed(unified)

        priority = get_priority_area()
        affected = build_potentially_affected_districts(
            priority.get("NAME_2") if priority else None
        )

        return {
            "status": "success",
            "engine": {
                "name": (
                    "Flood Guard AI Unified Risk & Alert Engine"
                ),
                "type": (
                    "Project-defined decision-support rule engine"
                ),
            },
            "overall_status": unified["overall_status"],
            "alert_level": unified["alert_level"],
            "message": unified["message"],
            "priority_district_count": unified[
                "priority_district_count"
            ],
            "priority_districts": unified[
                "priority_districts"
            ],
            "decision_basis": unified["decision_basis"],
            "priority_area": priority,
            "potentially_affected_districts": affected,
            "gis": {
                "district_count": int(len(data)),
                "average_risk_score": round(
                    float(data["spatial_risk_score"].mean()),
                    4,
                ),
                "minimum_risk_score": round(
                    float(data["spatial_risk_score"].min()),
                    4,
                ),
                "maximum_risk_score": round(
                    float(data["spatial_risk_score"].max()),
                    4,
                ),
                "risk_distribution": unified[
                    "risk_distribution"
                ],
            },
            "ml_prediction": (
                ml_result
                if ml_result is not None
                else {"available": False}
            ),
            "forecast": forecast_result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unified decision-support engine failed: "
                + str(exc)
            ),
        )


@app.get("/api/alerts")
def get_alerts():
    decision = get_decision_support()

    return {
        "status": "success",
        "overall_status": decision.get("overall_status"),
        "alert_level": decision.get("alert_level"),
        "message": decision.get("message"),
        "priority_districts": decision.get(
            "priority_districts",
            [],
        ),
        "decision_basis": decision.get(
            "decision_basis",
            [],
        ),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "note": (
            "Alerts are project-defined decision-support "
            "levels, not official public warnings."
        ),
    }


@app.get("/api/emergency-actions")
def get_emergency_actions():
    try:
        result = build_emergency_action_plan()
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Emergency action engine failed: "
                + str(exc)
            ),
        )


@app.get("/api/situation-summary")
def get_situation_summary():
    decision = get_decision_support()
    priority = decision.get("priority_area")
    forecast = decision.get("forecast", {})
    affected = decision.get(
        "potentially_affected_districts",
        [],
    )

    parts = [
        (
            "Current project-wide risk status is "
            f"{decision.get('overall_status', 'Unavailable')} "
            f"with an alert level of "
            f"{decision.get('alert_level', 'Unavailable')}."
        )
    ]

    if priority:
        parts.append(
            (
                f"The highest calculated spatial-risk district is "
                f"{priority.get('NAME_2')} with a spatial risk "
                f"score of {float(priority.get('spatial_risk_score', 0)):.3f}."
            )
        )

    if forecast.get("trend") not in [None, "Unavailable"]:
        parts.append(
            f"The water-level forecast trend is {forecast.get('trend')}."
        )

    if affected:
        names = [
            item.get("district")
            for item in affected[:3]
            if item.get("district")
        ]

        if names:
            parts.append(
                "Potentially affected adjacent districts identified "
                f"by spatial screening include {', '.join(names)}."
            )

    return {
        "status": "success",
        "summary": " ".join(parts),
        "basis": {
            "overall_status": decision.get("overall_status"),
            "priority_area": priority,
            "forecast": forecast,
            "potentially_affected_districts": affected,
        },
        "note": (
            "This is an automated summary of project outputs. "
            "It is not an official flood warning."
        ),
    }


# ============================================================
# DISTRICT SEARCH / PROFILE / COMPARISON / PRIORITY
# ============================================================

@app.get("/api/districts")
def get_districts():
    if district_data.empty:
        return {
            "status": "error",
            "message": "GIS district data not found",
        }

    return {
        "status": "success",
        "district_count": len(district_data),
        "districts": clean_records(
            district_data.to_dict(orient="records")
        ),
    }


@app.get("/api/districts/search")
def search_districts(
    q: str = Query(..., min_length=1)
):
    if district_data.empty:
        raise HTTPException(
            status_code=404,
            detail="GIS district data is not available.",
        )

    query = q.strip().lower()
    matches = district_data[
        district_data["NAME_2"]
        .astype(str)
        .str.lower()
        .str.contains(query, na=False)
    ]

    results = [
        build_area_profile(record)
        for record in matches.to_dict(orient="records")
    ]

    return {
        "status": "success",
        "query": q,
        "result_count": len(results),
        "districts": clean_records(results),
    }


@app.get("/api/districts/profile/{district_name}")
def district_profile(district_name: str):
    profile = find_district(district_name)

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="District not found.",
        )

    return {
        "status": "success",
        "district": profile,
        "potentially_affected_adjacent_districts": (
            build_potentially_affected_districts(
                district_name
            )
        ),
    }


@app.get("/api/districts/compare")
def compare_districts(
    first: str = Query(..., min_length=1),
    second: str = Query(..., min_length=1),
):
    first_profile = find_district(first)
    second_profile = find_district(second)

    if first_profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"District not found: {first}",
        )

    if second_profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"District not found: {second}",
        )

    comparison_fields = [
        "spatial_risk_score",
        "risk_level",
        "historical_flood_events",
        "elevation_m",
        "distance_to_river_m",
        "elevation_risk",
        "drainage_risk",
        "historical_risk",
        "lulc_risk",
        "vulnerability_score",
        "impact_score",
        "impact_risk_level",
    ]

    comparison = [
        {
            "metric": field,
            "first_district": first_profile.get(field),
            "second_district": second_profile.get(field),
        }
        for field in comparison_fields
    ]

    return {
        "status": "success",
        "first_district": first_profile,
        "second_district": second_profile,
        "comparison": clean_records(comparison),
    }


@app.get("/api/priority-area")
def priority_area():
    priority = get_priority_area()

    if priority is None:
        raise HTTPException(
            status_code=404,
            detail="Priority area could not be determined.",
        )

    latest = get_latest_ml_prediction()
    reason = {"available": False}

    if latest:
        input_data = pd.DataFrame(
            [[
                latest["features_used"][feature]
                for feature in MODEL_FEATURES
            ]],
            columns=MODEL_FEATURES,
        )

        explanation = build_model_explainability(
            input_data,
            latest,
        )
        reason = build_flood_reason(explanation)

    return {
        "status": "success",
        "priority_area": priority,
        "reason": reason,
        "potentially_affected_districts": (
            build_potentially_affected_districts(
                priority.get("NAME_2")
            )
        ),
        "note": (
            "Priority area means the district with the highest "
            "calculated spatial-risk score. It does not identify "
            "an exact physical flood origin."
        ),
    }


# ============================================================
# RISK / HEATMAP / PROPAGATION
# ============================================================

@app.get("/api/risk")
def get_risk():
    if district_data.empty:
        return {
            "status": "error",
            "message": "GIS risk data not found",
        }

    risk_columns = [
        "district_id",
        "NAME_1",
        "NAME_2",
        "historical_flood_events",
        "distance_to_river_m",
        "elevation_m",
        "lulc_class",
        "elevation_risk",
        "drainage_risk",
        "historical_risk",
        "lulc_risk",
        "spatial_risk_score",
        "risk_level",
    ]

    available_columns = [
        column
        for column in risk_columns
        if column in district_data.columns
    ]

    risk_data = district_data[
        available_columns
    ].copy()

    risk_distribution = (
        risk_data["risk_level"].value_counts().to_dict()
        if "risk_level" in risk_data.columns
        else {}
    )

    numeric_risk = (
        pd.to_numeric(
            risk_data["spatial_risk_score"],
            errors="coerce",
        )
        if "spatial_risk_score" in risk_data.columns
        else pd.Series(dtype=float)
    )

    return {
        "status": "success",
        "district_count": len(risk_data),
        "average_risk_score": (
            float(numeric_risk.mean())
            if not numeric_risk.empty
            else None
        ),
        "minimum_risk_score": (
            float(numeric_risk.min())
            if not numeric_risk.empty
            else None
        ),
        "maximum_risk_score": (
            float(numeric_risk.max())
            if not numeric_risk.empty
            else None
        ),
        "risk_distribution": {
            str(key): int(value)
            for key, value in risk_distribution.items()
        },
        "priority_area": get_priority_area(),
        "districts": clean_records(
            risk_data.to_dict(orient="records")
        ),
    }


@app.get("/api/risk/heatmap")
def get_risk_heatmap():
    return {
        "status": "success",
        "available": True,
        "points": build_risk_points(),
        "method": (
            "District-centroid risk points generated from the "
            "existing spatial risk and boundary layers."
        ),
    }


@app.get("/api/risk/propagation")
def get_risk_propagation(
    source: str | None = None,
):
    priority = (
        find_district(source)
        if source
        else get_priority_area()
    )

    if priority is None:
        raise HTTPException(
            status_code=404,
            detail="Source district could not be determined.",
        )

    source_name = priority.get("NAME_2")

    return {
        "status": "success",
        "source_district": source_name,
        "source_risk": priority,
        "potentially_affected_districts": (
            build_potentially_affected_districts(
                source_name
            )
        ),
        "method": (
            "Administrative adjacency, existing risk and "
            "relative-elevation screening. This is not a "
            "hydraulic flood-routing simulation."
        ),
    }


# ============================================================
# MAP APIS
# ============================================================

@app.get("/api/map")
def get_map():
    if not os.path.exists(BOUNDARY_FILE):
        return {
            "status": "error",
            "message": "Boundary GeoPackage not found",
        }

    if district_data.empty:
        return {
            "status": "error",
            "message": "Risk data not found",
        }

    try:
        districts = gpd.read_file(
            BOUNDARY_FILE,
            layer="ADM_ADM_2",
        )

        if "NAME_1" in districts.columns:
            districts = districts[
                districts["NAME_1"].astype(str).str.lower().eq("odisha")
            ].copy()
        elif "STATE" in districts.columns:
            districts = districts[
                districts["STATE"].astype(str).str.lower().eq("odisha")
            ].copy()

        risk_columns = [
            "NAME_2",
            "district_id",
            "spatial_risk_score",
            "risk_level",
            "historical_flood_events",
            "distance_to_river_m",
            "elevation_m",
            "elevation_risk",
            "drainage_risk",
            "historical_risk",
            "lulc_risk",
        ]

        available_columns = [
            column
            for column in risk_columns
            if column in district_data.columns
        ]

        risk_data = district_data[
            available_columns
        ].copy()

        districts = districts.merge(
            risk_data,
            on="NAME_2",
            how="left",
        )

        if districts.crs is not None:
            districts = districts.to_crs(epsg=4326)

        return {
            "status": "success",
            "district_count": len(districts),
            "geojson": json.loads(districts.to_json()),
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


@app.get("/api/map/layers")
def get_map_layers():
    return {
        "status": "success",
        "layers": build_gis_layer_status(),
        "heatmap": {
            "available": True,
            "endpoint": "/api/risk/heatmap",
        },
        "historical_flood_locations": {
            "available": os.path.exists(
                HISTORICAL_FLOOD_FILE
            ),
            "endpoint": "/api/map/historical-floods",
        },
        "priority_area": {
            "available": get_priority_area() is not None,
            "endpoint": "/api/priority-area",
        },
    }


@app.get("/api/map/historical-floods")
def get_historical_floods():
    return {
        "status": "success",
        **get_historical_flood_geojson(),
    }


# ============================================================
# AFFECTED DISTRICTS / CITIES
# ============================================================

@app.get("/api/affected-districts")
def get_affected_districts(
    source: str | None = None,
):
    source_profile = (
        find_district(source)
        if source
        else get_priority_area()
    )

    if source_profile is None:
        raise HTTPException(
            status_code=404,
            detail="Source district could not be determined.",
        )

    return {
        "status": "success",
        "source_district": source_profile.get("NAME_2"),
        "affected_districts": (
            build_potentially_affected_districts(
                source_profile.get("NAME_2")
            )
        ),
        "method": (
            "Administrative adjacency + existing spatial risk + "
            "relative-elevation screening. This is a planning "
            "screening, not a hydraulic propagation model."
        ),
    }



@app.get("/api/affected-cities")
def get_affected_cities(
    source: str | None = None,
):
    """
    Return potentially affected urban places using the existing
    district-level Flood Guard AI spatial risk score.

    Census data supplies urban-place names and 2011 population.
    No city coordinates or city-level hydraulic inundation are
    inferred.
    """

    if not os.path.exists(CITY_FILE):
        return {
            "status": "success",
            "available": False,
            "source_district": source,
            "cities": [],
            "message": "Census city dataset not found.",
        }

    if district_data.empty:
        return {
            "status": "success",
            "available": False,
            "source_district": source,
            "cities": [],
            "message": "District GIS risk data is unavailable.",
        }

    try:
        cities = pd.read_csv(CITY_FILE)

        required_columns = [
            "city_name",
            "district",
            "district_code",
            "town_code",
            "population_2011",
            "urban_rural",
        ]

        missing = [
            column
            for column in required_columns
            if column not in cities.columns
        ]

        if missing:
            return {
                "status": "success",
                "available": False,
                "source_district": source,
                "cities": [],
                "message": (
                    "City dataset is missing required columns: "
                    + ", ".join(missing)
                ),
            }

        risk_lookup = {}

        for record in district_data.to_dict(
            orient="records"
        ):
            district_name = str(
                record.get("NAME_2", "")
            ).strip()

            if not district_name:
                continue

            try:
                risk_score = float(
                    record.get(
                        "spatial_risk_score",
                        0.0,
                    )
                    or 0.0
                )
            except (TypeError, ValueError):
                risk_score = 0.0

            risk_score = max(
                0.0,
                min(1.0, risk_score),
            )

            risk_level = str(
                record.get(
                    "risk_level",
                    spatial_score_to_risk_level(
                        risk_score
                    ),
                )
            ).strip()

            risk_lookup[
                district_name.lower()
            ] = {
                "district": district_name,
                "risk_score": risk_score,
                "risk_level": risk_level,
            }

        cities["district_key"] = (
            cities["district"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        cities["district_risk_score"] = (
            cities["district_key"]
            .map(
                lambda key: (
                    risk_lookup
                    .get(key, {})
                    .get("risk_score")
                )
            )
        )

        cities["district_risk_level"] = (
            cities["district_key"]
            .map(
                lambda key: (
                    risk_lookup
                    .get(key, {})
                    .get("risk_level")
                )
            )
        )

        if source:

            source_record = find_district(source)

            if source_record is None:
                return {
                    "status": "success",
                    "available": False,
                    "source_district": source,
                    "cities": [],
                    "message": (
                        f"District not found: {source}"
                    ),
                }

            source_name = str(
                source_record.get(
                    "NAME_2",
                    "",
                )
            ).strip()

            filtered = cities[
                cities["district_key"]
                == source_name.lower()
            ].copy()

            selection_method = (
                "All Census urban places within selected district"
            )

        else:

            source_name = None

            filtered = cities[
                cities["district_risk_level"]
                .astype(str)
                .str.lower()
                .isin(
                    {
                        "high",
                        "very high",
                    }
                )
            ].copy()

            selection_method = (
                "Census urban places in High / Very High-risk districts"
            )

        records = []

        for record in filtered.to_dict(
            orient="records"
        ):

            risk_score = record.get(
                "district_risk_score"
            )

            if risk_score is None:
                continue

            try:
                population_2011 = int(
                    float(
                        record.get(
                            "population_2011",
                            0,
                        )
                        or 0
                    )
                )
            except (TypeError, ValueError):
                population_2011 = 0

            records.append(
                {
                    "city_name": record.get(
                        "city_name"
                    ),
                    "district": record.get(
                        "district"
                    ),
                    "district_code": record.get(
                        "district_code"
                    ),
                    "town_code": record.get(
                        "town_code"
                    ),
                    "population_2011": population_2011,
                    "population_year": 2011,
                    "urban_rural": record.get(
                        "urban_rural"
                    ),
                    "district_risk_score": round(
                        float(risk_score),
                        4,
                    ),
                    "district_risk_level": record.get(
                        "district_risk_level"
                    ),
                    "screening_basis": (
                        "District-level risk linkage; "
                        "not city-level inundation modelling"
                    ),
                }
            )

        records.sort(
            key=lambda item: (
                -float(
                    item.get(
                        "district_risk_score",
                        0.0,
                    )
                    or 0.0
                ),
                -int(
                    item.get(
                        "population_2011",
                        0,
                    )
                    or 0
                ),
            )
        )

        total_urban_population_2011 = sum(
            int(
                item.get(
                    "population_2011",
                    0,
                )
                or 0
            )
            for item in records
        )

        return {
            "status": "success",
            "available": True,
            "source_district": source_name,
            "selection_method": selection_method,
            "city_count": len(records),
            "total_urban_population_2011": (
                total_urban_population_2011
            ),
            "cities": clean_records(records),
            "data_year": 2011,
            "note": (
                "Urban-place population is Census 2011. "
                "Potential impact is screened from the "
                "linked district's existing GIS risk score. "
                "City coordinates and hydraulic inundation "
                "are not inferred."
            ),
        }

    except Exception as exc:
        return {
            "status": "success",
            "available": False,
            "source_district": source,
            "cities": [],
            "message": (
                "Affected city analysis failed: "
                + str(exc)
            ),
        }



@app.get("/api/population-impact")
def get_population_impact(source: str | None = None):
    """Return district-level WorldPop population exposure screening."""

    if not os.path.exists(POPULATION_DISTRICT_FILE):
        return {
            "status": "success",
            "available": False,
            "source_district": source,
            "districts": [],
            "message": "WorldPop district population dataset not found.",
        }

    if district_data.empty:
        return {
            "status": "success",
            "available": False,
            "source_district": source,
            "districts": [],
            "message": "District GIS risk data is unavailable.",
        }

    try:
        population_df = pd.read_csv(POPULATION_DISTRICT_FILE)

        required_columns = [
            "district",
            "population_worldpop_2020",
        ]
        missing = [
            col for col in required_columns
            if col not in population_df.columns
        ]

        if missing:
            return {
                "status": "success",
                "available": False,
                "source_district": source,
                "districts": [],
                "message": (
                    "WorldPop dataset is missing required columns: "
                    + ", ".join(missing)
                ),
            }

        population_lookup = {}
        for record in population_df.to_dict(orient="records"):
            district_name = str(record.get("district", "")).strip()
            if not district_name:
                continue
            try:
                population_lookup[district_name.lower()] = float(
                    record.get("population_worldpop_2020", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                population_lookup[district_name.lower()] = 0.0

        selected_source_name = None
        if source:
            source_record = find_district(source)
            if source_record is None:
                return {
                    "status": "success",
                    "available": False,
                    "source_district": source,
                    "districts": [],
                    "message": f"District not found: {source}",
                }
            selected_source_name = str(
                source_record.get("NAME_2", "")
            ).strip()

        rows = []
        for record in district_data.to_dict(orient="records"):
            district_name = str(record.get("NAME_2", "")).strip()
            if not district_name:
                continue

            if (
                selected_source_name
                and district_name.lower() != selected_source_name.lower()
            ):
                continue

            population = population_lookup.get(district_name.lower())
            if population is None:
                continue

            try:
                risk_score = float(
                    record.get("spatial_risk_score", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                risk_score = 0.0

            risk_score = max(0.0, min(1.0, risk_score))
            estimated_at_risk = population * risk_score
            risk_level = str(
                record.get(
                    "risk_level",
                    spatial_score_to_risk_level(risk_score),
                )
            ).strip()

            rows.append(
                {
                    "district": district_name,
                    "population_worldpop_2020": round(population, 2),
                    "population_year": 2020,
                    "spatial_risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "estimated_population_at_risk": round(
                        estimated_at_risk, 2
                    ),
                }
            )

        rows.sort(
            key=lambda item: -float(
                item["estimated_population_at_risk"]
            )
        )

        total_population = sum(
            float(item["population_worldpop_2020"])
            for item in rows
        )
        total_estimated_at_risk = sum(
            float(item["estimated_population_at_risk"])
            for item in rows
        )

        return {
            "status": "success",
            "available": True,
            "source_district": selected_source_name,
            "district_count": len(rows),
            "total_population_worldpop_2020": round(
                total_population, 2
            ),
            "estimated_population_at_risk": round(
                total_estimated_at_risk, 2
            ),
            "population_year": 2020,
            "method": (
                "WorldPop 2020 district population multiplied by "
                "existing spatial risk score."
            ),
            "interpretation": (
                "Risk-weighted population exposure indicator; not "
                "confirmed flood inundation or evacuation population."
            ),
            "districts": rows,
        }

    except Exception as exc:
        return {
            "status": "success",
            "available": False,
            "source_district": source,
            "districts": [],
            "message": "Population exposure analysis failed: " + str(exc),
        }


# ============================================================
# FORECAST APIS
# ============================================================

@app.get("/api/forecast")
def get_forecast():
    try:
        current_forecast_data, current_next_forecast_data = (
            load_forecast_frames()
        )

        if (
            current_forecast_data.empty
            and current_next_forecast_data.empty
        ):
            return {
                "status": "error",
                "message": "Forecast data not found",
            }

        return {
            "status": "success",
            "historical_forecast_count": len(
                current_forecast_data
            ),
            "historical_forecast": clean_records(
                current_forecast_data.to_dict(orient="records")
            ),
            "next_forecast": clean_records(
                current_next_forecast_data.to_dict(orient="records")
            ),
            "forecast_risk": build_forecast_risk_rows(),
            "forecast_summary": get_forecast_summary(),
            "risk_method_note": (
                "Forecast risk is an empirical water-level "
                "percentile proxy, not a calibrated probability."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to load forecast data: " + str(exc),
        )


@app.get("/api/forecast/risk")
def get_forecast_risk():
    return {
        "status": "success",
        "forecast_risk": build_forecast_risk_rows(),
    }


@app.get("/api/forecast/timeline")
def get_forecast_timeline():
    forecast_risk = build_forecast_risk_rows()

    timeline = []

    for item in forecast_risk:
        day = item.get("forecast_day")

        timeline.append(
            {
                "period": (
                    "Current"
                    if day in [0, "0"]
                    else f"Day {day}"
                ),
                "forecast_day": day,
                "date": item.get("forecast_date"),
                "water_level_m": item.get(
                    "predicted_water_level_m"
                ),
                "risk_level": item.get("risk_level"),
            }
        )

    return {
        "status": "success",
        "timeline": timeline,
        "periods": [
            "Current",
            "1 Day",
            "3 Day",
            "7 Day",
        ],
    }



# ============================================================
# RAINFALL ANOMALY
# ============================================================

def prepare_rainfall_long() -> pd.DataFrame:
    """Load Odisha district-wise IMD daily rainfall into standard long format."""
    rainfall_file = BASE_DIR / "Datasets" / "data" / "rainfall" / "rainfall_districtwise_daily_imd.csv"

    if not rainfall_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(rainfall_file)
    except Exception:
        return pd.DataFrame()

    required = ["State", "District", "Date", "Daily Actual"]

    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    state = df["State"].astype(str).str.strip().str.upper()

    df = df[state.isin(["ODISHA", "ORISSA"])].copy()

    if df.empty:
        return pd.DataFrame()

    out = df[["Date", "District", "Daily Actual"]].copy()

    out.columns = ["date", "district", "rainfall_mm"]

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce"
    )

    out["rainfall_mm"] = pd.to_numeric(
        out["rainfall_mm"],
        errors="coerce"
    )

    out["district"] = (
        out["district"]
        .astype(str)
        .str.strip()
    )

    out = out.dropna(
        subset=["date", "district", "rainfall_mm"]
    )

    out = out[
        out["district"].ne("")
    ].copy()

    return (
        out
        .sort_values(["date", "district"])
        .reset_index(drop=True)
    )

def build_rainfall_anomaly() -> dict:
    data = prepare_rainfall_long()
    if data.empty:
        return {
            "available": False,
            "message": "District-wise rainfall data could not be interpreted.",
        }

    latest_date = data["date"].max()
    latest = data[data["date"] == latest_date].copy()
    rainfall_source = "IMD district dataset (latest available day)"

    # Prefer today's live rainfall (Open-Meteo, via the live ML module) when it is available.
    try:
        try:
            from backend import live_ml_api as _lm
        except ImportError:
            import live_ml_api as _lm
        _live = _lm.get_live_ml()
        _today_rain = {}
        for _name, _row in (_live.get("districts") or {}).items():
            for _d in _row.get("series", []):
                if _d.get("is_today"):
                    _today_rain[str(_name).strip().lower()] = float(_d.get("rain_mm") or 0.0)
        if _today_rain:
            _districts = sorted(data["district"].dropna().unique())
            _rows = [
                {"district": _d, "rainfall_mm": _today_rain[str(_d).strip().lower()]}
                for _d in _districts
                if str(_d).strip().lower() in _today_rain
            ]
            if _rows:
                latest = pd.DataFrame(_rows)
                latest_date = pd.Timestamp(_live.get("today"))
                rainfall_source = "Live Open-Meteo rainfall today (district centre)"
    except Exception:
        pass

    baseline = (
        data.groupby("district")["rainfall_mm"]
        .agg(median="median", mean="mean", std="std")
        .reset_index()
    )

    latest = latest.merge(baseline, on="district", how="left")
    latest["anomaly_mm"] = latest["rainfall_mm"] - latest["median"]
    latest["anomaly_percent"] = np.where(
        latest["median"].abs() > 0.001,
        latest["anomaly_mm"] / latest["median"].abs() * 100,
        np.nan,
    )
    latest["z_score"] = np.where(
        latest["std"].fillna(0).abs() > 0.001,
        latest["anomaly_mm"] / latest["std"].abs(),
        0,
    )

    def label(z_score, rain_mm):
        # z-score decides how unusual the rain is; IMD heavy-rain levels keep small totals from looking extreme
        if rain_mm >= 115.6 or (z_score >= 3 and rain_mm >= 64.5):
            return "Extreme High"
        if rain_mm >= 64.5 or z_score >= 2:
            return "High"
        if z_score >= 1:
            return "Above Normal"
        if z_score <= -1:
            return "Low"
        return "Normal"

    latest["anomaly_level"] = [
        label(z, float(mm) if pd.notna(mm) else 0.0)
        for z, mm in zip(latest["z_score"], latest["rainfall_mm"])
    ]

    latest = latest.sort_values(
        ["anomaly_level", "z_score"],
        ascending=[True, False],
    )

    state_average = (
        float(latest["rainfall_mm"].mean())
        if not latest.empty
        else None
    )

    return {
        "available": True,
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "district_count": int(len(latest)),
        "state_average_rainfall_mm": (
            round(state_average, 2)
            if state_average is not None
            else None
        ),
        "anomalies": clean_records(
            latest[
                [
                    "district", "rainfall_mm", "median",
                    "anomaly_mm", "anomaly_percent",
                    "z_score", "anomaly_level",
                ]
            ].to_dict("records")
        ),
        "rainfall_source": rainfall_source,
        "method": (
            rainfall_source + " is compared with "
            "each district's historical median and standard deviation "
            "from the IMD district dataset."
        ),
        "note": (
            "This is an anomaly indicator, not a flood probability "
            "and not an official warning."
        ),
    }


@app.get("/api/rainfall-anomaly")
def rainfall_anomaly():
    return {"status": "success", **build_rainfall_anomaly()}


# ============================================================
# WEATHER APIS
# ============================================================


@app.get("/api/weather/historical-analysis")
def historical_weather_analysis():
    """
    Historical weather analysis using the project's local
    weather_odisha.csv.csv dataset.

    This endpoint is descriptive historical analysis only.
    It is not a flood probability or official warning.
    """

    if not os.path.exists(WEATHER_FILE):
        return {
            "status": "success",
            "available": False,
            "message": (
                "Historical weather dataset not found: "
                + WEATHER_FILE
            ),
            "records": [],
            "historical": [],
            "monthly": [],
        }

    try:
        df = pd.read_csv(WEATHER_FILE, skiprows=13)
        # NASA_POWER_DOY_FIX
        if {"YEAR", "DOY"}.issubset(df.columns):
            df["date"] = pd.to_datetime(
                df["YEAR"].astype(int).astype(str)
                + df["DOY"].astype(int).astype(str).str.zfill(3),
                format="%Y%j",
                errors="coerce",
            )
        if df.empty:
            return {
                "status": "success",
                "available": False,
                "message": "Historical weather dataset is empty.",
                "records": [],
                "historical": [],
                "monthly": [],
            }

        # ----------------------------------------------------
        # Flexible column detection
        # ----------------------------------------------------
        normalized = {
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_"): column
            for column in df.columns
        }

        def choose_column(candidates):
            for candidate in candidates:
                key = (
                    str(candidate)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                if key in normalized:
                    return normalized[key]

            # Fallback to partial matching
            for key, original in normalized.items():
                for candidate in candidates:
                    candidate_key = (
                        str(candidate)
                        .strip()
                        .lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                    )

                    if candidate_key in key:
                        return original

            return None

        date_col = choose_column([
            "date",
            "datetime",
            "timestamp",
            "time",
        ])

        temperature_col = choose_column([
            "temperature_2m",
            "temperature_c",
            "temperature",
            "temp_c",
            "temp",
            "t2m",
        ])

        rainfall_col = choose_column([
            "precipitation",
            "precipitation_mm",
            "rainfall",
            "rainfall_mm",
            "rain_mm",
            "prectotcorr",
            "prectot",
        ])

        humidity_col = choose_column([
            "relative_humidity_2m",
            "relative_humidity",
            "humidity",
            "humidity_pct",
            "rh2m",
            "rh",
        ])

        wind_col = choose_column([
            "wind_speed_10m",
            "wind_speed",
            "wind_speed_kmh",
            "wind_speed_ms",
            "ws10m",
            "wind",
        ])

        pressure_col = choose_column([
            "surface_pressure",
            "pressure",
            "pressure_hpa",
            "pressure_kpa",
            "ps",
        ])

        if date_col is None:
            return {
                "status": "success",
                "available": False,
                "message": (
                    "Could not identify a date column in the "
                    "historical weather dataset."
                ),
                "columns": list(df.columns),
                "records": [],
                "historical": [],
                "monthly": [],
            }

        # ----------------------------------------------------
        # Clean date
        # ----------------------------------------------------
        df["analysis_date"] = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

        df = df.dropna(
            subset=["analysis_date"]
        ).copy()

        # ----------------------------------------------------
        # Convert detected numeric columns
        # ----------------------------------------------------
        field_map = {
            "temperature_c": temperature_col,
            "rainfall_mm": rainfall_col,
            "humidity_percent": humidity_col,
            "wind_speed": wind_col,
            "pressure": pressure_col,
        }

        for output_name, source_column in field_map.items():

            if source_column is not None:
                df[output_name] = pd.to_numeric(
                    df[source_column],
                    errors="coerce",
                )

                # Common NASA POWER missing-value marker
                df[output_name] = df[output_name].replace(
                    [-999, -999.0],
                    np.nan,
                )

            else:
                df[output_name] = np.nan


        # HISTORICAL_WEATHER_UNIT_NORMALIZATION_V2
        wind_source_name = str(wind_col or "").strip().lower()
        pressure_source_name = str(pressure_col or "").strip().lower()

        # NASA POWER WS10M is m/s -> dashboard uses km/h.
        if (
            wind_source_name == "ws10m"
            or wind_source_name.endswith("_ms")
            or "wind_speed_ms" in wind_source_name
        ):
            df["wind_speed"] = df["wind_speed"] * 3.6

        # NASA POWER PS is kPa -> dashboard uses hPa.
        if (
            pressure_source_name == "ps"
            or pressure_source_name.endswith("_kpa")
            or "pressure_kpa" in pressure_source_name
        ):
            df["pressure"] = df["pressure"] * 10.0

        df = (
            df.sort_values("analysis_date")
            .reset_index(drop=True)
        )

        if df.empty:
            return {
                "status": "success",
                "available": False,
                "message": "No valid historical weather dates were found.",
                "records": [],
                "historical": [],
                "monthly": [],
            }

        # ----------------------------------------------------
        # Descriptive trend helper
        # ----------------------------------------------------
        def calculate_trend(series):

            clean = pd.to_numeric(
                series,
                errors="coerce",
            ).dropna()

            if len(clean) < 4:
                return "Insufficient Data"

            midpoint = len(clean) // 2

            first_half = float(
                clean.iloc[:midpoint].mean()
            )

            second_half = float(
                clean.iloc[midpoint:].mean()
            )

            if abs(first_half) < 0.001:
                return "Stable"

            change_percent = (
                (second_half - first_half)
                / abs(first_half)
                * 100
            )

            if change_percent >= 2:
                return "Rising"

            if change_percent <= -2:
                return "Falling"

            return "Stable"

        # ----------------------------------------------------
        # Summary statistics
        # ----------------------------------------------------
        def safe_mean(column):
            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            return (
                round(float(values.mean()), 2)
                if not values.empty
                else None
            )

        def safe_min(column):
            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            return (
                round(float(values.min()), 2)
                if not values.empty
                else None
            )

        def safe_max(column):
            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            return (
                round(float(values.max()), 2)
                if not values.empty
                else None
            )

        rainfall_values = pd.to_numeric(
            df["rainfall_mm"],
            errors="coerce",
        ).dropna()

        total_rainfall = (
            round(float(rainfall_values.sum()), 2)
            if not rainfall_values.empty
            else None
        )

        summary = {
            "record_count": int(len(df)),

            "start_date": (
                df["analysis_date"]
                .min()
                .strftime("%Y-%m-%d")
            ),

            "end_date": (
                df["analysis_date"]
                .max()
                .strftime("%Y-%m-%d")
            ),

            "average_temperature_c": safe_mean(
                "temperature_c"
            ),

            "minimum_temperature_c": safe_min(
                "temperature_c"
            ),

            "maximum_temperature_c": safe_max(
                "temperature_c"
            ),

            "average_rainfall_mm": safe_mean(
                "rainfall_mm"
            ),

            "total_rainfall_mm": total_rainfall,

            "maximum_daily_rainfall_mm": safe_max(
                "rainfall_mm"
            ),

            "average_humidity_percent": safe_mean(
                "humidity_percent"
            ),

            "average_wind_speed": safe_mean(
                "wind_speed"
            ),

            "average_pressure": safe_mean(
                "pressure"
            ),

            "temperature_trend": calculate_trend(
                df["temperature_c"]
            ),

            "rainfall_trend": calculate_trend(
                df["rainfall_mm"]
            ),

            "humidity_trend": calculate_trend(
                df["humidity_percent"]
            ),

            "wind_trend": calculate_trend(
                df["wind_speed"]
            ),

            "pressure_trend": calculate_trend(
                df["pressure"]
            ),
        }

        # ----------------------------------------------------
        # Daily historical records
        # ----------------------------------------------------
        history = []

        for _, row in df.iterrows():

            history.append(
                {
                    "date": row["analysis_date"].strftime(
                        "%Y-%m-%d"
                    ),

                    "temperature_c": (
                        float(row["temperature_c"])
                        if pd.notna(row["temperature_c"])
                        else None
                    ),

                    "rainfall_mm": (
                        float(row["rainfall_mm"])
                        if pd.notna(row["rainfall_mm"])
                        else None
                    ),

                    "humidity_percent": (
                        float(row["humidity_percent"])
                        if pd.notna(row["humidity_percent"])
                        else None
                    ),

                    "wind_speed": (
                        float(row["wind_speed"])
                        if pd.notna(row["wind_speed"])
                        else None
                    ),

                    "pressure": (
                        float(row["pressure"])
                        if pd.notna(row["pressure"])
                        else None
                    ),
                }
            )

        # ----------------------------------------------------
        # Monthly aggregation
        # ----------------------------------------------------
        monthly_source = df.set_index(
            "analysis_date"
        )

        monthly = (
            monthly_source[
                [
                    "temperature_c",
                    "rainfall_mm",
                    "humidity_percent",
                    "wind_speed",
                    "pressure",
                ]
            ]
            .resample("MS")
            .mean()
            .reset_index()
        )

        monthly_rows = []

        for _, row in monthly.iterrows():

            monthly_rows.append(
                {
                    "month": row[
                        "analysis_date"
                    ].strftime("%Y-%m"),

                    "temperature_c": (
                        round(
                            float(
                                row["temperature_c"]
                            ),
                            2,
                        )
                        if pd.notna(
                            row["temperature_c"]
                        )
                        else None
                    ),

                    "rainfall_mm": (
                        round(
                            float(
                                row["rainfall_mm"]
                            ),
                            2,
                        )
                        if pd.notna(
                            row["rainfall_mm"]
                        )
                        else None
                    ),

                    "humidity_percent": (
                        round(
                            float(
                                row[
                                    "humidity_percent"
                                ]
                            ),
                            2,
                        )
                        if pd.notna(
                            row["humidity_percent"]
                        )
                        else None
                    ),

                    "wind_speed": (
                        round(
                            float(
                                row["wind_speed"]
                            ),
                            2,
                        )
                        if pd.notna(
                            row["wind_speed"]
                        )
                        else None
                    ),

                    "pressure": (
                        round(
                            float(
                                row["pressure"]
                            ),
                            2,
                        )
                        if pd.notna(
                            row["pressure"]
                        )
                        else None
                    ),
                }
            )


        # HISTORICAL_WEATHER_ANNUAL_SUMMARY_V2
        annual_source = df.copy()

        annual_source["analysis_year"] = (
            annual_source["analysis_date"].dt.year
        )

        annual_grouped = (
            annual_source
            .groupby("analysis_year")
            [
                [
                    "temperature_c",
                    "humidity_percent",
                    "wind_speed",
                    "pressure",
                ]
            ]
            .mean()
            .reset_index()
        )

        annual_counts = (
            annual_source
            .groupby("analysis_year")
            .size()
            .to_dict()
        )

        annual_rows = []

        for _, row in annual_grouped.iterrows():

            year_value = int(row["analysis_year"])

            annual_rows.append(
                {
                    "year": year_value,

                    "avg_temperature_c": (
                        round(float(row["temperature_c"]), 2)
                        if pd.notna(row["temperature_c"])
                        else None
                    ),

                    "avg_humidity_pct": (
                        round(float(row["humidity_percent"]), 2)
                        if pd.notna(row["humidity_percent"])
                        else None
                    ),

                    "avg_wind_speed_kmh": (
                        round(float(row["wind_speed"]), 2)
                        if pd.notna(row["wind_speed"])
                        else None
                    ),

                    "avg_pressure_hpa": (
                        round(float(row["pressure"]), 2)
                        if pd.notna(row["pressure"])
                        else None
                    ),

                    "records": int(
                        annual_counts.get(year_value, 0)
                    ),
                }
            )

        return {
            "status": "success",
            "available": True,

            "source": (
                "Flood Guard AI local historical "
                "weather dataset"
            ),

            "dataset": WEATHER_FILE,

            "columns_detected": {
                "date": date_col,
                "temperature": temperature_col,
                "rainfall": rainfall_col,
                "humidity": humidity_col,
                "wind": wind_col,
                "pressure": pressure_col,
            },

            # Several aliases are returned so the existing
            # frontend can consume the analysis safely.
            "summary": summary,
            "analysis": summary,
            "historical_analysis": summary,

            "records": clean_records(history),
            "historical": clean_records(history),
            "history": clean_records(history),

            "monthly": clean_records(
                monthly_rows
            ),

            "annual": clean_records(
                annual_rows
            ),

            "annual": clean_records(
                annual_rows
            ),

            "trend": {
                "temperature": summary[
                    "temperature_trend"
                ],
                "rainfall": summary[
                    "rainfall_trend"
                ],
                "humidity": summary[
                    "humidity_trend"
                ],
                "wind": summary[
                    "wind_trend"
                ],
                "pressure": summary[
                    "pressure_trend"
                ],
            },

            "note": (
                "Historical weather statistics and trends "
                "are descriptive summaries of the available "
                "project dataset. They are not flood "
                "probabilities or official warnings."
            ),
        }

    except Exception as exc:
        return {
            "status": "success",
            "available": False,
            "message": (
                "Historical weather analysis failed: "
                + str(exc)
            ),
            "records": [],
            "historical": [],
            "monthly": [],
        }



@app.get("/api/weather")
def get_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    forecast_days: int = Query(7, ge=1, le=16),
):
    try:
        payload = fetch_open_meteo_weather(
            latitude,
            longitude,
            forecast_days,
        )

        return {
            "status": "success",
            "weather": {
                "available": True,
                "latitude": latitude,
                "longitude": longitude,
                "timezone": payload.get("timezone"),
                "daily": build_daily_weather_rows(payload),
                "hourly": payload.get("hourly", {}),
                "source": "Open-Meteo forecast API",
            },
        }

    except Exception as exc:
        return {
            "status": "success",
            "weather": {
                "available": False,
                "latitude": latitude,
                "longitude": longitude,
                "message": (
                    "Weather service could not be reached: "
                    + str(exc)
                ),
            },
        }


@app.get("/api/weather/district/{district_name}")
def get_weather_for_district(district_name: str):
    return {
        "status": "success",
        "weather": get_district_weather(district_name),
    }


# ============================================================
# AREA ALERT / VULNERABILITY / IMPACT
# ============================================================

@app.get("/api/area-alert/{district_name}")
def area_alert(district_name: str):
    district = find_district(district_name)

    if district is None:
        raise HTTPException(
            status_code=404,
            detail="District not found.",
        )

    risk_level = district.get("risk_level", "Unavailable")
    risk_score = district.get("spatial_risk_score")
    forecast = get_forecast_summary()

    if risk_level == "Very High":
        alert_level = "Critical"
    elif risk_level == "High":
        alert_level = "High"
    elif risk_level == "Moderate":
        alert_level = "Moderate"
    elif risk_level == "Low":
        alert_level = "Monitor"
    else:
        alert_level = "Unavailable"

    weather = get_district_weather(district_name)

    weather_note = "Weather forecast unavailable."

    if weather.get("available"):
        daily = weather.get("daily", [])
        rainfall_values = [
            float(item["rainfall_mm"])
            for item in daily
            if item.get("rainfall_mm") is not None
        ]

        if rainfall_values:
            weather_note = (
                f"Maximum forecast daily precipitation over "
                f"the next {len(daily)} days is "
                f"{max(rainfall_values):.1f} mm."
            )

    message = (
        f"{district.get('NAME_2')} currently has {risk_level} "
        f"spatial risk (score {float(risk_score or 0):.3f}). "
        f"{weather_note}"
    )

    # Live 7-day outlook (river discharge + rain) decides the alert when available;
    # the long-term spatial risk is kept as background information.
    spatial_alert_level = alert_level
    live_outlook = None
    try:
        try:
            from backend import live_forecast_api as _lf
        except ImportError:
            import live_forecast_api as _lf
        _name, _lat, _lon, _risk = _lf._find(str(district.get("NAME_2")))
        _live = _lf.live_for(_name, _lat, _lon, _risk)
        _peak = _live.get("peak") or {}
        live_level = _peak.get("risk_level")
        if live_level:
            alert_level = {
                "Low": "Monitor",
                "Moderate": "Moderate",
                "High": "High",
                "Very High": "Critical",
            }.get(live_level, alert_level)
            future_rain = [
                d.get("rain_mm") or 0
                for d in _live.get("days", [])
                if not d.get("is_past")
            ]
            live_outlook = {
                "peak_level": live_level,
                "peak_date": _peak.get("date"),
                "today_level": _live.get("today_level"),
                "peak_discharge_m3s": _peak.get("discharge_m3s"),
                "high_flow_p95_m3s": (_live.get("thresholds_m3s") or {}).get("p95"),
                "max_rain_mm": max(future_rain) if future_rain else None,
            }
            message = (
                f"{district.get('NAME_2')}: live 7-day flood outlook is {live_level} "
                f"(highest on {_peak.get('date')}; {_peak.get('reason')}). "
                f"Long-term spatial risk is {risk_level} (score {float(risk_score or 0):.3f}). "
                f"{weather_note}"
            )
    except Exception:
        live_outlook = None

    return {
        "status": "success",
        "district": district,
        "alert_level": alert_level,
        "spatial_alert_level": spatial_alert_level,
        "live_outlook": live_outlook,
        "message": message,
        "forecast": forecast,
        "weather": weather,
        "note": (
            "Area alerts are project-defined decision-support "
            "outputs and are not official emergency warnings."
        ),
    }


@app.get("/api/vulnerability")
def get_vulnerability():
    if district_data.empty:
        raise HTTPException(
            status_code=404,
            detail="GIS risk data is unavailable.",
        )

    records = []

    for record in district_data.to_dict(orient="records"):
        profile = build_area_profile(record)
        records.append(
            {
                "district_id": profile.get("district_id"),
                "district": profile.get("NAME_2"),
                "vulnerability_score": profile.get(
                    "vulnerability_score"
                ),
                "impact_score": profile.get("impact_score"),
                "impact_risk_level": profile.get(
                    "impact_risk_level"
                ),
            }
        )

    return {
        "status": "success",
        "method": (
            "Existing elevation, drainage, historical-flood, "
            "and land-use risk components."
        ),
        "districts": clean_records(records),
        "note": (
            "Population and infrastructure exposure require "
            "additional datasets and are not inferred here."
        ),
    }


@app.get("/api/impact")
def get_impact():
    if district_data.empty:
        raise HTTPException(
            status_code=404,
            detail="GIS risk data is unavailable.",
        )

    data = district_data.copy()

    if "spatial_risk_score" in data.columns:
        data["risk_numeric"] = pd.to_numeric(
            data["spatial_risk_score"],
            errors="coerce",
        ).fillna(0.0)
        data = data.sort_values(
            "risk_numeric",
            ascending=False,
        ).head(10)

    rows = []

    for record in data.to_dict(orient="records"):
        profile = build_area_profile(record)

        rows.append(
            {
                "district": profile.get("NAME_2"),
                "risk_score": profile.get("spatial_risk_score"),
                "vulnerability_score": profile.get(
                    "vulnerability_score"
                ),
                "impact_score": profile.get("impact_score"),
                "impact_risk_level": profile.get(
                    "impact_risk_level"
                ),
                "population_at_risk": None,
                "critical_infrastructure_exposure": None,
                "infrastructure_damage_risk": None,
            }
        )

    return {
        "status": "success",
        "districts": clean_records(rows),
        "additional_data_required": [
            "population",
            "critical infrastructure",
            "shelters",
            "roads",
        ],
        "note": (
            "Unsupported impact fields remain null until the "
            "corresponding datasets are integrated."
        ),
    }


# ============================================================
# WHAT-IF APIS
# ============================================================

@app.get("/api/what-if")
def get_what_if(
    scenario: str | None = None,
):
    if what_if_data.empty:
        return {
            "status": "error",
            "message": "What-if scenario data not found",
        }

    available_scenarios = sorted(
        what_if_data["scenario"]
        .dropna()
        .unique()
        .tolist()
    )

    result_data = what_if_data.copy()

    if scenario:
        result_data = result_data[
            result_data["scenario"]
            .astype(str)
            .str.lower()
            .eq(scenario.lower())
        ]

        if result_data.empty:
            return {
                "status": "error",
                "message": "Scenario not found",
                "available_scenarios": available_scenarios,
            }

    results = clean_records(
        result_data.to_dict(orient="records")
    )

    summary_records = []

    if not what_if_summary.empty:
        summary_data = what_if_summary.copy()

        if scenario:
            summary_data = summary_data[
                summary_data["scenario"]
                .astype(str)
                .str.lower()
                .eq(scenario.lower())
            ]

        summary_records = clean_records(
            summary_data.to_dict(orient="records")
        )

    return {
        "status": "success",
        "selected_scenario": scenario if scenario else "all",
        "available_scenarios": available_scenarios,
        "result_count": len(results),
        "summary": summary_records,
        "results": results,
    }


@app.post("/api/what-if/analyze")
def analyze_what_if(request: WhatIfAnalysisRequest):
    if district_data.empty:
        raise HTTPException(
            status_code=404,
            detail="GIS spatial risk data is not available",
        )

    try:
        result = district_data.copy()

        rainfall_change = request.rainfall_change_percent / 100.0
        water_level_change = request.water_level_change_m
        drainage_change = (
            request.drainage_effectiveness_change_percent / 100.0
        )

        if "spatial_risk_score" not in result.columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "spatial_risk_score is missing from GIS risk data"
                ),
            )

        result["base_risk_score"] = pd.to_numeric(
            result["spatial_risk_score"],
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0)

        rainfall_effect = rainfall_change * 0.20
        water_level_effect = (
            water_level_change / 0.5
        ) * 0.10
        drainage_effect = -drainage_change * 0.20
        total_adjustment = (
            rainfall_effect
            + water_level_effect
            + drainage_effect
        )

        result["scenario_risk_score"] = (
            result["base_risk_score"] + total_adjustment
        ).clip(0.0, 1.0)

        result["risk_level"] = pd.cut(
            result["scenario_risk_score"],
            bins=[
                -0.001,
                0.25,
                0.50,
                0.75,
                1.0,
            ],
            labels=[
                "Low",
                "Moderate",
                "High",
                "Very High",
            ],
        ).astype(str)

        result["scenario"] = "Custom Scenario"
        result["rainfall_change_percent"] = (
            request.rainfall_change_percent
        )
        result["water_level_change_m"] = (
            request.water_level_change_m
        )
        result[
            "drainage_effectiveness_change_percent"
        ] = request.drainage_effectiveness_change_percent

        risk_distribution = (
            result["risk_level"]
            .value_counts()
            .to_dict()
        )

        response_columns = [
            "district_id",
            "NAME_1",
            "NAME_2",
            "historical_flood_events",
            "distance_to_river_m",
            "elevation_m",
            "elevation_risk",
            "drainage_risk",
            "historical_risk",
            "lulc_risk",
            "spatial_risk_score",
            "base_risk_score",
            "scenario_risk_score",
            "risk_level",
        ]

        available_columns = [
            column
            for column in response_columns
            if column in result.columns
        ]

        results = clean_records(
            result[available_columns].to_dict(orient="records")
        )

        response = {
            "status": "success",
            "scenario": {
                "rainfall_change_percent": (
                    request.rainfall_change_percent
                ),
                "water_level_change_m": (
                    request.water_level_change_m
                ),
                "drainage_effectiveness_change_percent": (
                    request.drainage_effectiveness_change_percent
                ),
            },
            "summary": {
                "average_risk_score": round(
                    float(result["scenario_risk_score"].mean()),
                    4,
                ),
                "minimum_risk_score": round(
                    float(result["scenario_risk_score"].min()),
                    4,
                ),
                "maximum_risk_score": round(
                    float(result["scenario_risk_score"].max()),
                    4,
                ),
                "low_districts": int(
                    risk_distribution.get("Low", 0)
                ),
                "moderate_districts": int(
                    risk_distribution.get("Moderate", 0)
                ),
                "high_districts": int(
                    risk_distribution.get("High", 0)
                ),
                "very_high_districts": int(
                    risk_distribution.get("Very High", 0)
                ),
            },
            "result_count": len(results),
            "results": results,
            "note": (
                "This scenario engine applies project-defined "
                "environmental adjustments to the existing spatial "
                "risk score. It is not a calibrated physical flood "
                "probability model."
            ),
        }

        save_scenario_history(response)

        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Dynamic What-If analysis failed: " + str(exc),
        )


@app.get("/api/what-if/compare")
def compare_what_if():
    if what_if_summary.empty:
        return {
            "status": "success",
            "available": False,
            "scenarios": [],
            "message": "No What-If summary data is available.",
        }

    scenarios = clean_records(
        what_if_summary.to_dict(orient="records")
    )

    return {
        "status": "success",
        "available": True,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


# ============================================================
# HISTORY APIS
# ============================================================

@app.get("/api/history/predictions")
def get_prediction_history(
    limit: int = Query(50, ge=1, le=500)
):
    connection = get_db_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        results = []

        for row in rows:
            item = dict(row)

            try:
                item["features_used"] = json.loads(
                    item.pop("features_json") or "{}"
                )
            except Exception:
                item["features_used"] = {}

            results.append(item)

        return {
            "status": "success",
            "count": len(results),
            "predictions": results,
        }
    finally:
        connection.close()


@app.get("/api/history/alerts")
def get_alert_history(
    limit: int = Query(50, ge=1, le=500)
):
    connection = get_db_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM alert_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        results = []

        for row in rows:
            item = dict(row)

            try:
                item["priority_districts"] = json.loads(
                    item.pop("priority_districts_json") or "[]"
                )
            except Exception:
                item["priority_districts"] = []

            results.append(item)

        return {
            "status": "success",
            "count": len(results),
            "alerts": results,
        }
    finally:
        connection.close()


@app.get("/api/history/scenarios")
def get_scenario_history(
    limit: int = Query(50, ge=1, le=500)
):
    connection = get_db_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM scenario_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return {
            "status": "success",
            "count": len(rows),
            "scenarios": [dict(row) for row in rows],
        }
    finally:
        connection.close()


@app.get("/api/history/reports")
def get_report_history(
    limit: int = Query(50, ge=1, le=500)
):
    connection = get_db_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM report_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        results = []

        for row in rows:
            item = dict(row)

            try:
                item["summary"] = json.loads(
                    item.pop("summary_json") or "{}"
                )
            except Exception:
                item["summary"] = {}

            results.append(item)

        return {
            "status": "success",
            "count": len(results),
            "reports": results,
        }
    finally:
        connection.close()


# ============================================================
# REPORT / DATA SOURCE / FEATURE READINESS APIS
# ============================================================

@app.get("/api/report/summary")
def get_report_summary():
    summary = build_report_summary()
    save_report_history(
        "risk_summary",
        summary,
    )

    return {
        "status": "success",
        "report": summary,
    }


@app.get("/api/data-sources")
def get_data_sources():
    return {
        "status": "success",
        "sources": {
            "spatial_risk": {
                "path": GIS_FILE,
                "available": os.path.exists(GIS_FILE),
            },
            "district_boundaries": {
                "path": BOUNDARY_FILE,
                "available": os.path.exists(BOUNDARY_FILE),
            },
            "historical_flood": {
                "path": HISTORICAL_FLOOD_FILE,
                "available": os.path.exists(
                    HISTORICAL_FLOOD_FILE
                ),
            },
            "river_network": {
                "path": RIVER_FILE,
                "available": os.path.exists(RIVER_FILE),
            },
            "rainfall": {
                "district_file": RAINFALL_DISTRICT_FILE,
                "state_file": RAINFALL_STATE_FILE,
                "available": (
                    os.path.exists(RAINFALL_DISTRICT_FILE)
                    or os.path.exists(RAINFALL_STATE_FILE)
                ),
            },
            "weather": {
                "path": WEATHER_FILE,
                "available": os.path.exists(WEATHER_FILE),
            },
            "forecast": {
                "historical_file": FORECAST_FILE,
                "future_file": NEXT_FORECAST_FILE,
                "available": (
                    os.path.exists(FORECAST_FILE)
                    or os.path.exists(NEXT_FORECAST_FILE)
                ),
            },
            "machine_learning": {
                "model_path": MODEL_FILE,
                "training_data": ML_TRAINING_FILE,
                "available": (
                    os.path.exists(MODEL_FILE)
                    and os.path.exists(ML_TRAINING_FILE)
                ),
            },
            "city_data": {
                "path": CITY_FILE,
                "available": os.path.exists(CITY_FILE),
            },
            "live_openstreetmap": {
                "available": True,
                "provider": "OpenStreetMap Overpass API",
            },
            "live_routing": {
                "available": True,
                "provider": "OSRM routing service",
            },
        },
    }


@app.get("/api/feature-readiness")
def get_feature_readiness():
    layers = build_gis_layer_status()

    return {
        "status": "success",
        "features": {
            "flood_risk_score": (
                not district_data.empty
                and "spatial_risk_score" in district_data.columns
            ),
            "flood_prediction": flood_model is not None,
            "flood_probability": flood_model is not None,
            "flood_explainability": (
                flood_model is not None
                and hasattr(
                    flood_model,
                    "feature_importances_",
                )
            ),
            "priority_area": get_priority_area() is not None,
            "forecast": (
                not forecast_data.empty
                or not next_forecast_data.empty
            ),
            "flood_risk_forecast": bool(
                build_forecast_risk_rows()
            ),
            "future_weather": True,
            "risk_timeline": True,
            "future_affected_districts": (
                district_geometry_gdf is not None
            ),
            "potentially_affected_cities": os.path.exists(CITY_FILE),
            "risk_propagation_screening": (
                district_geometry_gdf is not None
            ),
            "historical_flood_locations": layers[
                "historical_flood"
            ]["available"],
            "river_network": layers[
                "river_network"
            ]["available"],
            "vulnerability": not district_data.empty,
            "impact_assessment": not district_data.empty,
            "database_history": os.path.exists(DATABASE_FILE),
            "critical_infrastructure_risk": True,
            "infrastructure_exposure": True,
            "infrastructure_damage_risk": True,
            "shelter_safe_zone_screening": True,
            "road_connectivity_screening": True,
            "evacuation_route_planning": True,
            "city_spatial_risk_screening": True,
            "inundation_screening": True,
        },
        "notes": {
            "cities": (
                "City-level impact requires a city/urban location dataset."
            ),
            "population": (
                "Population-at-risk requires population or gridded population data."
            ),
            "infrastructure": (
                "Critical-infrastructure risk requires infrastructure location data."
            ),
            "shelters": (
                "Shelter mapping requires shelter/location data."
            ),
            "roads": (
                "Road and evacuation analysis uses live OpenStreetMap "
                "road geometry and OSRM routing when internet access is available."
            ),
            "infrastructure_live": (
                "Critical infrastructure screening uses OpenStreetMap "
                "public-service facility locations."
            ),
            "shelter_live": (
                "Shelter/safe-facility screening uses OpenStreetMap "
                "shelter/assembly-point data with a community-centre/school fallback."
            ),
            "city_spatial": (
                "City spatial screening uses OpenStreetMap city/town points, "
                "district risk, and river proximity; it is not hydraulic modelling."
            ),
            "inundation_screening": (
                "Inundation screening is a relative index based on existing "
                "spatial, drainage, elevation and water-level-change indicators."
            ),
        },
    }


# ============================================================
# RAPID EMERGENCY GIS INTELLIGENCE
# ============================================================

try:
    from shapely.geometry import Point, LineString
except Exception:
    Point = None
    LineString = None

OSM_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OSM_RETRY_CODES = (429, 502, 503, 504)   # server busy: wait and try again

# OpenStreetMap results are saved here, so a district loads instantly next time
# (and still works when the live service is slow or offline).
OSM_CACHE_DIR = os.path.join(BASE_DIR, "Datasets", "processed_data", "osm_cache")
OSM_CACHE_DAYS = 30


def _overpass_bbox(context):
    """Overpass expects (south, west, north, east); shapely bounds are (west, south, east, north)."""
    minx, miny, maxx, maxy = context["bbox"]
    return f"{miny},{minx},{maxy},{maxx}"

OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"


def _normalise_place_name(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _get_district_context(district_name):
    if district_geometry_gdf is None or district_geometry_gdf.empty:
        return None

    target = _normalise_place_name(district_name)

    matches = district_geometry_gdf[
        district_geometry_gdf["NAME_2"]
        .astype(str)
        .map(_normalise_place_name)
        .eq(target)
    ]

    if matches.empty:
        partial = district_geometry_gdf[
            district_geometry_gdf["NAME_2"]
            .astype(str)
            .map(_normalise_place_name)
            .str.contains(target, na=False)
        ]
        if partial.empty:
            return None
        matches = partial

    geom = matches.iloc[0].geometry
    risk_row = find_district(str(matches.iloc[0]["NAME_2"]))

    if geom is None or geom.is_empty:
        return None

    try:
        district_risk = float(
            risk_row.get("spatial_risk_score", 0.0)
            if risk_row
            else 0.0
        )
    except Exception:
        district_risk = 0.0

    return {
        "district": str(matches.iloc[0]["NAME_2"]),
        "geometry": geom,
        "risk_score": float(np.clip(district_risk, 0.0, 1.0)),
        "bbox": tuple(float(v) for v in geom.bounds),
    }


def _overpass_request(query):
    import hashlib
    key = hashlib.md5(" ".join(str(query).split()).encode("utf-8")).hexdigest()
    cache_file = os.path.join(OSM_CACHE_DIR, key + ".json")
    cached = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as handle:
                cached = json.load(handle)
            age_days = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 86400.0
            # an old empty result may be a busy-server reply: fetch it again once.
            # Results checked by this version carry a ".ok" marker and are trusted, even when empty.
            trusted = bool(cached) or os.path.exists(cache_file + ".ok")
            if age_days <= OSM_CACHE_DAYS and trusted:
                return cached
        except Exception:
            cached = None

    import time as _time
    from urllib.error import HTTPError
    last_error = None
    # Windows: this PC's network checks HTTPS traffic, which Python rejects but Windows accepts.
    # Try PowerShell first, which uses the Windows network settings and certificates.
    if os.name == "nt":
        import subprocess, tempfile
        for url in OSM_OVERPASS_URLS[:2]:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    qfile, ofile = os.path.join(tmp, "q.txt"), os.path.join(tmp, "o.json")
                    with open(qfile, "w", encoding="utf-8") as handle:
                        handle.write(query)
                    ps = (
                        "$ProgressPreference='SilentlyContinue'; "
                        f"$q = Get-Content -Raw -LiteralPath '{qfile}'; "
                        f"Invoke-WebRequest '{url}' -Method Post -Body @{{data=$q}} "
                        "-UserAgent 'FloodGuardAI/2.0 (academic project; contact: student)' "
                        "-Headers @{Accept='application/json'} -TimeoutSec 90 -UseBasicParsing "
                        f"-OutFile '{ofile}'"
                    )
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=120,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if os.path.exists(ofile) and os.path.getsize(ofile) > 0:
                        with open(ofile, encoding="utf-8") as handle:
                            payload = json.load(handle)
                        elements = payload.get("elements", [])
                        if not elements and payload.get("remark"):
                            last_error = f"{url}: {payload.get('remark')[:120]}"
                            continue
                        os.makedirs(OSM_CACHE_DIR, exist_ok=True)
                        with open(cache_file, "w", encoding="utf-8") as handle:
                            json.dump(elements, handle)
                        open(cache_file + ".ok", "w").close()
                        print(f"[OpenStreetMap] {len(elements)} records via PowerShell from {url}")
                        return elements
            except Exception as exc:
                last_error = f"PowerShell {url}: {exc}"
    for url in OSM_OVERPASS_URLS:
        for attempt in range(3):
            try:
                request = Request(
                    url,
                    data=urlencode({"data": query}).encode("utf-8"),
                    headers={
                        "User-Agent": "FloodGuardAI/2.0 (academic project; contact: student)",
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    method="POST",
                )
                with urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                elements = payload.get("elements", [])
                if not elements and payload.get("remark"):
                    last_error = f"{url}: {payload.get('remark')[:120]}"
                    break
                try:
                    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as handle:
                        json.dump(elements, handle)
                    open(cache_file + ".ok", "w").close()
                except Exception:
                    pass
                return elements
            except HTTPError as exc:
                last_error = f"{url}: HTTP {exc.code}"
                if exc.code in OSM_RETRY_CODES and attempt < 2:
                    _time.sleep(4 * (attempt + 1))
                    continue
                break
            except Exception as exc:
                last_error = f"{url}: {exc}"
                break
    print(f"[OpenStreetMap] no live result ({last_error}); using saved copy if any")

    # live service unavailable: fall back to an older saved copy if there is one
    return cached if cached is not None else []


_RIVER_REFERENCE_CACHE = {}


def _district_river_reference(context):
    if context is None or not os.path.exists(RIVER_FILE):
        return None
    cache_key = context.get("district")
    if cache_key in _RIVER_REFERENCE_CACHE:
        return _RIVER_REFERENCE_CACHE[cache_key]
    result = _district_river_reference_uncached(context)
    _RIVER_REFERENCE_CACHE[cache_key] = result
    return result


def _district_river_reference_uncached(context):

    try:
        minx, miny, maxx, maxy = context["bbox"]

        from shapely.geometry import box as _box
        rivers = gpd.read_file(
            RIVER_FILE,
            bbox=gpd.GeoDataFrame(geometry=[_box(minx, miny, maxx, maxy)], crs=4326),
        )

        if rivers.empty:
            return None

        if rivers.crs is None:
            rivers = rivers.set_crs(epsg=4326)

        rivers = rivers.to_crs(epsg=3857)
        geometries = [
            geom
            for geom in rivers.geometry
            if geom is not None and not geom.is_empty
        ]

        if not geometries:
            return None

        # keep the river lines separate (no union) and simplified to 50 m: much faster to measure
        series = gpd.GeoSeries(geometries, crs=3857).simplify(50)
        series = series[~series.is_empty].reset_index(drop=True)
        _ = series.sindex
        return series

    except Exception as exc:
        print(f"[River network] could not read rivers for {context.get('district')}: {exc}")
        return None


def _point_river_distance_m(point, river_reference):
    if (
        river_reference is None
        or Point is None
        or point is None
    ):
        return None

    try:
        projected = gpd.GeoSeries(
            [point],
            crs=4326,
        ).to_crs(epsg=3857).iloc[0]

        if hasattr(river_reference, "sindex"):
            try:
                _, dist = river_reference.sindex.nearest(projected, return_distance=True)
                return float(np.min(dist))
            except Exception:
                return float(river_reference.distance(projected).min())
        return float(projected.distance(river_reference))

    except Exception:
        return None


def _extract_osm_point(element):
    element_type = element.get("type")

    if element_type == "node":
        lat = element.get("lat")
        lon = element.get("lon")
    else:
        center = element.get("center") or {}
        lat = center.get("lat")
        lon = center.get("lon")

    if lat is None or lon is None:
        return None

    try:
        return float(lat), float(lon)
    except Exception:
        return None


def _filter_points_to_district(elements, context):
    if context is None or Point is None:
        return []

    geometry = context["geometry"]
    result = []

    for element in elements:
        coords = _extract_osm_point(element)

        if coords is None:
            continue

        lat, lon = coords
        point = Point(lon, lat)

        try:
            inside = geometry.covers(point)
        except Exception:
            inside = False

        if not inside:
            continue

        result.append(
            {
                "osm_id": element.get("id"),
                "latitude": lat,
                "longitude": lon,
                "name": (
                    element.get("tags", {}).get("name")
                    or "Unnamed facility"
                ),
                "tags": element.get("tags", {}),
            }
        )

    return result


def _infrastructure_category(tags):
    amenity = str(tags.get("amenity", "")).lower()

    weights = {
        "hospital": 1.00,
        "fire_station": 0.95,
        "police": 0.90,
        "clinic": 0.82,
        "doctors": 0.80,
        "school": 0.75,
        "college": 0.72,
        "university": 0.72,
    }

    category = amenity or "public facility"
    return category, float(weights.get(amenity, 0.65))


def _risk_from_location(
    district_risk,
    category_weight=0.65,
    distance_to_river_m=None,
):
    if distance_to_river_m is None:
        proximity_risk = 0.50
    else:
        proximity_risk = float(
            np.clip(
                1.0 - (float(distance_to_river_m) / 5000.0),
                0.0,
                1.0,
            )
        )

    score = (
        0.65 * district_risk
        + 0.20 * proximity_risk
        + 0.15 * category_weight
    )

    return float(np.clip(score, 0.0, 1.0))


@app.get("/api/infrastructure")
def get_infrastructure(
    district: str = Query(..., min_length=1),
):
    context = _get_district_context(district)

    if context is None:
        raise HTTPException(404, "District geometry/risk data not found.")

    bbox = _overpass_bbox(context)

    query = f"""
    [out:json][timeout:20];
    nwr
      ["amenity"~"hospital|fire_station|police|clinic|doctors|school|college|university"]
      ({bbox});
    out center tags;
    """

    try:
        elements = _overpass_request(query)
        facilities = _filter_points_to_district(elements, context)
        river_reference = _district_river_reference(context)

        records = []

        for item in facilities:
            tags = item["tags"]
            category, category_weight = _infrastructure_category(tags)
            point = Point(
                item["longitude"],
                item["latitude"],
            ) if Point is not None else None

            river_distance = _point_river_distance_m(
                point,
                river_reference,
            )

            risk = _risk_from_location(
                context["risk_score"],
                category_weight,
                river_distance,
            )

            level = spatial_score_to_risk_level(risk)

            records.append(
                {
                    "osm_id": item["osm_id"],
                    "name": item["name"],
                    "category": category,
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "distance_to_river_m": (
                        round(river_distance, 1)
                        if river_distance is not None
                        else None
                    ),
                    "risk_score": round(risk, 4),
                    "risk_level": level,
                }
            )

        records.sort(
            key=lambda row: (
                -float(row["risk_score"]),
                str(row["name"]).lower(),
            )
        )

        high_risk = [
            row for row in records
            if row["risk_score"] >= 0.65
        ]

        risk_sum = sum(row["risk_score"] for row in records)
        exposure_index = (
            min(1.0, risk_sum / max(len(records), 1))
            if records
            else 0.0
        )

        damage_risk = float(
            np.clip(
                exposure_index
                * (
                    0.70
                    + (
                        len(high_risk)
                        / max(len(records), 1)
                    )
                    * 0.30
                ),
                0.0,
                1.0,
            )
        )

        return clean_nested(
            {
                "status": "success",
                "available": True,
                "district": context["district"],
                "asset_count": len(records),
                "high_risk_asset_count": len(high_risk),
                "exposure_index": round(
                    exposure_index,
                    4,
                ),
                "damage_risk": round(
                    damage_risk,
                    4,
                ),
                "damage_risk_level": spatial_score_to_risk_level(
                    damage_risk
                ),
                "assets": records[:500],
                "method": (
                    "OpenStreetMap public-service facility locations "
                    "combined with the project's district spatial risk "
                    "and local river-proximity screening."
                ),
                "note": (
                    "Infrastructure exposure and damage are screening "
                    "indicators, not engineering loss estimates."
                ),
            }
        )

    except Exception as exc:
        return {
            "status": "success",
            "available": False,
            "district": context["district"],
            "asset_count": 0,
            "high_risk_asset_count": 0,
            "exposure_index": None,
            "damage_risk": None,
            "assets": [],
            "message": str(exc),
            "note": (
                "Live OpenStreetMap infrastructure data could not be "
                "retrieved. No facilities are invented."
            ),
        }


@app.get("/api/shelters")
def get_shelters(
    district: str = Query(..., min_length=1),
):
    context = _get_district_context(district)

    if context is None:
        raise HTTPException(404, "District geometry/risk data not found.")

    bbox = _overpass_bbox(context)

    primary_query = f"""
    [out:json][timeout:20];
    (
      nwr["amenity"="shelter"]({bbox});
      nwr["emergency"="assembly_point"]({bbox});
    );
    out center tags;
    """

    fallback_query = f"""
    [out:json][timeout:20];
    (
      nwr["amenity"="community_centre"]({bbox});
      nwr["amenity"="school"]({bbox});
    );
    out center tags;
    """

    try:
        elements = _overpass_request(primary_query)
        source_type = "verified OSM shelter/assembly tags"

        facilities = _filter_points_to_district(
            elements,
            context,
        )

        if not facilities:
            elements = _overpass_request(fallback_query)
            source_type = (
                "potential safe-facility fallback "
                "(community centre/school)"
            )
            facilities = _filter_points_to_district(
                elements,
                context,
            )

        river_reference = _district_river_reference(context)

        records = []

        for item in facilities:
            tags = item["tags"]
            amenity = str(tags.get("amenity", "")).lower()
            emergency_tag = str(
                tags.get("emergency", "")
            ).lower()

            verified = (
                amenity == "shelter"
                or emergency_tag == "assembly_point"
            )

            point = Point(
                item["longitude"],
                item["latitude"],
            ) if Point is not None else None

            river_distance = _point_river_distance_m(
                point,
                river_reference,
            )

            river_risk = (
                0.50
                if river_distance is None
                else float(
                    np.clip(
                        1.0
                        - (
                            float(river_distance)
                            / 5000.0
                        ),
                        0.0,
                        1.0,
                    )
                )
            )

            exposure_score = float(
                np.clip(
                    (
                        0.75 * context["risk_score"]
                        + 0.25 * river_risk
                    ),
                    0.0,
                    1.0,
                )
            )

            suitability = float(
                np.clip(
                    1.0 - exposure_score,
                    0.0,
                    1.0,
                )
            )

            records.append(
                {
                    "osm_id": item["osm_id"],
                    "name": item["name"],
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "facility_type": (
                        "Shelter"
                        if verified
                        else amenity or "potential safe facility"
                    ),
                    "verified_osm_shelter": verified,
                    "distance_to_river_m": (
                        round(river_distance, 1)
                        if river_distance is not None
                        else None
                    ),
                    "exposure_score": round(
                        exposure_score,
                        4,
                    ),
                    "suitability_score": round(
                        suitability,
                        4,
                    ),
                }
            )

        records.sort(
            key=lambda row: (
                -float(row["suitability_score"]),
                str(row["name"]).lower(),
            )
        )

        return clean_nested(
            {
                "status": "success",
                "available": bool(records),
                "district": context["district"],
                "facility_count": len(records),
                "verified_shelter_count": sum(
                    1
                    for row in records
                    if row["verified_osm_shelter"]
                ),
                "best_facility": (
                    records[0]
                    if records
                    else None
                ),
                "facilities": records[:300],
                "source_type": source_type,
                "method": (
                    "OpenStreetMap shelter/assembly-point locations "
                    "with community-centre/school fallback and "
                    "project spatial-risk/river-proximity screening."
                ),
                "note": (
                    "Suitability is a screening score. Facilities "
                    "must be locally verified before emergency use."
                ),
            }
        )

    except Exception as exc:
        return {
            "status": "success",
            "available": False,
            "district": context["district"],
            "facility_count": 0,
            "verified_shelter_count": 0,
            "best_facility": None,
            "facilities": [],
            "message": str(exc),
            "note": (
                "Live shelter/facility data could not be retrieved."
            ),
        }


@app.get("/api/roads")
def get_roads(
    district: str = Query(..., min_length=1),
):
    context = _get_district_context(district)

    if context is None:
        raise HTTPException(404, "District geometry/risk data not found.")

    bbox = _overpass_bbox(context)

    query = f"""
    [out:json][timeout:25];
    way
      ["highway"~"motorway|trunk|primary|secondary|tertiary"]
      ({bbox});
    out geom tags;
    """

    class_weights = {
        "motorway": 1.00,
        "trunk": 0.95,
        "primary": 0.85,
        "secondary": 0.72,
        "tertiary": 0.60,
    }

    try:
        elements = _overpass_request(query)
        geometry = context["geometry"]

        roads = []
        total_km = 0.0
        weighted_exposure_km = 0.0

        for element in elements:
            geometry_points = element.get("geometry") or []

            if len(geometry_points) < 2 or LineString is None:
                continue

            try:
                line = LineString(
                    [
                        (
                            float(point["lon"]),
                            float(point["lat"]),
                        )
                        for point in geometry_points
                    ]
                )

                if not line.intersects(geometry):
                    continue

                tags = element.get("tags", {})
                road_class = str(
                    tags.get("highway", "road")
                ).lower()

                if road_class not in class_weights:
                    continue

                projected_line = gpd.GeoSeries(
                    [line],
                    crs=4326,
                ).to_crs(epsg=3857).iloc[0]

                length_km = float(
                    projected_line.length / 1000.0
                )

                risk = float(
                    np.clip(
                        0.75 * context["risk_score"]
                        + 0.25 * class_weights[road_class],
                        0.0,
                        1.0,
                    )
                )

                total_km += length_km
                weighted_exposure_km += (
                    length_km * risk
                )

                coords = [
                    [float(p[1]), float(p[0])]
                    for p in geometry_points
                ]

                roads.append(
                    {
                        "osm_id": element.get("id"),
                        "name": tags.get(
                            "name",
                            "Unnamed road",
                        ),
                        "road_class": road_class,
                        "length_km": round(
                            length_km,
                            3,
                        ),
                        "risk_score": round(
                            risk,
                            4,
                        ),
                        "risk_level": spatial_score_to_risk_level(
                            risk
                        ),
                        "coordinates": coords,
                    }
                )

            except Exception:
                continue

        roads.sort(
            key=lambda row: (
                -float(row["risk_score"]),
                -float(row["length_km"]),
            )
        )

        exposure_index = float(
            np.clip(
                weighted_exposure_km
                / max(total_km, 0.001),
                0.0,
                1.0,
            )
        )

        return clean_nested(
            {
                "status": "success",
                "available": bool(roads),
                "district": context["district"],
                "road_segment_count": len(roads),
                "major_road_km": round(
                    total_km,
                    2,
                ),
                "road_exposure_index": round(
                    exposure_index,
                    4,
                ),
                "road_risk_level": spatial_score_to_risk_level(
                    exposure_index
                ),
                "roads": roads[:600],
                "method": (
                    "OpenStreetMap major-road geometry combined with "
                    "the selected district's spatial-risk score and "
                    "road-class criticality."
                ),
                "note": (
                    "This is a road-connectivity/flood-exposure "
                    "screening indicator, not a traffic simulation."
                ),
            }
        )

    except Exception as exc:
        return {
            "status": "success",
            "available": False,
            "district": context["district"],
            "road_segment_count": 0,
            "major_road_km": 0,
            "road_exposure_index": None,
            "road_risk_level": "Unavailable",
            "roads": [],
            "message": str(exc),
            "note": (
                "Live OpenStreetMap road data could not be retrieved."
            ),
        }


@app.get("/api/evacuation-route")
def get_evacuation_route(
    district: str = Query(..., min_length=1),
):
    context = _get_district_context(district)

    if context is None:
        raise HTTPException(
            status_code=404,
            detail="District geometry/risk data not found.",
        )

    unavailable = {
        "status": "success",
        "available": False,
        "district": context["district"],
        "route_available": False,
        "origin": None,
        "destination": None,
        "distance_km": None,
        "duration_minutes": None,
        "route_risk_score": None,
        "route_risk_level": "Unavailable",
        "route_coordinates": None,
        "method": (
            "Road route (OSRM, OpenStreetMap roads) from the district representative point "
            "to the highest-ranked mapped safe facility."
        ),
        "note": "Routes do not account for live road closures or flooded roads.",
    }

    # destination: the best mapped safe facility (from the saved OpenStreetMap shelter data)
    try:
        shelters = get_shelters(district)
    except Exception:
        shelters = None
    facilities = (shelters or {}).get("facilities") or []
    if not facilities:
        unavailable["message"] = "No mapped safe facility is available for this district yet."
        return clean_nested(unavailable)
    best = max(facilities, key=lambda f: f.get("suitability_score") or 0)

    origin_point = context["geometry"].representative_point()
    origin = {"name": f"{context['district']} (district representative point)",
              "latitude": float(origin_point.y), "longitude": float(origin_point.x)}
    destination = {"name": best.get("name") or "Mapped safe facility",
                   "facility_type": best.get("facility_type"),
                   "latitude": float(best["latitude"]), "longitude": float(best["longitude"]),
                   "suitability_score": best.get("suitability_score")}

    url = (f"{OSRM_ROUTE_URL}/{origin['longitude']:.5f},{origin['latitude']:.5f};"
           f"{destination['longitude']:.5f},{destination['latitude']:.5f}"
           "?overview=simplified&geometries=geojson")
    data = _web_get_json(url)
    routes = (data or {}).get("routes") or []
    if not routes:
        unavailable.update(origin=origin, destination=destination,
                           message="The road-routing service could not be reached; destination shown without a route.")
        return clean_nested(unavailable)

    route = routes[0]
    coords = (route.get("geometry") or {}).get("coordinates") or []
    suitability = float(best.get("suitability_score") or 0.5)
    route_risk = float(np.clip(0.6 * context["risk_score"] + 0.4 * (1.0 - suitability), 0.0, 1.0))
    return clean_nested({
        **unavailable,
        "available": True,
        "route_available": True,
        "origin": origin,
        "destination": destination,
        "distance_km": round(float(route.get("distance", 0)) / 1000.0, 2),
        "duration_minutes": round(float(route.get("duration", 0)) / 60.0, 1),
        "route_risk_score": round(route_risk, 3),
        "route_risk_level": spatial_score_to_risk_level(route_risk),
        "route_coordinates": coords,
        "route_risk_method": "60% district spatial risk + 40% (1 − destination suitability)",
        "message": "Road route generated with OSRM on OpenStreetMap roads.",
    })


def _web_get_json(url, timeout=30):
    """GET a JSON web service; saved to the OpenStreetMap cache. On Windows, PowerShell is
    used when Python's own connection is refused by the network's HTTPS checking."""
    import hashlib
    key = "web_" + hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_file = os.path.join(OSM_CACHE_DIR, key + ".json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass
    data = None
    try:
        request = Request(url, headers={"User-Agent": "FloodGuardAI/2.0 (academic project; contact: student)",
                                        "Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[Routing] Python request failed ({exc}); trying PowerShell")
    if data is None and os.name == "nt":
        import subprocess, tempfile
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ofile = os.path.join(tmp, "o.json")
                ps = ("$ProgressPreference='SilentlyContinue'; "
                      f"Invoke-WebRequest '{url}' -UserAgent 'FloodGuardAI/2.0 (academic project; contact: student)' "
                      f"-TimeoutSec 60 -UseBasicParsing -OutFile '{ofile}'")
                subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=90,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if os.path.exists(ofile) and os.path.getsize(ofile) > 0:
                    with open(ofile, encoding="utf-8") as handle:
                        data = json.load(handle)
        except Exception as exc:
            print(f"[Routing] PowerShell request failed: {exc}")
    if data is not None:
        try:
            os.makedirs(OSM_CACHE_DIR, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except Exception:
            pass
    return data


@app.get("/api/city-risk")
def get_city_risk(
    district: str = Query(..., min_length=1),
):
    context = _get_district_context(district)

    if context is None:
        raise HTTPException(404, "District geometry/risk data not found.")

    bbox = _overpass_bbox(context)

    query = f"""
    [out:json][timeout:20];
    node
      ["place"~"city|town|municipality"]
      ({bbox});
    out body;
    """

    try:
        elements = _overpass_request(query)
        places = _filter_points_to_district(
            elements,
            context,
        )

        river_reference = _district_river_reference(context)

        records = []

        for item in places:
            point = Point(
                item["longitude"],
                item["latitude"],
            ) if Point is not None else None

            river_distance = _point_river_distance_m(
                point,
                river_reference,
            )

            if river_distance is None:
                proximity_risk = 0.50
            else:
                proximity_risk = float(
                    np.clip(
                        1.0
                        - (
                            float(river_distance)
                            / 5000.0
                        ),
                        0.0,
                        1.0,
                    )
                )

            score = float(
                np.clip(
                    0.70 * context["risk_score"]
                    + 0.30 * proximity_risk,
                    0.0,
                    1.0,
                )
            )

            records.append(
                {
                    "city_name": item["name"],
                    "place_type": (
                        item["tags"].get("place")
                        or "urban place"
                    ),
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "distance_to_river_m": (
                        round(river_distance, 1)
                        if river_distance is not None
                        else None
                    ),
                    "district_risk_score": round(
                        context["risk_score"],
                        4,
                    ),
                    "city_spatial_risk_score": round(
                        score,
                        4,
                    ),
                    "risk_level": spatial_score_to_risk_level(
                        score
                    ),
                }
            )

        records.sort(
            key=lambda row: (
                -float(row["city_spatial_risk_score"]),
                str(row["city_name"]).lower(),
            )
        )

        return clean_nested(
            {
                "status": "success",
                "available": bool(records),
                "district": context["district"],
                "city_count": len(records),
                "highest_risk_city": (
                    records[0]
                    if records
                    else None
                ),
                "cities": records[:200],
                "method": (
                    "OpenStreetMap city/town points combined with "
                    "district spatial risk and local river proximity "
                    "to provide city-level spatial screening."
                ),
                "note": (
                    "This improves spatial localisation but is still "
                    "screening, not hydraulic city inundation modelling."
                ),
            }
        )

    except Exception as exc:
        return {
            "status": "success",
            "available": False,
            "district": context["district"],
            "city_count": 0,
            "highest_risk_city": None,
            "cities": [],
            "message": str(exc),
        }


@app.get("/api/inundation-screening")
def get_inundation_screening(
    district: str = Query(..., min_length=1),
    water_level_change_m: float = Query(
        0.0,
        ge=-10.0,
        le=10.0,
    ),
):
    profile = find_district(district)

    if profile is None:
        raise HTTPException(404, "District risk data not found.")

    risk_score = float(
        np.clip(
            profile.get("spatial_risk_score") or 0.0,
            0.0,
            1.0,
        )
    )

    drainage_risk = float(
        np.clip(
            profile.get("drainage_risk") or 0.0,
            0.0,
            1.0,
        )
    )

    elevation_risk = float(
        np.clip(
            profile.get("elevation_risk") or 0.0,
            0.0,
            1.0,
        )
    )

    base_index = float(
        np.clip(
            0.45 * risk_score
            + 0.35 * drainage_risk
            + 0.20 * elevation_risk,
            0.0,
            1.0,
        )
    )

    adjusted_index = float(
        np.clip(
            base_index
            + (
                float(water_level_change_m)
                * 0.12
            ),
            0.0,
            1.0,
        )
    )

    return {
        "status": "success",
        "available": True,
        "district": profile.get("NAME_2"),
        "base_inundation_screening_index": round(
            base_index,
            4,
        ),
        "water_level_adjusted_index": round(
            adjusted_index,
            4,
        ),
        "risk_level": spatial_score_to_risk_level(
            adjusted_index
        ),
        "components": {
            "spatial_risk": round(
                risk_score,
                4,
            ),
            "drainage_risk": round(
                drainage_risk,
                4,
            ),
            "elevation_risk": round(
                elevation_risk,
                4,
            ),
            "water_level_change_m": float(
                water_level_change_m
            ),
        },
        "method": (
            "Relative inundation screening from the existing "
            "district spatial risk, drainage risk, elevation risk "
            "and user-supplied water-level-change stress."
        ),
        "note": (
            "This is a terrain/risk screening index, not a calibrated "
            "hydraulic depth or inundation simulation."
        ),
    }

# ============================================================
# MODEL PERFORMANCE
# ============================================================

def build_model_performance():
    if flood_model is None:
        return {
            "available": False,
            "message": "Flood prediction model is not available.",
        }

    result = {
        "available": True,
        "model_name": type(flood_model).__name__,
        "model_file": MODEL_FILE,
        "model_features": MODEL_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "model_parameters": {},
        "feature_importance": [],
        "evaluation": {
            "available": False,
            "message": (
                "Held-out evaluation could not be generated."
            ),
        },
    }

    parameter_names = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "class_weight",
        "random_state",
        "n_jobs",
        "max_features",
    ]

    for parameter in parameter_names:
        if hasattr(flood_model, parameter):
            result["model_parameters"][parameter] = clean_value(
                getattr(flood_model, parameter)
            )

    if hasattr(flood_model, "feature_importances_"):
        importances = np.asarray(
            flood_model.feature_importances_,
            dtype=float,
        )

        names = MODEL_FEATURES.copy()

        if hasattr(flood_model, "feature_names_in_"):
            model_names = list(flood_model.feature_names_in_)
            if len(model_names) == len(importances):
                names = model_names

        importance_records = []

        for index, importance in enumerate(importances):
            feature = (
                names[index]
                if index < len(names)
                else f"feature_{index}"
            )

            importance_records.append(
                {
                    "feature": feature,
                    "display_name": FEATURE_DISPLAY_NAMES.get(
                        feature,
                        feature,
                    ),
                    "importance": float(importance),
                    "importance_percent": float(
                        importance * 100.0
                    ),
                }
            )

        importance_records.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        result["feature_importance"] = importance_records

    target_column = None

    for candidate in [
        "high_water_event",
        "target",
        "label",
        "flood_event",
    ]:
        if candidate in ml_training_data.columns:
            target_column = candidate
            break

    if (
        not ml_training_data.empty
        and target_column is not None
        and all(
            feature in ml_training_data.columns
            for feature in MODEL_FEATURES
        )
    ):
        try:
            evaluation_data = (
                ml_training_data[
                    MODEL_FEATURES + [target_column]
                ]
                .copy()
            )

            for feature in MODEL_FEATURES:
                evaluation_data[feature] = pd.to_numeric(
                    evaluation_data[feature],
                    errors="coerce",
                )

            evaluation_data[target_column] = pd.to_numeric(
                evaluation_data[target_column],
                errors="coerce",
            )

            evaluation_data = (
                evaluation_data
                .dropna()
                .reset_index(drop=True)
            )

            if len(evaluation_data) >= 20:
                from sklearn.model_selection import train_test_split

                X = evaluation_data[MODEL_FEATURES]
                y = evaluation_data[target_column]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y,
                )

                evaluation_model = clone(flood_model)

                evaluation_model.fit(
                    X_train,
                    y_train,
                )

                y_pred = evaluation_model.predict(
                    X_test
                )

                y_probability = None

                if hasattr(
                    evaluation_model,
                    "predict_proba"
                ):
                    try:
                        y_probability = (
                            evaluation_model
                            .predict_proba(X_test)[:, 1]
                        )
                    except Exception:
                        y_probability = None

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                )

                roc_auc = None

                if (
                    y_probability is not None
                    and len(
                        np.unique(y_test)
                    ) > 1
                ):
                    try:
                        roc_auc = roc_auc_score(
                            y_test,
                            y_probability,
                        )
                    except ValueError:
                        roc_auc = None

                result["evaluation"] = {
                    "available": True,
                    "evaluation_type": (
                        "Stratified random 80/20 holdout evaluation "
                        "matching models/04_train_model.py"
                    ),
                    "target_column": target_column,
                    "training_samples": int(
                        len(X_train)
                    ),
                    "test_samples": int(
                        len(X_test)
                    ),
                    "negative_test_events": int(
                        (y_test == 0).sum()
                    ),
                    "positive_test_events": int(
                        (y_test == 1).sum()
                    ),
                    "positive_class_share_percent": float(
                        (y_test == 1).mean() * 100.0
                    ),
                    "accuracy_percent": float(
                        accuracy_score(
                            y_test,
                            y_pred,
                        ) * 100.0
                    ),
                    "precision_percent": float(
                        precision * 100.0
                    ),
                    "recall_percent": float(
                        recall * 100.0
                    ),
                    "f1_percent": float(
                        f1 * 100.0
                    ),
                    "roc_auc": (
                        float(roc_auc)
                        if roc_auc is not None
                        else None
                    ),
                    "confusion_matrix": cm.tolist(),
                    "target_note": (
                        "The project target represents the available "
                        "high-water event proxy used during model training; "
                        "it is not confirmed flood-ground-truth."
                    ),
                }
        except Exception as exc:
            result["evaluation"] = {
                "available": False,
                "message": "Holdout evaluation failed: " + str(exc),
            }

    return clean_nested(result)


@app.get("/api/model-performance")
def get_model_performance():
    try:
        result = build_model_performance()
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model performance analysis failed: " + str(exc),
        )


# ============================================================
# DATA QUALITY
# ============================================================

def profile_csv_file(file_path):
    profile = {
        "path": file_path,
        "exists": os.path.exists(file_path),
        "readable": False,
        "size_mb": None,
        "modified_at": None,
        "modified_days_ago": None,
        "row_count": None,
        "column_count": None,
        "missing_cells_percent": None,
        "date_range": None,
        "status": "Missing",
    }

    if not os.path.exists(file_path):
        return profile

    try:
        stats = os.stat(file_path)
        profile["size_mb"] = round(
            stats.st_size / (1024 * 1024),
            3,
        )

        modified_datetime = datetime.fromtimestamp(stats.st_mtime)

        profile["modified_at"] = modified_datetime.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        profile["modified_days_ago"] = round(
            (
                datetime.now() - modified_datetime
            ).total_seconds()
            / 86400.0,
            2,
        )

        try:
            data = pd.read_csv(file_path)
        except Exception:
            # NASA POWER style files start with a text header block ending in "-END HEADER-"
            skip = 0
            with open(file_path, encoding="utf-8", errors="ignore") as handle:
                for index, line in enumerate(handle):
                    if "END HEADER" in line.upper():
                        skip = index + 1
                        break
                    if index > 60:
                        break
            data = pd.read_csv(file_path, skiprows=skip)
        profile["readable"] = True
        profile["row_count"] = int(len(data))
        profile["column_count"] = int(len(data.columns))

        if data.size > 0:
            profile["missing_cells_percent"] = round(
                (
                    data.isna().sum().sum()
                    / data.size
                )
                * 100.0,
                2,
            )

        date_column = get_date_column(data)

        if not date_column and {"YEAR", "DOY"}.issubset(data.columns):
            try:
                ydoy = pd.to_datetime(
                    data["YEAR"].astype(int).astype(str)
                    + data["DOY"].astype(int).astype(str).str.zfill(3),
                    format="%Y%j",
                    errors="coerce",
                ).dropna()
                if not ydoy.empty:
                    profile["date_range"] = {
                        "start": ydoy.min().strftime("%Y-%m-%d"),
                        "end": ydoy.max().strftime("%Y-%m-%d"),
                    }
            except Exception:
                pass

        if date_column:
            date_values = pd.to_datetime(
                data[date_column],
                errors="coerce",
            ).dropna()

            if not date_values.empty:
                profile["date_range"] = {
                    "start": pd.Timestamp(
                        date_values.min()
                    ).strftime("%Y-%m-%d"),
                    "end": pd.Timestamp(
                        date_values.max()
                    ).strftime("%Y-%m-%d"),
                }

        profile["status"] = "Available"

    except Exception as exc:
        profile["status"] = "Read Error"
        profile["error"] = str(exc)

    return profile


def build_data_quality_report():
    csv_files = {
        "Spatial Risk Data": GIS_FILE,
        "What-If Results": WHAT_IF_FILE,
        "What-If Summary": WHAT_IF_SUMMARY_FILE,
        "Historical Forecast": FORECAST_FILE,
        "7-Day Forecast": NEXT_FORECAST_FILE,
        "ML Training Data": ML_TRAINING_FILE,
        "District Rainfall": RAINFALL_DISTRICT_FILE,
        "State Rainfall": RAINFALL_STATE_FILE,
        "Weather": WEATHER_FILE,
    }

    datasets = []

    for name, path in csv_files.items():
        item = profile_csv_file(path)
        item["name"] = name
        datasets.append(item)

    non_csv_files = {
        "Boundary GeoPackage": BOUNDARY_FILE,
        "Historical Flood Layer": HISTORICAL_FLOOD_FILE,
        "River Network": RIVER_FILE,
        "Spatial Risk GeoPackage": GIS_GPKG_FILE,
        "Flood Prediction Model": MODEL_FILE,
        "Application Database": DATABASE_FILE,
        "City Dataset": CITY_FILE,
    }

    files = []

    for name, path in non_csv_files.items():
        item = {
            "name": name,
            "path": path,
            "exists": os.path.exists(path),
            "status": (
                "Available" if os.path.exists(path) else "Missing"
            ),
            "size_mb": None,
            "modified_at": None,
            "modified_days_ago": None,
        }

        if os.path.exists(path):
            try:
                stats = os.stat(path)

                item["size_mb"] = round(
                    stats.st_size / (1024 * 1024),
                    3,
                )

                modified_datetime = datetime.fromtimestamp(
                    stats.st_mtime
                )

                item["modified_at"] = modified_datetime.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                item["modified_days_ago"] = round(
                    (
                        datetime.now() - modified_datetime
                    ).total_seconds()
                    / 86400.0,
                    2,
                )
            except Exception as exc:
                item["status"] = "Read Error"
                item["error"] = str(exc)

        files.append(item)

    total_datasets = len(datasets)
    available_datasets = sum(
        1
        for item in datasets
        if item["status"] == "Available"
    )

    total_files = len(files)
    available_files = sum(
        1
        for item in files
        if item["status"] == "Available"
    )

    if (
        available_datasets == total_datasets
        and available_files == total_files
    ):
        overall_status = "Healthy"
    elif available_datasets > 0 or available_files > 0:
        overall_status = "Partial"
    else:
        overall_status = "Unavailable"

    return {
        "available": True,
        "overall_status": overall_status,
        "summary": {
            "csv_datasets": total_datasets,
            "available_csv_datasets": available_datasets,
            "supporting_files": total_files,
            "available_supporting_files": available_files,
        },
        "datasets": datasets,
        "supporting_files": files,
        "freshness_note": (
            "File modification time indicates when the local project file "
            "was last changed. It does not prove that the underlying source "
            "data is real-time or live."
        ),
    }


@app.get("/api/data-quality")
def get_data_quality():
    try:
        result = build_data_quality_report()
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Data quality report failed: " + str(exc),
        )

# --- Flood Guard AI v2 (real flood events) ---
try:
    from backend.flood_v2_api import router as flood_v2_router
except ImportError:
    from flood_v2_api import router as flood_v2_router
app.include_router(flood_v2_router)

# --- Live 7-day flood forecast (Open-Meteo Flood API / GloFAS + rainfall) ---
try:
    from backend.live_forecast_api import router as live_forecast_router
except ImportError:
    from live_forecast_api import router as live_forecast_router
app.include_router(live_forecast_router)

# --- Live ML prediction (v2 model on live rainfall) and district response plan ---
try:
    from backend.live_ml_api import router as live_ml_router
    from backend.response_api import router as response_router
except ImportError:
    from live_ml_api import router as live_ml_router
    from response_api import router as response_router
app.include_router(live_ml_router)
app.include_router(response_router)
