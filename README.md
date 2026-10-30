# Flood Guard AI

## AI-Based Urban Flood Forecasting, Impact Mapping and What-If Emergency Decision Support System

Flood Guard AI is an academic AI + GIS decision-support system for district-level flood-risk analysis in Odisha. It combines rainfall, water-level, weather, terrain, land-use/land-cover, drainage, historical flood, machine-learning, forecasting, and geospatial analysis in an interactive web dashboard.

---

## Main Features

* Odisha district-level spatial flood-risk map
* District Explorer with risk indicators
* Random Forest high-water-event classification
* Water-level forecasting using Random Forest regression
* Dynamic What-If emergency scenario analysis
* Predefined What-If scenario analysis
* FastAPI backend
* React + Vite frontend
* Leaflet interactive mapping
* Recharts forecasting visualization
* End-to-end frontend and backend integration

---

## Technology Stack

### Backend

* Python 3.13
* FastAPI
* Uvicorn
* Pandas
* NumPy
* Scikit-learn
* GeoPandas
* Rasterio
* XGBoost

### Frontend

* React
* Vite
* Leaflet
* React-Leaflet
* Recharts

### GIS

* QGIS
* GADM administrative boundaries
* ESA WorldCover 10 m LULC
* Copernicus DEM
* River/drainage network data
* Historical flood inventory

---

## Project Workflow

```text
Datasets
   ↓
Validation
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
ML Target Creation
   ↓
Machine Learning
   ↓
Forecasting
   ↓
GIS Risk Mapping
   ↓
What-If Analysis
   ↓
FastAPI Backend
   ↓
React Frontend
   ↓
Flood Guard AI Dashboard
```

---

## Project Structure

```text
D:\Flood_Ai
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_backend.bat
├── run_frontend.bat
│
├── backend
├── frontend
├── Datasets          (data, processed_data, DEM, LULC)
├── pipeline_v2       (v2 flood early warning)
├── preprocessing
├── models
├── forecasting
├── gis
├── what_if
├── notebooks
├── tests
└── docs
```

---

<!-- v2-start -->
## Flood Early Warning v2 (trained on real flood events)

**Question:** Will a flood start in this district today or in the next 7 days?

| | |
|---|---|
| Labels | India Flood Inventory v3 (real reported flood events): 129 real flood starts in the test years |
| Inputs | IMD gridded daily rainfall (0.25 deg), NASA POWER weather, season, elevation, distance to river |
| Not used | district ID and land-cover class code (they are labels, not measurements, and let models memorise districts) |
| Split | train 2000-2016, test 2017-2023 (time split: the model never sees the future) |
| Alerts | Warning = top 3% of district-days (levels set on 2013-2016) |

**Results on the unseen years 2017-2023** (every model raises the same number of alerts):

| Model | ROC-AUC | PR-AUC | Lift over random | Real floods warned |
|---|---|---|---|---|
| Rainfall rule (baseline) | 0.668 | 0.0252 | 1.9x | 32% |
| Logistic Regression | 0.782 | 0.0498 | 3.7x | 36% |
| Random Forest | 0.715 | 0.0257 | 1.9x | 12% |
| XGBoost | 0.706 | 0.0251 | 1.9x | 13% |

Selected model: **Logistic Regression**: warns of **36%** of real flood starts
on the day or up to 7 days before (simple rainfall rule: 32%), with about 11 alert days per district per year.

Floods are rare (about 1.4% of district-days), so PR-AUC values are low in absolute terms;
the lift column shows how much better than random alerts each model is. Logistic Regression beat the tree models
because only 129 test events (and a few hundred training events) exist: the simpler model generalises better.
Three problem set-ups were compared first (`pipeline_v2/s04_experiments.py`, results in `models/v2_reports/experiments.csv`).

**Dashboard:** the "Flood Early Warning v2" panel shows the warning level of every district for any past date,
the main reasons for each warning, the last 90 days of rainfall against the probability, and a what-if (more rain).
API: `/api/v2/warnings`, `/api/v2/district/{name}`, `/api/v2/what-if`, `/api/v2/model-report`.

**Run again:** `python -m pipeline_v2.run_all` (download, dataset, model, risk map).
All data is in `Datasets/` (see `Datasets/README.md`).
<!-- v2-end -->

## Datasets

The project uses the following data sources:

* IMD rainfall data
* CWC water-level data
* NASA POWER weather data
* GADM administrative boundaries
* ESA WorldCover 10 m LULC
* Copernicus DEM 30 m
* River/drainage network data
* India flood inventory

The available datasets have different temporal and spatial coverage.

---

## Machine Learning

The main classification model is a Random Forest classifier.

The classification target is:

```text
high_water_event
```

The target is derived from the available water-level observations and represents a **high-water proxy**, not confirmed flood ground truth.

### Model

```text
models/flood_guard_random_forest.pkl
```

### Prediction API

```text
GET  /api/predict/latest
POST /api/predict
```

The prediction endpoint returns:

* Prediction label
* Flood probability
* Risk level
* Reference date
* Model input information

---

## Water-Level Forecasting

The forecasting module uses a Random Forest regression model with lag, rolling, change, and calendar-based time-series features.

### Main forecasting script

```text
forecasting/05_water_level_forecast.py
```

### Outputs

```text
forecasting/water_level_forecast.csv
forecasting/next_water_level_forecast.csv
```

The current pipeline:

* Creates a continuous daily timeline
* Handles missing observations for model features
* Detects extreme observed anomalies for modelling
* Creates lag and rolling features
* Uses chronological train/test splitting
* Applies forecast change bounds
* Produces historical predictions
* Produces a next-day forecast

The forecasting component is an academic prototype and is not a live operational flood-warning system.

---

## GIS Risk Mapping

The GIS module combines multiple spatial indicators, including:

* Elevation
* Distance to river/drainage
* Historical flood exposure
* LULC
* Derived risk components

Current coverage:

```text
30 Odisha districts
```

Current base spatial-risk distribution:

```text
High       : 20
Moderate   : 8
Very High  : 2
Low        : 0
```

These are project-defined spatial risk categories and are not official government flood probabilities.

---

## What-If Analysis

Flood Guard AI supports predefined and dynamic What-If scenarios.

### Predefined scenarios

```text
Base
Rainfall +25%
Water Level +0.5m
Drainage Effectiveness -15%
Combined Stress Scenario
```

### Dynamic scenario inputs

```text
Rainfall Change (%)
Water-Level Change (m)
Drainage Effectiveness Change (%)
```

### Dynamic API

```text
POST /api/what-if/analyze
```

The dynamic analysis calculates adjusted district-level spatial risk using project-defined mathematical risk adjustments.

These scenarios are intended for academic decision-support analysis and are **not calibrated hydraulic or hydrodynamic simulations**.

---

## FastAPI Backend

Backend URL:

```text
http://127.0.0.1:8000
```

### API Endpoints

```text
GET  /
GET  /api/health

GET  /api/districts
GET  /api/risk
GET  /api/map

GET  /api/forecast

GET  /api/what-if
POST /api/what-if/analyze

GET  /api/predict/latest
POST /api/predict
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

## React Frontend

Frontend URL:

```text
http://localhost:5173
```

The dashboard contains:

* Risk summary cards
* Odisha spatial risk map
* District Explorer
* AI Flood Prediction
* Water-Level Forecast
* Dynamic What-If Analysis
* Predefined What-If scenarios
* Risk distribution
* District risk table
* Model and data status

---

## How to Run

### Option 1: Using Batch Files

From:

```text
D:\Flood_Ai
```

Run:

```text
run_backend.bat
```

and:

```text
run_frontend.bat
```

Keep both processes running while using the dashboard.

### Option 2: Backend Manually

Open PowerShell:

```powershell
cd D:\Flood_Ai
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Frontend Manually

Open a second PowerShell:

```powershell
cd D:\Flood_Ai\frontend
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

## Production Build

To create the React production build:

```powershell
cd D:\Flood_Ai\frontend
npm run build
```

The frontend project is configured for a production build through Vite.

---

## Current Functional Status

The following core components are integrated:

```text
Dataset Processing        ✅
Machine Learning          ✅
Water-Level Forecasting   ✅
GIS Risk Mapping          ✅
Dynamic What-If Analysis  ✅
FastAPI Backend           ✅
React Frontend            ✅
Frontend ↔ Backend        ✅
```

The project is currently in the **testing, documentation, cleanup, and final presentation-polish stage**.

---

## Limitations

This is an academic research and decision-support prototype.

Important limitations include:

1. The datasets do not all cover the same historical period.
2. The ML target is a high-water proxy rather than confirmed flood ground truth.
3. Some GIS calculations use representative spatial points or derived indicators.
4. The What-If component uses project-defined mathematical risk adjustments.
5. Water-level forecasting is based on the available historical dataset and is not a live 2026 operational forecast.
6. The system does not provide official real-time government flood warnings.
7. Model results depend on the quality, coverage, and limitations of the available datasets.

---

## Future Scope

* Longer overlapping historical datasets
* Official flood-event labels
* Real-time rainfall and river-level data
* Improved hydrological forecasting
* High-resolution flood-depth modelling
* Satellite-based flood detection
* Automated emergency alerts
* Evacuation-route optimization
* Cloud deployment
* Mobile application integration

---

## Academic Disclaimer

Flood Guard AI is an academic MCA project and research prototype. Its risk scores, forecasts, machine-learning predictions, and What-If outputs are intended for academic analysis, demonstration, and decision-support research. They should not be treated as official emergency warnings, government advisories, calibrated operational flood probabilities, or a replacement for authoritative disaster-management systems.
