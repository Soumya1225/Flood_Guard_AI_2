# Flood Guard AI – Project Structure

AI-Based Urban Flood Forecasting, Impact Mapping and What-If Emergency Decision Support for Odisha.

## How to run

```
Start-Process .\run_backend.bat; Start-Sleep 8; Start-Process .\run_frontend.bat; Start-Sleep 15; Start-Process "http://127.0.0.1:5174"
```

- Backend (FastAPI): http://127.0.0.1:8001 (API docs: http://127.0.0.1:8001/docs)
- Frontend (React dashboard): http://127.0.0.1:5174

## Folder overview (in the order the project works)

| Step | Folder / file | What it contains | Output |
|---|---|---|---|
| 0 | `Datasets/data/` | Raw data: IMD rainfall, CWC river water level, weather, Odisha boundaries, river network, India Flood Inventory, DEM (elevation), land cover, WorldPop / Census population | – |
| 0 | `download_dem.py`, `download_lulc.py` | Download the elevation (DEM) and land-cover tiles | `Datasets/data/dem`, `Datasets/data/lulc` |
| 1 | `notebooks/scripts/01_validate_datasets.py` | Checks that every dataset is present and readable | report |
| 2 | `preprocessing/01_preprocess_all.py` | Cleans rainfall, water level, weather and flood data | `Datasets/processed_data/*_processed.csv` |
| 3 | `preprocessing/02_build_features.py` | Builds model features (water-level means, changes, weather, date) | `Datasets/processed_data/` |
| 4 | `preprocessing/03_create_ml_target.py` | Creates the target `high_water_event` | `Datasets/processed_data/ml_training_data.csv` |
| 5 | `preprocessing/04_build_city_population.py`, `05_build_population_exposure.py` | Town populations and WorldPop population per district | `Datasets/processed_data/population_*.csv` |
| 6 | `models/04_train_model.py` | Trains the **Random Forest Classifier** (flood prediction) | `models/flood_guard_random_forest.pkl` |
| 7 | `forecasting/05_water_level_forecast.py` | **Random Forest Regressor** – 7-day water-level forecast | `forecasting/*.csv` |
| 8 | `gis/06_build_risk_map.py` | **Spatial risk index** for all 30 districts | `gis/odisha_spatial_risk.csv / .gpkg` |
| 9 | `what_if/07_what_if_analysis.py` | Predefined **what-if scenarios** | `what_if/what_if_results.csv`, `what_if_summary.csv` |
| 10 | `backend/main.py` | **FastAPI** server: loads the models and results, serves all dashboard APIs (prediction, forecast, risk map, impact, weather, emergency GIS, what-if, history) | API on port 8001 |
| 10 | `backend/flood_v2_api.py` | Extra v2 flood-warning API (model trained on real flood events, 2000–2023) | `/api/v2/...` |
| 10 | `backend/live_forecast_api.py` | **Live river outlook**: GloFAS river discharge + rain forecast → daily risk per district | `/api/live/...`, cache `Datasets/processed_data/glofas_district_reference.json` |
| 10 | `backend/live_ml_api.py` | **Live AI prediction**: v2 Logistic Regression run on live Open-Meteo rain and weather | `/api/live-ml/...` |
| 10 | `backend/response_api.py` | **Response plan**: response level, relief needs, towns, contacts, warnings, checklist | `/api/response/...` |
| 10 | `Datasets/processed_data/osm_cache/` | Cached OpenStreetMap results for Emergency GIS (30 days) | cache |
| 11 | `database/flood_guard.db` | **SQLite** database with saved predictions, alerts, scenarios and reports | history |
| 12 | `frontend/` | **React** dashboard (Leaflet maps, Recharts charts) with a sidebar and 11 pages | http://127.0.0.1:5174 |

### Experiment (version 2)

| Folder | What it contains |
|---|---|
| `pipeline_v2/` | v2 pipeline: download 24 years of IMD rainfall + NASA POWER weather (`s01`), build district-day dataset (`s02`), train and compare Logistic Regression / Random Forest / XGBoost (`s03`, `s04`) |
| `Datasets/data/v2/` | v2 raw data (IMD gridded rainfall 2000–2023, NASA POWER weather, India Flood Inventory CSV) |
| `models/flood_onset_model_v2.pkl`, `models/v2_reports/` | v2 model and its test results |
| `tests/` | Test for the v2 pipeline |

## Frontend files (`frontend/src`)

| File | Purpose |
|---|---|
| `main.jsx` | Starts the React app |
| `App.jsx` | Main dashboard: loads data from the backend, risk map, district explorer, AI prediction, forecasts, what-if, model performance, data quality |
| `Sidebar.jsx`, `Pages.css` | Left sidebar and page switching |
| `ProjectOverview.jsx` | Project Overview page (what, how, methods, results) |
| `ExtendedIntelligence.jsx` | Situation summary, 1/3/7-day risk, weather, affected districts, hotspots, population impact, cities, rainfall anomaly, scenarios, history |
| `EmergencyGISIntelligence.jsx` | Emergency GIS: infrastructure, shelters, roads, evacuation, inundation screening |
| `ModelsUsed.jsx` | Models & Methods Used section |
| `FloodWarningV2.jsx` | Past-event replay of the v2 flood early-warning model |
| `LiveFloodForecast.jsx` | Live GloFAS river outlook (Flood Forecast page) |
| `LiveMLPrediction.jsx` | Live AI flood prediction for all districts (AI Prediction page) |
| `ResponsePlan.jsx` | Flood response plan (Emergency Response page) |
| `App.css`, `index.css` | Styles |

## Dashboard pages

| Sidebar page | Main contents |
|---|---|
| Project Overview | What the project does, how it works, which method is used for what, tested results |
| 1. Overview | Alert level, AI situation summary, KPIs, unified risk & alert engine |
| 1. Risk Map | Odisha spatial risk map, district explorer, district comparison, risk distribution |
| 2. Flood Forecast | Live 7-day river outlook (GloFAS), all-district outlook, model reference forecasts |
| 2. Weather & Rainfall | Live weather, 7-day weather forecast, historical weather, rainfall anomaly, area alert |
| 3. AI Prediction | Live AI prediction (Logistic Regression v2), past-event replay, Random Forest reference, explainability |
| 3. Models & Performance | Models & methods used, accuracy, confusion matrix, feature importance |
| 4. Impact & Spread | Affected districts, hotspots, population at risk, affected cities, city priority |
| 4. Emergency Response | Flood response plan, emergency GIS intelligence, emergency action panel |
| 4. What-If Analysis | Dynamic what-if, scenario comparison, scenario explorer |
| 5. Data & History | Data quality, feature readiness, project status, saved history |

## Not part of the project folder

Old backups, patch scripts, logs and update zips were moved to
`D:\Final_Projects\Flood_Guard_AI_archive` (kept, not deleted).
