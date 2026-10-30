import { useEffect, useMemo, useState } from "react";

import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
} from "react-leaflet";

import L from "leaflet";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

import "leaflet/dist/leaflet.css";
import "./App.css";
import ExtendedIntelligence from "./ExtendedIntelligence";
import EmergencyGISIntelligence from "./EmergencyGISIntelligence";
import FloodWarningV2 from "./FloodWarningV2";
import ModelsUsed from "./ModelsUsed";
import Sidebar, { PAGES } from "./Sidebar";
import ProjectOverview from "./ProjectOverview";
import LiveFloodForecast from "./LiveFloodForecast";
import LiveMLPrediction from "./LiveMLPrediction";
import ResponsePlan from "./ResponsePlan";
import "./Pages.css";


const API_BASE = "/api";


const riskColors = {
  Low: "#22c55e",
  Moderate: "#eab308",
  High: "#f97316",
  "Very High": "#dc2626",
};


function round2(value) {
  const n = Number(value);
  return Number.isFinite(n) ? String(Math.round(n * 100) / 100) : String(value ?? "");
}

function getRiskColor(riskLevel) {
  return (
    riskColors[riskLevel] ||
    "#94a3b8"
  );
}


function getTrendColor(trend) {

  if (trend === "Rising") {
    return "#dc2626";
  }

  if (trend === "Falling") {
    return "#2563eb";
  }

  return "#64748b";
}


function formatNumber(value, digits = 3) {

  return typeof value === "number"
    ? value.toFixed(digits)
    : "N/A";
}


function getMapDistrictName(feature) {

  const properties =
    feature?.properties || {};

  return (
    properties.NAME_2 ||
    properties.district ||
    properties.DISTRICT ||
    properties.name ||
    "Unknown District"
  );
}


// ============================================================
// MAP CONTROLLER
// ============================================================

function MapController({ geojson }) {

  const map = useMap();

  useEffect(() => {

    if (
      geojson &&
      geojson.features &&
      geojson.features.length > 0
    ) {

      const layer =
        L.geoJSON(geojson);

      const bounds =
        layer.getBounds();

      if (bounds.isValid()) {

        map.fitBounds(
          bounds,
          {
            padding: [20, 20],
          }
        );
      }
    }

  }, [geojson, map]);

  useEffect(() => {
    const refit = () => {
      setTimeout(() => {
        map.invalidateSize();
        if (geojson && geojson.features && geojson.features.length > 0) {
          const b = L.geoJSON(geojson).getBounds();
          if (b.isValid()) map.fitBounds(b, { padding: [20, 20] });
        }
      }, 60);
    };
    window.addEventListener("fg-page-change", refit);
    return () => window.removeEventListener("fg-page-change", refit);
  }, [geojson, map]);

  return null;
}


// ============================================================
// RISK BADGE
// ============================================================

function RiskBadge({ level }) {

  return (
    <span
      className="risk-badge"
      style={{
        backgroundColor:
          getRiskColor(level),
      }}
    >
      {level || "N/A"}
    </span>
  );
}


// ============================================================
// METRIC CARD
// ============================================================

function MetricCard({
  label,
  value,
  valueColor,
}) {

  return (
    <div className="scenario-card">

      <span>
        {label}
      </span>

      <strong
        style={{
          color:
            valueColor ||
            undefined,
        }}
      >
        {value}
      </strong>

    </div>
  );
}


// ============================================================
// FORECAST DATA
// ============================================================

function prepareForecastData(records) {

  if (!Array.isArray(records)) {
    return [];
  }

  return records
    .map((record) => {

      const date =
        record.date ??
        record.Date ??
        record.forecast_date ??
        record.timestamp ??
        "";

      const actual =
        record.actual_water_level ??
        record.actual ??
        record.water_level_mean_m ??
        record.actual_water_level_mean_m ??
        null;

      const predicted =
        record.predicted_water_level_m ??
        record.predicted_water_level ??
        record.predicted ??
        record.prediction ??
        record.forecast ??
        record.y_pred ??
        null;

      return {

        date:
          String(date).slice(
            0,
            10
          ),

        actual:
          typeof actual === "number"
            ? actual
            : null,

        predicted:
          typeof predicted === "number"
            ? predicted
            : null,

      };
    })
    .filter(
      (item) =>
        item.date &&
        (
          item.actual !== null ||
          item.predicted !== null
        )
    );
}


function prepareFutureForecastData(records) {

  if (!Array.isArray(records)) {
    return [];
  }

  return records
    .map((record) => {

      const forecastDay =
        record.forecast_day ??
        null;

      const date =
        record.forecast_date ??
        record.date ??
        record.Date ??
        "";

      const predicted =
        record.predicted_water_level_m ??
        record.predicted_water_level ??
        record.predicted ??
        record.forecast ??
        null;

      const change =
        record.prediction_change_m ??
        null;

      const trend =
        record.trend ??
        "Stable";

      const lower =
        record.lower_estimate_m ??
        null;

      const upper =
        record.upper_estimate_m ??
        null;

      return {

        forecastDay:
          typeof forecastDay === "number"
            ? forecastDay
            : null,

        date:
          String(date).slice(
            0,
            10
          ),

        predicted:
          typeof predicted === "number"
            ? predicted
            : null,

        change:
          typeof change === "number"
            ? change
            : null,

        trend:
          String(trend),

        lower:
          typeof lower === "number"
            ? lower
            : null,

        upper:
          typeof upper === "number"
            ? upper
            : null,

      };
    })
    .filter(
      (item) =>
        item.date &&
        item.predicted !== null
    );
}


// ============================================================
// MAIN APP
// ============================================================

function App() {

  // ==========================================================
  // GENERAL STATE
  // ==========================================================

  const [mapData, setMapData] =
    useState(null);

  const [riskData, setRiskData] =
    useState(null);

  const [forecastData, setForecastData] =
    useState([]);

  const [nextForecast, setNextForecast] =
    useState([]);

  const [whatIfData, setWhatIfData] =
    useState(null);

  const [selectedScenario, setSelectedScenario] =
    useState("Base");

  const [selectedDistrictId, setSelectedDistrictId] =
    useState("");

  const [page, setPage] = useState(() => {
    const h = (window.location.hash || "").replace("#", "");
    return PAGES.some((p) => p.id === h) ? h : "about";
  });

  useEffect(() => {
    if (window.location.hash !== "#" + page) {
      window.history.replaceState(null, "", "#" + page);
    }
    window.scrollTo(0, 0);
    window.dispatchEvent(new Event("fg-page-change"));
    const t = setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
    return () => clearTimeout(t);
  }, [page]);

  const [comparisonDistrictId, setComparisonDistrictId] =
    useState("");

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");


  // ==========================================================
  // FLOOD PREDICTION
  // ==========================================================

  const [predictionInput, setPredictionInput] =
    useState({
      water_level_mean_m: "",
      water_level_max_m: "",
      water_level_min_m: "",
      water_level_change_m: "",
      temperature: "",
      humidity: "",
      wind_speed: "",
      pressure: "",
      water_level_3day_mean_m: "",
      water_level_7day_mean_m: "",
      month: "",
      day_of_year: "",
    });

  const [predictionData, setPredictionData] =
    useState(null);

  const [predictionLoading, setPredictionLoading] =
    useState(false);

  const [predictionError, setPredictionError] =
    useState("");


  // ==========================================================
  // DYNAMIC WHAT-IF
  // ==========================================================

  const [whatIfInput, setWhatIfInput] =
    useState({
      rainfall_change_percent: 25,
      water_level_change_m: 0.5,
      drainage_effectiveness_change_percent: -15,
    });

  const [dynamicWhatIfData, setDynamicWhatIfData] =
    useState(null);

  const [whatIfLoading, setWhatIfLoading] =
    useState(false);

  const [dynamicWhatIfError, setDynamicWhatIfError] =
    useState("");

  const [whatIfMapActive, setWhatIfMapActive] =
    useState(false);

  const [mapRefreshKey, setMapRefreshKey] =
    useState(0);


  // ==========================================================
  // FEATURE DATA
  // ==========================================================

  const [decisionSupport, setDecisionSupport] =
    useState(null);

  const [explainability, setExplainability] =
    useState(null);

  const [emergencyActions, setEmergencyActions] =
    useState(null);

  const [modelPerformance, setModelPerformance] =
    useState(null);

  const [dataQuality, setDataQuality] =
    useState(null);


  // ==========================================================
  // LOAD DASHBOARD DATA
  // ==========================================================

  useEffect(() => {

    const loadDashboard =
      async () => {

        try {

          const endpointNames = [
            "map",
            "risk",
            "forecast",
            "what-if",
            "decision-support",
            "explain/latest",
            "emergency-actions",
            "model-performance",
            "data-quality",
            "predict/latest",
          ];

          const responses =
            await Promise.all(
              endpointNames.map(
                (endpoint) =>
                  fetch(
                    `${API_BASE}/${endpoint}`
                  )
              )
            );

          const jsonResults =
            await Promise.all(
              responses.map(
                async (response) => {

                  try {
                    return await response.json();
                  } catch {
                    return {};
                  }

                }
              )
            );


          const mapResponse =
            responses[0];

          const riskResponse =
            responses[1];

          const forecastResponse =
            responses[2];

          const whatIfResponse =
            responses[3];

          const mapJson =
            jsonResults[0];

          const riskJson =
            jsonResults[1];

          const forecastJson =
            jsonResults[2];

          const whatIfJson =
            jsonResults[3];

          if (
            !mapResponse.ok ||
            !riskResponse.ok ||
            !forecastResponse.ok ||
            !whatIfResponse.ok
          ) {

            throw new Error(
              "Failed to load dashboard data."
            );
          }


          if (
            mapJson.status !==
              "success" ||
            riskJson.status !==
              "success" ||
            forecastJson.status !==
              "success" ||
            whatIfJson.status !==
              "success"
          ) {

            throw new Error(
              "Backend returned an error."
            );
          }


          setMapData(
            mapJson.geojson
          );

          setRiskData(
            riskJson
          );

          setForecastData(
            prepareForecastData(
              forecastJson.historical_forecast
            )
          );

          setNextForecast(
            forecastJson.next_forecast ||
            []
          );

          setWhatIfData(
            whatIfJson
          );


          if (
            Array.isArray(
              riskJson.districts
            ) &&
            riskJson.districts.length > 0
          ) {

            setSelectedDistrictId(
              riskJson.districts[0]
                .district_id
            );

            if (
              riskJson.districts.length >
              1
            ) {

              setComparisonDistrictId(
                riskJson.districts[1]
                  .district_id
              );
            }
          }


          if (
            Array.isArray(
              whatIfJson.available_scenarios
            ) &&
            whatIfJson.available_scenarios.length >
            0
          ) {

            setSelectedScenario(
              whatIfJson.available_scenarios.includes(
                "Base"
              )
                ? "Base"
                : whatIfJson
                    .available_scenarios[0]
            );
          }


          // --------------------------------------------------
          // FEATURE 1
          // --------------------------------------------------

          if (
            responses[4].ok &&
            jsonResults[4].status ===
              "success"
          ) {

            setDecisionSupport(
              jsonResults[4]
            );
          }


          // --------------------------------------------------
          // FEATURE 4
          // --------------------------------------------------

          if (
            responses[5].ok &&
            jsonResults[5].status ===
              "success"
          ) {

            setExplainability(
              jsonResults[5]
            );
          }


          // --------------------------------------------------
          // FEATURE 5
          // --------------------------------------------------

          if (
            responses[6].ok &&
            jsonResults[6].status ===
              "success"
          ) {

            setEmergencyActions(
              jsonResults[6]
            );
          }


          // --------------------------------------------------
          // FEATURE 8
          // --------------------------------------------------

          if (
            responses[7].ok &&
            jsonResults[7].status ===
              "success"
          ) {

            setModelPerformance(
              jsonResults[7]
            );
          }


          // --------------------------------------------------
          // FEATURE 10
          // --------------------------------------------------

          if (
            responses[8].ok &&
            jsonResults[8].status ===
              "success"
          ) {

            setDataQuality(
              jsonResults[8]
            );
          }


          // --------------------------------------------------
          // LATEST PREDICTION
          // --------------------------------------------------

          if (
            responses[9].ok &&
            jsonResults[9].status ===
              "success"
          ) {

            const latestPrediction =
              jsonResults[9];

            setPredictionData(
              latestPrediction
            );

            if (
              latestPrediction.features_used
            ) {

              const features =
                latestPrediction.features_used;

              setPredictionInput({

                water_level_mean_m:
                  round2(
                    features
                      .water_level_mean_m
                  ),

                water_level_max_m:
                  round2(
                    features
                      .water_level_max_m
                  ),

                water_level_min_m:
                  round2(
                    features
                      .water_level_min_m
                  ),

                water_level_change_m:
                  round2(
                    features
                      .water_level_change_m
                  ),

                temperature:
                  round2(
                    features.temperature
                  ),

                humidity:
                  round2(
                    features.humidity
                  ),

                wind_speed:
                  round2(
                    features.wind_speed
                  ),

                pressure:
                  round2(
                    features.pressure
                  ),

                water_level_3day_mean_m:
                  round2(
                    features
                      .water_level_3day_mean_m
                  ),

                water_level_7day_mean_m:
                  round2(
                    features
                      .water_level_7day_mean_m
                  ),

                month:
                  round2(
                    features.month
                  ),

                day_of_year:
                  round2(
                    features.day_of_year
                  ),

              });
            }
          }


          setLoading(false);

        } catch (err) {

          console.error(
            "Dashboard loading error:",
            err
          );

          setError(
            "Unable to load Flood Guard AI data. Make sure FastAPI is running."
          );

          setLoading(false);
        }
      };


    loadDashboard();

  }, []);


  // ==========================================================
  // FLOOD PREDICTION INPUT
  // ==========================================================

  const handlePredictionInputChange =
    (event) => {

      const {
        name,
        value,
      } = event.target;

      setPredictionInput(
        (previous) => ({
          ...previous,
          [name]: value,
        })
      );
    };


  // ==========================================================
  // MANUAL FLOOD PREDICTION
  // ==========================================================

  const handleFloodPrediction =
    async (event) => {

      event.preventDefault();

      setPredictionError("");

      const hasEmptyValue =
        Object.values(
          predictionInput
        ).some(
          (value) =>
            value === ""
        );

      if (hasEmptyValue) {

        setPredictionError(
          "Please enter all flood prediction inputs."
        );

        return;
      }


      try {

        setPredictionLoading(
          true
        );


        const payload = {

          water_level_mean_m:
            Number(
              predictionInput
                .water_level_mean_m
            ),

          water_level_max_m:
            Number(
              predictionInput
                .water_level_max_m
            ),

          water_level_min_m:
            Number(
              predictionInput
                .water_level_min_m
            ),

          water_level_change_m:
            Number(
              predictionInput
                .water_level_change_m
            ),

          temperature:
            Number(
              predictionInput
                .temperature
            ),

          humidity:
            Number(
              predictionInput
                .humidity
            ),

          wind_speed:
            Number(
              predictionInput
                .wind_speed
            ),

          pressure:
            Number(
              predictionInput
                .pressure
            ),

          water_level_3day_mean_m:
            Number(
              predictionInput
                .water_level_3day_mean_m
            ),

          water_level_7day_mean_m:
            Number(
              predictionInput
                .water_level_7day_mean_m
            ),

          month:
            Number(
              predictionInput
                .month
            ),

          day_of_year:
            Number(
              predictionInput
                .day_of_year
            ),

        };


        const response =
          await fetch(
            `${API_BASE}/predict`,
            {
              method:
                "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body:
                JSON.stringify(
                  payload
                ),
            }
          );


        const result =
          await response.json();


        if (!response.ok) {

          throw new Error(
            result.detail ||
              "Flood prediction request failed."
          );
        }


        if (
          result.status !==
          "success"
        ) {

          throw new Error(
            result.message ||
              "Flood prediction failed."
          );
        }


        setPredictionData(
          result
        );


        // Refresh explainability
        try {

          const explainResponse =
            await fetch(
              `${API_BASE}/explain`,
              {
                method:
                  "POST",

                headers: {
                  "Content-Type":
                    "application/json",
                },

                body:
                  JSON.stringify(
                    payload
                  ),
              }
            );


          const explainResult =
            await explainResponse.json();


          if (
            explainResponse.ok &&
            explainResult.status ===
              "success"
          ) {

            setExplainability(
              explainResult
            );
          }

        } catch (explainError) {

          console.error(
            "Explainability refresh error:",
            explainError
          );
        }

      } catch (err) {

        console.error(
          err
        );

        setPredictionError(
          err.message ||
            "Unable to generate flood prediction."
        );

      } finally {

        setPredictionLoading(
          false
        );
      }
    };


  // ==========================================================
  // LATEST AVAILABLE PREDICTION
  // ==========================================================

  const handleLatestAvailablePrediction =
    async () => {

      setPredictionError("");

      try {

        setPredictionLoading(
          true
        );


        const response =
          await fetch(
            `${API_BASE}/predict/latest`
          );


        const result =
          await response.json();


        if (!response.ok) {

          throw new Error(
            result.detail ||
              "Unable to load latest ML prediction."
          );
        }


        if (
          result.status !==
          "success"
        ) {

          throw new Error(
            result.message ||
              "Latest prediction failed."
          );
        }


        setPredictionData(
          result
        );


        if (
          result.features_used
        ) {

          const features =
            result.features_used;

          setPredictionInput({

            water_level_mean_m:
              String(
                features
                  .water_level_mean_m
              ),

            water_level_max_m:
              String(
                features
                  .water_level_max_m
              ),

            water_level_min_m:
              String(
                features
                  .water_level_min_m
              ),

            water_level_change_m:
              String(
                features
                  .water_level_change_m
              ),

            temperature:
              String(
                features.temperature
              ),

            humidity:
              String(
                features.humidity
              ),

            wind_speed:
              String(
                features.wind_speed
              ),

            pressure:
              String(
                features.pressure
              ),

            water_level_3day_mean_m:
              String(
                features
                  .water_level_3day_mean_m
              ),

            water_level_7day_mean_m:
              String(
                features
                  .water_level_7day_mean_m
              ),

            month:
              String(
                features.month
              ),

            day_of_year:
              String(
                features.day_of_year
              ),

          });
        }


        try {

          const explainResponse =
            await fetch(
              `${API_BASE}/explain/latest`
            );

          const explainResult =
            await explainResponse.json();

          if (
            explainResponse.ok &&
            explainResult.status ===
              "success"
          ) {

            setExplainability(
              explainResult
            );
          }

        } catch (explainError) {

          console.error(
            "Explainability refresh error:",
            explainError
          );
        }

      } catch (err) {

        console.error(
          err
        );

        setPredictionError(
          err.message ||
            "Unable to load latest prediction."
        );

      } finally {

        setPredictionLoading(
          false
        );
      }
    };


  // ==========================================================
  // WHAT-IF INPUT
  // ==========================================================

  const handleWhatIfInputChange =
    (event) => {

      const {
        name,
        value,
      } = event.target;

      setWhatIfInput(
        (previous) => ({
          ...previous,
          [name]: value,
        })
      );
    };


  // ==========================================================
  // DYNAMIC WHAT-IF ANALYSIS
  // ==========================================================

  const handleDynamicWhatIf =
    async (event) => {

      event.preventDefault();

      setDynamicWhatIfError("");


      try {

        setWhatIfLoading(
          true
        );


        const payload = {

          rainfall_change_percent:
            Number(
              whatIfInput
                .rainfall_change_percent
            ),

          water_level_change_m:
            Number(
              whatIfInput
                .water_level_change_m
            ),

          drainage_effectiveness_change_percent:
            Number(
              whatIfInput
                .drainage_effectiveness_change_percent
            ),

        };


        const response =
          await fetch(
            `${API_BASE}/what-if/analyze`,
            {
              method:
                "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body:
                JSON.stringify(
                  payload
                ),
            }
          );


        const result =
          await response.json();


        if (!response.ok) {

          throw new Error(
            result.detail ||
              "What-If analysis request failed."
          );
        }


        if (
          result.status !==
          "success"
        ) {

          throw new Error(
            result.message ||
              "What-If analysis failed."
          );
        }


        setDynamicWhatIfData(
          result
        );


        setWhatIfMapActive(
          true
        );


        setMapRefreshKey(
          (previous) =>
            previous + 1
        );

      } catch (err) {

        console.error(
          err
        );

        setDynamicWhatIfError(
          err.message ||
            "Unable to run What-If analysis."
        );

      } finally {

        setWhatIfLoading(
          false
        );
      }
    };


  // ==========================================================
  // RESET WHAT-IF MAP
  // ==========================================================

  const handleResetWhatIfMap =
    () => {

      setWhatIfMapActive(
        false
      );

      setDynamicWhatIfData(
        null
      );

      setDynamicWhatIfError(
        ""
      );

      setMapRefreshKey(
        (previous) =>
          previous + 1
      );
    };


  // ==========================================================
  // DISTRICT DATA
  // ==========================================================

  const districts =
    riskData?.districts ||
    [];


  const distribution =
    riskData?.risk_distribution ||
    {};


  const highRiskCount =
    (distribution.High || 0) +
    (distribution["Very High"] || 0);


  const selectedDistrict =
    districts.find(
      (district) =>
        Number(
          district.district_id
        ) ===
        Number(
          selectedDistrictId
        )
    ) ||
    districts[0];


  const comparisonDistrict =
    districts.find(
      (district) =>
        Number(
          district.district_id
        ) ===
        Number(
          comparisonDistrictId
        )
    ) ||
    districts.find(
      (district) =>
        Number(
          district.district_id
        ) !==
        Number(
          selectedDistrictId
        )
    );


  // ==========================================================
  // FORECAST DATA
  // ==========================================================

  const futureForecastData =
    prepareFutureForecastData(
      nextForecast
    );


  const nextForecastRecord =
    futureForecastData[0] ||
    null;


  const nextForecastDate =
    nextForecastRecord?.date ||
    "N/A";


  const nextForecastValue =
    nextForecastRecord?.predicted ??
    null;


  const forecastTrend =
    nextForecastRecord?.trend ||
    "Stable";


  // ==========================================================
  // HISTORICAL TREND
  // ==========================================================

  const historicalActuals =
    useMemo(
      () =>
        forecastData
          .filter(
            (item) =>
              item.actual !== null
          )
          .map(
            (item) => ({
              date:
                item.date,
              water_level:
                item.actual,
            })
          ),
      [forecastData]
    );


  const historicalValues =
    historicalActuals.map(
      (item) =>
        item.water_level
    );


  const historicalAverage =
    historicalValues.length > 0
      ? historicalValues.reduce(
          (sum, value) =>
            sum + value,
          0
        ) /
        historicalValues.length
      : null;


  const historicalMinimum =
    historicalValues.length > 0
      ? Math.min(
          ...historicalValues
        )
      : null;


  const historicalMaximum =
    historicalValues.length > 0
      ? Math.max(
          ...historicalValues
        )
      : null;


  const historicalStart =
    historicalValues.length > 0
      ? historicalValues[0]
      : null;


  const historicalEnd =
    historicalValues.length > 0
      ? historicalValues[
          historicalValues.length - 1
        ]
      : null;


  const historicalChange =
    historicalStart !== null &&
    historicalEnd !== null
      ? historicalEnd -
        historicalStart
      : null;


  const historicalTrend =
    historicalChange !== null
      ? historicalChange > 2
        ? "Rising"
        : historicalChange < -2
        ? "Falling"
        : "Stable"
      : "Unavailable";


  // ==========================================================
  // PRECOMPUTED SCENARIOS
  // ==========================================================

  const availableScenarios =
    whatIfData?.available_scenarios ||
    [];


  const selectedScenarioResults =
    Array.isArray(
      whatIfData?.results
    )
      ? whatIfData.results.filter(
          (item) =>
            String(
              item.scenario
            ).toLowerCase() ===
            String(
              selectedScenario
            ).toLowerCase()
        )
      : [];


  const selectedScenarioSummary =
    Array.isArray(
      whatIfData?.summary
    )
      ? whatIfData.summary.find(
          (item) =>
            String(
              item.scenario
            ).toLowerCase() ===
            String(
              selectedScenario
            ).toLowerCase()
        )
      : null;


  // ==========================================================
  // DYNAMIC WHAT-IF MAP MATCHING
  // ==========================================================

  const getDynamicScenarioDistrict =
    (feature) => {

      if (
        !whatIfMapActive ||
        !dynamicWhatIfData ||
        !Array.isArray(
          dynamicWhatIfData.results
        )
      ) {
        return null;
      }


      const mapDistrict =
        getMapDistrictName(
          feature
        )
          .trim()
          .toLowerCase();


      return (
        dynamicWhatIfData.results.find(
          (item) =>
            String(
              item.NAME_2 ??
              item.district ??
              item.DISTRICT ??
              ""
            )
              .trim()
              .toLowerCase() ===
            mapDistrict
        ) ||
        null
      );
    };


  // ==========================================================
  // REPORT DOWNLOAD
  // ==========================================================

  const downloadRiskReport =
    () => {

      const lines = [];

      const addLine =
        (value = "") =>
          lines.push(
            String(value)
          );


      addLine(
        "FLOOD GUARD AI"
      );

      addLine(
        "AI-Based Urban Flood Forecasting & Risk Decision Support"
      );

      addLine(
        "============================================================"
      );

      addLine(
        `Generated: ${new Date().toLocaleString()}`
      );

      addLine();


      // ------------------------------------------------------
      // OVERVIEW
      // ------------------------------------------------------

      addLine(
        "1. RISK OVERVIEW"
      );

      addLine(
        `Total Districts: ${
          riskData?.district_count ??
          "N/A"
        }`
      );

      addLine(
        `Average Risk Score: ${
          formatNumber(
            riskData?.average_risk_score
          )
        }`
      );

      addLine(
        `Highest Risk Score: ${
          formatNumber(
            riskData?.maximum_risk_score
          )
        }`
      );

      addLine(
        `High / Very High Districts: ${
          highRiskCount
        }`
      );

      addLine();


      // ------------------------------------------------------
      // DECISION SUPPORT
      // ------------------------------------------------------

      if (decisionSupport) {

        addLine(
          "2. UNIFIED RISK & ALERT"
        );

        addLine(
          `Overall Status: ${
            decisionSupport.overall_status
          }`
        );

        addLine(
          `Alert Level: ${
            decisionSupport.alert_level
          }`
        );

        addLine(
          `Message: ${
            decisionSupport.message
          }`
        );

        addLine(
          "Priority Districts:"
        );


        (
          decisionSupport.priority_districts ||
          []
        ).forEach(
          (
            district,
            index
          ) => {

            addLine(
              `${index + 1}. ${
                district.district
              } | ${
                district.risk_level
              } | Score ${
                typeof district.risk_score ===
                "number"
                  ? district.risk_score.toFixed(
                      3
                    )
                  : "N/A"
              }`
            );
          }
        );

        addLine();
      }


      // ------------------------------------------------------
      // FLOOD PREDICTION
      // ------------------------------------------------------

      if (predictionData) {

        addLine(
          "3. AI FLOOD PREDICTION  -  HISTORICAL MODEL REFERENCE (2020)"
        );

        addLine(
          `Prediction: ${
            predictionData.prediction_label
          }`
        );

        addLine(
          `Flood Probability: ${
            Number(
              predictionData
                .flood_probability_percent
            ).toFixed(2)
          }%`
        );

        addLine(
          `Risk Level: ${
            predictionData.risk_level
          }`
        );

        addLine(
          `Reference Date: ${
            predictionData.reference_date ||
            "N/A"
          }`
        );

        addLine();
      }


      // ------------------------------------------------------
      // FORECAST
      // ------------------------------------------------------

      addLine(
        "4. 7-DAY FORECAST"
      );

      futureForecastData.forEach(
        (item) => {

          addLine(
            `Day ${
              item.forecastDay ??
              ""
            } | ${
              item.date
            } | ${
              item.predicted.toFixed(
                2
              )
            } m | ${
              item.change !== null
                ? item.change.toFixed(
                    2
                  )
                : "N/A"
            } m | ${
              item.trend
            }`
          );
        }
      );

      addLine();


      // ------------------------------------------------------
      // HISTORICAL TREND
      // ------------------------------------------------------

      addLine(
        "5. HISTORICAL TREND"
      );

      addLine(
        `Average: ${
          historicalAverage !== null
            ? historicalAverage.toFixed(
                2
              )
            : "N/A"
        } m`
      );

      addLine(
        `Minimum: ${
          historicalMinimum !== null
            ? historicalMinimum.toFixed(
                2
              )
            : "N/A"
        } m`
      );

      addLine(
        `Maximum: ${
          historicalMaximum !== null
            ? historicalMaximum.toFixed(
                2
              )
            : "N/A"
        } m`
      );

      addLine(
        `Overall Trend: ${
          historicalTrend
        }`
      );

      addLine();


      // ------------------------------------------------------
      // WHAT-IF
      // ------------------------------------------------------

      if (dynamicWhatIfData) {

        addLine(
          "6. ACTIVE WHAT-IF SCENARIO"
        );

        addLine(
          `Rainfall Change: ${
            dynamicWhatIfData
              .scenario
              ?.rainfall_change_percent
          }%`
        );

        addLine(
          `Water-Level Change: ${
            dynamicWhatIfData
              .scenario
              ?.water_level_change_m
          } m`
        );

        addLine(
          `Drainage Effectiveness Change: ${
            dynamicWhatIfData
              .scenario
              ?.drainage_effectiveness_change_percent
          }%`
        );

        addLine(
          `Average Scenario Risk: ${
            dynamicWhatIfData
              .summary
              ?.average_risk_score ??
            "N/A"
          }`
        );

        addLine(
          `Very High Districts: ${
            dynamicWhatIfData
              .summary
              ?.very_high_districts ??
            0
          }`
        );

        addLine();
      }


      // ------------------------------------------------------
      // DISTRICT
      // ------------------------------------------------------

      if (selectedDistrict) {

        addLine(
          "7. SELECTED DISTRICT"
        );

        addLine(
          `District: ${
            selectedDistrict.NAME_2
          }`
        );

        addLine(
          `Risk Level: ${
            selectedDistrict.risk_level
          }`
        );

        addLine(
          `Spatial Risk Score: ${
            formatNumber(
              selectedDistrict
                .spatial_risk_score
            )
          }`
        );

        addLine(
          `Historical Flood Events: ${
            selectedDistrict
              .historical_flood_events ??
            "N/A"
          }`
        );

        addLine();
      }


      // ------------------------------------------------------
      // MODEL
      // ------------------------------------------------------

      if (modelPerformance) {

        addLine(
          "8. MODEL PERFORMANCE"
        );

        addLine(
          `Model: ${
            modelPerformance.model_name ||
            "N/A"
          }`
        );

        if (
          modelPerformance
            .evaluation
            ?.available
        ) {

          addLine(
            `Accuracy: ${
              modelPerformance
                .evaluation
                .accuracy_percent
                .toFixed(2)
            }%`
          );

          addLine(
            `Precision: ${
              modelPerformance
                .evaluation
                .precision_percent !==
              null
                ? modelPerformance
                    .evaluation
                    .precision_percent
                    .toFixed(2)
                : "N/A"
            }%`
          );

          addLine(
            `Recall: ${
              modelPerformance
                .evaluation
                .recall_percent !==
              null
                ? modelPerformance
                    .evaluation
                    .recall_percent
                    .toFixed(2)
                : "N/A"
            }%`
          );

          addLine(
            `F1 Score: ${
              modelPerformance
                .evaluation
                .f1_percent !==
              null
                ? modelPerformance
                    .evaluation
                    .f1_percent
                    .toFixed(2)
                : "N/A"
            }%`
          );
        }

        addLine();
      }


      // ------------------------------------------------------
      // IMPORTANT NOTE
      // ------------------------------------------------------

      addLine(
        "9. PROJECT NOTE"
      );

      addLine(
        "Flood Guard AI is an academic decision-support system."
      );

      addLine(
        "Its results are not official real-time flood warnings."
      );


      const blob =
        new Blob(
          [
            lines.join(
              "\n"
            ),
          ],
          {
            type:
              "text/plain;charset=utf-8",
          }
        );


      const url =
        URL.createObjectURL(
          blob
        );


      const link =
        document.createElement(
          "a"
        );


      link.href =
        url;


      link.download =
        `Flood_Guard_AI_Risk_Report_${new Date()
          .toISOString()
          .slice(
            0,
            10
          )}.txt`;


      document.body.appendChild(
        link
      );


      link.click();

      link.remove();

      URL.revokeObjectURL(
        url
      );
    };


  // ==========================================================
  // LOADING
  // ==========================================================

  if (loading) {

    return (
      <div className="loading-screen">

        <div className="loading-card">

          <h1>
            Flood Guard AI
          </h1>

          <p>
            Loading dashboard data...
          </p>

        </div>

      </div>
    );
  }


  // ==========================================================
  // ERROR
  // ==========================================================

  if (error) {

    return (
      <div className="loading-screen">

        <div className="error-card">

          <h1>
            Flood Guard AI
          </h1>

          <p>
            {error}
          </p>

          <p>
            Backend:{" "}
            <strong>
              http://127.0.0.1:8000
            </strong>
          </p>

        </div>

      </div>
    );
  }


  // ==========================================================
  // RETURN DASHBOARD
  // ==========================================================

  return (

    <div className="app">


      {/* ====================================================
          HEADER
      ==================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-icon">
            FG
          </div>

          <div>

            <h1>
              Flood Guard AI
            </h1>

            <p>
              AI-Based Urban Flood Forecasting &
              Risk Decision Support
            </p>

          </div>

        </div>


        <div className="connection-status">

          <span className="status-dot"></span>

          Backend Connected

        </div>

      </header>


      <style>{`.fg-main [data-page]:not([data-page="${page}"]) { display: none !important; } .fg-main [data-hide-on~="${page}"] { display: none !important; }`}</style>

      <div className="fg-layout">

      <Sidebar
        page={page}
        setPage={setPage}
        districts={districts}
        selectedDistrictId={selectedDistrict?.district_id ?? ""}
        setSelectedDistrictId={setSelectedDistrictId}
      />

      <main className="dashboard fg-main">

        <div data-hide-on="about" className="fg-page-title">

          <h2>{PAGES.find((p) => p.id === page)?.label}</h2>
          <p>{PAGES.find((p) => p.id === page)?.hint}</p>
        </div>

        <div data-page="about">
          <ProjectOverview
            riskData={riskData}
            highRiskCount={highRiskCount}
            decisionSupport={decisionSupport}
            modelPerformance={modelPerformance}
            onOpen={setPage}
          />
        </div>

        <div data-page="forecast">
          <LiveFloodForecast
            selectedDistrictName={selectedDistrict?.NAME_2 || ""}
          />
        </div>

        <ExtendedIntelligence
          selectedDistrictName={
            selectedDistrict?.NAME_2 || ""
          }
        />
        <div data-page="emergency">
          <ResponsePlan
            selectedDistrictName={selectedDistrict?.NAME_2 || ""}
          />
        </div>
        <div data-page="emergency">
          <EmergencyGISIntelligence
            selectedDistrictName={
              selectedDistrict?.NAME_2 || ""
            }
          />
        </div>
        <div data-page="prediction">
          <LiveMLPrediction
            selectedDistrictName={selectedDistrict?.NAME_2 || ""}
          />
        </div>
        <div data-page="prediction">
          <FloodWarningV2 />
        </div>
        <div data-page="models">
          <ModelsUsed />
        </div>


        {/* ====================================================
            PAGE HEADING
        ==================================================== */}

        <section data-page="dashboard" className="page-heading">

          <div>

            <h2>
              Odisha Flood Risk Dashboard
            </h2>

            <p>
              District-level spatial flood-risk
              monitoring and AI-assisted decision support.
            </p>

          </div>


          <button
            type="button"
            className="prediction-button"
            onClick={
              downloadRiskReport
            }
          >
            Download Risk Report
          </button>

        </section>


        {/* ====================================================
            SUMMARY
        ==================================================== */}

        <section data-page="dashboard" className="summary-grid">


          <div className="summary-card">

            <span>
              Total Districts
            </span>

            <strong>
              {riskData?.district_count || 0}
            </strong>

          </div>


          <div className="summary-card">

            <span>
              Average Risk Score
            </span>

            <strong>

              {formatNumber(
                riskData?.average_risk_score
              )}

            </strong>

          </div>


          <div className="summary-card">

            <span>
              Highest Risk Score
            </span>

            <strong>

              {formatNumber(
                riskData?.maximum_risk_score
              )}

            </strong>

          </div>


          <div className="summary-card">

            <span>
              High / Very High
            </span>

            <strong>
              {highRiskCount}
            </strong>

          </div>


        </section>


        {/* ====================================================
            FEATURE 1 - UNIFIED RISK & ALERT
        ==================================================== */}

        {decisionSupport && (

          <section data-page="dashboard" className="status-section">

            <div className="panel status-panel">


              <div className="status-header">

                <h3>
                  Unified Risk & Alert Engine
                </h3>


                <span
                  className="prototype-badge"
                  style={{
                    backgroundColor:
                      getRiskColor(
                        decisionSupport
                          .overall_status
                      ),
                  }}
                >
                  {
                    decisionSupport
                      .alert_level
                  }
                </span>

              </div>


              <div className="scenario-grid">

                <MetricCard
                  label="Overall Status"
                  value={
                    decisionSupport
                      .overall_status
                  }
                />

                <MetricCard
                  label="Alert Level"
                  value={
                    decisionSupport
                      .alert_level
                  }
                />

                <MetricCard
                  label="Priority Districts"
                  value={
                    decisionSupport
                      .priority_district_count ??
                    0
                  }
                />

                <MetricCard
                  label="Forecast Trend"
                  value={
                    decisionSupport
                      .forecast
                      ?.trend ||
                    "Unavailable"
                  }
                  valueColor={
                    getTrendColor(
                      decisionSupport
                        .forecast
                        ?.trend
                    )
                  }
                />

              </div>


              <div className="scenario-note">

                <strong>
                  Decision:
                </strong>{" "}

                {
                  decisionSupport.message
                }

              </div>


              <div className="scenario-table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Priority
                      </th>

                      <th>
                        District
                      </th>

                      <th>
                        Risk Score
                      </th>

                      <th>
                        Risk Level
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {(
                      decisionSupport
                        .priority_districts ||
                      []
                    ).map(
                      (
                        item,
                        index
                      ) => (

                        <tr
                          key={
                            `${item.district}-${index}`
                          }
                        >

                          <td>
                            {index + 1}
                          </td>

                          <td>
                            {
                              item.district
                            }
                          </td>

                          <td>
                            {formatNumber(
                              item.risk_score
                            )}
                          </td>

                          <td>
                            <RiskBadge
                              level={
                                item.risk_level
                              }
                            />
                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>


              {Array.isArray(
                decisionSupport
                  .decision_basis
              ) && (

                <div className="scenario-note">

                  <strong>
                    Decision Basis:
                  </strong>

                  <div
                    style={{
                      marginTop:
                        "8px",
                    }}
                  >

                    {decisionSupport
                      .decision_basis
                      .map(
                        (
                          basis,
                          index
                        ) => (
                          <div
                            key={
                              index
                            }
                          >
                            {basis}
                          </div>
                        )
                      )}

                  </div>

                </div>

              )}

            </div>

          </section>

        )}


        {/* ====================================================
            GIS MAP
        ==================================================== */}

        <section data-page="map" className="map-section">

          <div className="panel map-panel">


            <div className="panel-heading">

              <div>

                <h3>

                  {whatIfMapActive
                    ? "Odisha What-If Risk Map"
                    : "Odisha Spatial Risk Map"}

                </h3>

                <p>

                  {whatIfMapActive
                    ? "District-level risk under the selected What-If scenario."
                    : "District boundaries colored by calculated spatial risk level."}

                </p>

              </div>


              {whatIfMapActive && (

                <button
                  type="button"
                  className="prediction-button prediction-secondary-button"
                  onClick={
                    handleResetWhatIfMap
                  }
                >
                  Reset to Base Risk Map
                </button>

              )}

            </div>


            {whatIfMapActive &&
              dynamicWhatIfData && (

                <div
                  className="scenario-note"
                  style={{
                    marginBottom:
                      "16px",
                  }}
                >

                  <strong>
                    Active What-If Scenario:
                  </strong>{" "}

                  Rainfall{" "}

                  {
                    dynamicWhatIfData
                      .scenario
                      ?.rainfall_change_percent
                  }%

                  {" | "}

                  Water level{" "}

                  {
                    dynamicWhatIfData
                      .scenario
                      ?.water_level_change_m
                  } m

                  {" | "}

                  Drainage effectiveness{" "}

                  {
                    dynamicWhatIfData
                      .scenario
                      ?.drainage_effectiveness_change_percent
                  }%

                </div>

              )}


            <div className="map-wrapper">

              <MapContainer
                center={[
                  20.3,
                  84.5,
                ]}
                zoom={7}
                scrollWheelZoom={true}
                className="risk-map"
              >

                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />


                <MapController
                  geojson={mapData}
                />


                {mapData && (

                  <GeoJSON

                    key={
                      `risk-map-${mapRefreshKey}-${whatIfMapActive}`
                    }

                    data={mapData}


                    style={(feature) => {

                      const scenario =
                        getDynamicScenarioDistrict(
                          feature
                        );

                      const riskLevel =
                        scenario
                          ? (
                              scenario
                                .risk_level ||
                              "Unknown"
                            )
                          : (
                              feature
                                ?.properties
                                ?.risk_level ||
                              "Unknown"
                            );

                      return {

                        fillColor:
                          getRiskColor(
                            riskLevel
                          ),

                        weight:
                          1.5,

                        color:
                          "#ffffff",

                        fillOpacity:
                          0.65,

                      };

                    }}


                    onEachFeature={(
                      feature,
                      layer
                    ) => {

                      const properties =
                        feature
                          .properties ||
                        {};


                      const district =
                        getMapDistrictName(
                          feature
                        );


                      const matchingDistrict =
                        districts.find(
                          (item) =>
                            String(
                              item.NAME_2
                            )
                              .trim()
                              .toLowerCase() ===
                            String(
                              district
                            )
                              .trim()
                              .toLowerCase()
                        );


                      const scenario =
                        getDynamicScenarioDistrict(
                          feature
                        );


                      const riskLevel =
                        scenario
                          ? (
                              scenario
                                .risk_level ||
                              "Unknown"
                            )
                          : (
                              properties
                                .risk_level ||
                              "Unknown"
                            );


                      const baseScore =
                        properties
                          .spatial_risk_score;


                      const scenarioScore =
                        scenario
                          ? (
                              scenario
                                .scenario_risk_score ??
                              null
                            )
                          : null;


                      layer.on(
                        "click",
                        () => {

                          if (
                            matchingDistrict
                          ) {

                            setSelectedDistrictId(
                              matchingDistrict
                                .district_id
                            );

                          }

                        }
                      );


                      layer.bindTooltip(

                        whatIfMapActive &&
                        scenario

                          ? `${district} - ${riskLevel} (${formatNumber(
                              scenarioScore
                            )})`

                          : `${district} - ${riskLevel}`,

                        {
                          sticky:
                            true,
                        }

                      );


                      layer.bindPopup(`

                        <div class="map-popup">

                          <h3>
                            ${district}
                          </h3>

                          <p>
                            <strong>
                              Risk Level:
                            </strong>

                            <span
                              style="
                                color: ${getRiskColor(
                                  riskLevel
                                )};
                                font-weight: bold;
                              "
                            >
                              ${riskLevel}
                            </span>
                          </p>

                          ${
                            whatIfMapActive &&
                            scenario

                              ? `
                                <p>
                                  <strong>
                                    Scenario Risk Score:
                                  </strong>
                                  ${
                                    typeof scenarioScore ===
                                    "number"
                                      ? scenarioScore.toFixed(
                                          3
                                        )
                                      : "N/A"
                                  }
                                </p>

                                <p>
                                  <strong>
                                    Base Risk Score:
                                  </strong>
                                  ${
                                    typeof baseScore ===
                                    "number"
                                      ? baseScore.toFixed(
                                          3
                                        )
                                      : "N/A"
                                  }
                                </p>

                                <p>
                                  <strong>
                                    Map Mode:
                                  </strong>
                                  What-If Scenario
                                </p>
                              `

                              : `
                                <p>
                                  <strong>
                                    Risk Score:
                                  </strong>
                                  ${
                                    typeof baseScore ===
                                    "number"
                                      ? baseScore.toFixed(
                                          3
                                        )
                                      : "N/A"
                                  }
                                </p>
                              `
                          }

                          <p>
                            <strong>
                              Historical Flood Events:
                            </strong>

                            ${
                              properties
                                .historical_flood_events ??
                              "N/A"
                            }
                          </p>

                          <p>
                            <strong>
                              Elevation:
                            </strong>

                            ${
                              typeof properties
                                .elevation_m ===
                              "number"
                                ? `${properties.elevation_m.toFixed(
                                    1
                                  )} m`
                                : "N/A"
                            }
                          </p>

                          <p>
                            <strong>
                              Distance to River:
                            </strong>

                            ${
                              typeof properties
                                .distance_to_river_m ===
                              "number"
                                ? `${properties.distance_to_river_m.toFixed(
                                    0
                                  )} m`
                                : "N/A"
                            }
                          </p>

                        </div>

                      `);

                    }}

                  />

                )}

              </MapContainer>

            </div>


            <div className="map-legend">

              <span className="legend-title">
                Risk Level
              </span>


              {Object.entries(
                riskColors
              ).map(
                ([level, color]) => (

                  <div
                    className="legend-item"
                    key={level}
                  >

                    <span
                      className="legend-color"
                      style={{
                        backgroundColor:
                          color,
                      }}
                    ></span>

                    <span>
                      {level}
                    </span>

                  </div>

                )
              )}

            </div>

          </div>

        </section>


        {/* ====================================================
            DISTRICT EXPLORER
        ==================================================== */}

        <section data-page="map" className="district-explorer-section">

          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  District Explorer
                </h3>

                <p>
                  Select a district to inspect its
                  calculated flood-risk indicators.
                </p>

              </div>


              <div className="district-selector">

                <label htmlFor="district">
                  District
                </label>


                <select
                  id="district"
                  value={
                    selectedDistrict
                      ?.district_id ??
                    ""
                  }
                  onChange={
                    (event) =>
                      setSelectedDistrictId(
                        event.target.value
                      )
                  }
                >

                  {districts.map(
                    (district) => (

                      <option
                        key={
                          district.district_id
                        }
                        value={
                          district.district_id
                        }
                      >
                        {
                          district.NAME_2
                        }
                      </option>

                    )
                  )}

                </select>

              </div>

            </div>


            {selectedDistrict && (

              <div className="district-detail-grid">


                <div className="district-main-card">

                  <span>
                    Selected District
                  </span>

                  <h2>
                    {
                      selectedDistrict.NAME_2
                    }
                  </h2>


                  <div
                    className="district-risk-level"
                    style={{
                      backgroundColor:
                        getRiskColor(
                          selectedDistrict
                            .risk_level
                        ),
                    }}
                  >
                    {
                      selectedDistrict
                        .risk_level
                    }
                  </div>

                </div>


                <div className="district-metric-card">

                  <span>
                    Spatial Risk Score
                  </span>

                  <strong>
                    {formatNumber(
                      selectedDistrict
                        .spatial_risk_score
                    )}
                  </strong>

                </div>


                <div className="district-metric-card">

                  <span>
                    Historical Flood Events
                  </span>

                  <strong>
                    {
                      selectedDistrict
                        .historical_flood_events ??
                      "N/A"
                    }
                  </strong>

                </div>


                <div className="district-metric-card">

                  <span>
                    Elevation
                  </span>

                  <strong>

                    {typeof selectedDistrict
                      .elevation_m ===
                    "number"

                      ? `${selectedDistrict.elevation_m.toFixed(
                          1
                        )} m`

                      : "N/A"}

                  </strong>

                </div>


                <div className="district-metric-card">

                  <span>
                    Distance to River
                  </span>

                  <strong>

                    {typeof selectedDistrict
                      .distance_to_river_m ===
                    "number"

                      ? `${selectedDistrict.distance_to_river_m.toFixed(
                          0
                        )} m`

                      : "N/A"}

                  </strong>

                </div>


                <div className="district-metric-card">

                  <span>
                    Elevation Risk
                  </span>

                  <strong>
                    {formatNumber(
                      selectedDistrict
                        .elevation_risk
                    )}
                  </strong>

                </div>


                <div className="district-metric-card">

                  <span>
                    Drainage Risk
                  </span>

                  <strong>
                    {formatNumber(
                      selectedDistrict
                        .drainage_risk
                    )}
                  </strong>

                </div>


                <div className="district-metric-card">

                  <span>
                    Historical Risk
                  </span>

                  <strong>
                    {formatNumber(
                      selectedDistrict
                        .historical_risk
                    )}
                  </strong>

                </div>

              </div>

            )}

          </div>

        </section>


        {/* ====================================================
            FEATURE 6 - DISTRICT COMPARISON
        ==================================================== */}

        <section data-page="map" className="district-explorer-section">

          <div className="panel">

            <div className="panel-heading">

              <div>

                <h3>
                  District Comparison
                </h3>

                <p>
                  Compare two Odisha districts using
                  their calculated spatial-risk indicators.
                </p>

              </div>

            </div>


            <div className="prediction-form">

              <div className="prediction-grid">


                <div className="prediction-field">

                  <label>
                    District A
                  </label>

                  <select
                    value={
                      selectedDistrict
                        ?.district_id ??
                      ""
                    }
                    onChange={
                      (event) =>
                        setSelectedDistrictId(
                          event.target.value
                        )
                    }
                  >

                    {districts.map(
                      (district) => (

                        <option
                          key={
                            district.district_id
                          }
                          value={
                            district.district_id
                          }
                        >
                          {
                            district.NAME_2
                          }
                        </option>

                      )
                    )}

                  </select>

                </div>


                <div className="prediction-field">

                  <label>
                    District B
                  </label>

                  <select
                    value={
                      comparisonDistrict
                        ?.district_id ??
                      ""
                    }
                    onChange={
                      (event) =>
                        setComparisonDistrictId(
                          event.target.value
                        )
                    }
                  >

                    {districts.map(
                      (district) => (

                        <option
                          key={
                            district.district_id
                          }
                          value={
                            district.district_id
                          }
                        >
                          {
                            district.NAME_2
                          }
                        </option>

                      )
                    )}

                  </select>

                </div>

              </div>

            </div>


            {selectedDistrict &&
              comparisonDistrict && (

                <div className="scenario-table-wrapper">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Indicator
                        </th>

                        <th>
                          {
                            selectedDistrict.NAME_2
                          }
                        </th>

                        <th>
                          {
                            comparisonDistrict.NAME_2
                          }
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      <tr>

                        <td>
                          Risk Score
                        </td>

                        <td>
                          {formatNumber(
                            selectedDistrict
                              .spatial_risk_score
                          )}
                        </td>

                        <td>
                          {formatNumber(
                            comparisonDistrict
                              .spatial_risk_score
                          )}
                        </td>

                      </tr>


                      <tr>

                        <td>
                          Risk Level
                        </td>

                        <td>
                          <RiskBadge
                            level={
                              selectedDistrict
                                .risk_level
                            }
                          />
                        </td>

                        <td>
                          <RiskBadge
                            level={
                              comparisonDistrict
                                .risk_level
                            }
                          />
                        </td>

                      </tr>


                      <tr>

                        <td>
                          Elevation
                        </td>

                        <td>
                          {typeof selectedDistrict
                            .elevation_m ===
                          "number"
                            ? `${selectedDistrict.elevation_m.toFixed(
                                1
                              )} m`
                            : "N/A"}
                        </td>

                        <td>
                          {typeof comparisonDistrict
                            .elevation_m ===
                          "number"
                            ? `${comparisonDistrict.elevation_m.toFixed(
                                1
                              )} m`
                            : "N/A"}
                        </td>

                      </tr>


                      <tr>

                        <td>
                          Distance to River
                        </td>

                        <td>
                          {typeof selectedDistrict
                            .distance_to_river_m ===
                          "number"
                            ? `${selectedDistrict.distance_to_river_m.toFixed(
                                0
                              )} m`
                            : "N/A"}
                        </td>

                        <td>
                          {typeof comparisonDistrict
                            .distance_to_river_m ===
                          "number"
                            ? `${comparisonDistrict.distance_to_river_m.toFixed(
                                0
                              )} m`
                            : "N/A"}
                        </td>

                      </tr>


                      <tr>

                        <td>
                          Historical Flood Events
                        </td>

                        <td>
                          {
                            selectedDistrict
                              .historical_flood_events ??
                            "N/A"
                          }
                        </td>

                        <td>
                          {
                            comparisonDistrict
                              .historical_flood_events ??
                            "N/A"
                          }
                        </td>

                      </tr>


                      <tr>

                        <td>
                          Drainage Risk
                        </td>

                        <td>
                          {formatNumber(
                            selectedDistrict
                              .drainage_risk
                          )}
                        </td>

                        <td>
                          {formatNumber(
                            comparisonDistrict
                              .drainage_risk
                          )}
                        </td>

                      </tr>

                    </tbody>

                  </table>

                </div>

              )}

          </div>

        </section>


        {/* ====================================================
            AI FLOOD PREDICTION  -  HISTORICAL MODEL REFERENCE (2020)
        ==================================================== */}

        <section data-page="prediction" className="prediction-section">

          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  AI Flood Prediction  -  Historical Model Reference (2020)
                </h3>

                <p>
                  Flood-event classification using
                  the trained Random Forest model.
                </p>

              </div>

            </div>


            {predictionData && (

              <div className="prediction-result">


                <div className="prediction-result-card">

                  <span>
                    Prediction
                  </span>

                  <strong>
                    {
                      predictionData
                        .prediction_label
                    }
                  </strong>

                </div>


                <div className="prediction-result-card">

                  <span>
                    Flood Probability
                  </span>

                  <strong>

                    {Number(
                      predictionData
                        .flood_probability_percent
                    ).toFixed(2)}
                    %

                  </strong>

                </div>


                <div className="prediction-result-card">

                  <span>
                    Risk Level
                  </span>

                  <strong
                    style={{
                      color:
                        getRiskColor(
                          predictionData
                            .risk_level
                        ),
                    }}
                  >
                    {
                      predictionData
                        .risk_level
                    }
                  </strong>

                </div>


                {predictionData.reference_date && (

                  <div className="prediction-result-card">

                    <span>
                      Reference Date
                    </span>

                    <strong>
                      {
                        predictionData
                          .reference_date
                      }
                    </strong>

                  </div>

                )}

              </div>

            )}


            <form
              className="prediction-form"
              onSubmit={
                handleFloodPrediction
              }
            >

              <div className="prediction-grid">

                {[
                  [
                    "water_level_mean_m",
                    "Water Level Mean (m)",
                    "Mean water level",
                  ],
                  [
                    "water_level_max_m",
                    "Water Level Maximum (m)",
                    "Maximum level",
                  ],
                  [
                    "water_level_min_m",
                    "Water Level Minimum (m)",
                    "Minimum level",
                  ],
                  [
                    "water_level_change_m",
                    "Water Level Change (m)",
                    "Level change",
                  ],
                  [
                    "temperature",
                    "Temperature",
                    "Temperature",
                  ],
                  [
                    "humidity",
                    "Humidity",
                    "Humidity",
                  ],
                  [
                    "wind_speed",
                    "Wind Speed",
                    "Wind speed",
                  ],
                  [
                    "pressure",
                    "Pressure",
                    "Atmospheric pressure",
                  ],
                  [
                    "water_level_3day_mean_m",
                    "3-Day Water-Level Mean (m)",
                    "3-day mean",
                  ],
                  [
                    "water_level_7day_mean_m",
                    "7-Day Water-Level Mean (m)",
                    "7-day mean",
                  ],
                  [
                    "month",
                    "Month",
                    "1 - 12",
                  ],
                  [
                    "day_of_year",
                    "Day of Year",
                    "1 - 366",
                  ],
                ].map(
                  ([
                    name,
                    label,
                    placeholder,
                  ]) => (

                    <div
                      className="prediction-field"
                      key={
                        name
                      }
                    >

                      <label>
                        {label}
                      </label>

                      <input
                        type="number"
                        step="any"
                        name={name}
                        value={
                          predictionInput[
                            name
                          ]
                        }
                        onChange={
                          handlePredictionInputChange
                        }
                        placeholder={
                          placeholder
                        }
                      />

                    </div>

                  )
                )}

              </div>


              <div
                className="prediction-actions"
                style={{
                  marginTop:
                    "16px",
                }}
              >

                <button
                  type="submit"
                  className="prediction-button"
                  disabled={
                    predictionLoading
                  }
                >
                  {predictionLoading
                    ? "Generating Prediction..."
                    : "Generate Flood Prediction"}
                </button>


                <button
                  type="button"
                  className="prediction-button prediction-secondary-button"
                  disabled={
                    predictionLoading
                  }
                  onClick={
                    handleLatestAvailablePrediction
                  }
                >
                  {predictionLoading
                    ? "Loading..."
                    : "Use Latest Available Data"}
                </button>

              </div>

            </form>


            {predictionError && (

              <div className="prediction-error">

                {predictionError}

              </div>

            )}

          </div>

        </section>


        {/* ====================================================
            FEATURE 4 - AI EXPLAINABILITY
        ==================================================== */}

        {explainability &&
          explainability.available && (

            <section data-page="prediction" className="forecast-section">

              <div className="panel">


                <div className="panel-heading">

                  <div>

                    <h3>
                      AI Explainability
                    </h3>

                    <p>
                      Main factors used by the Random
                      Forest model and their overall importance.
                    </p>

                  </div>

                </div>


                <div className="scenario-grid">

                  <MetricCard
                    label="Prediction"
                    value={
                      explainability
                        .prediction
                        ?.prediction_label ||
                      "N/A"
                    }
                  />

                  <MetricCard
                    label="Probability"
                    value={
                      typeof explainability
                        .prediction
                        ?.flood_probability_percent ===
                      "number"
                        ? `${explainability.prediction.flood_probability_percent.toFixed(
                            2
                          )}%`
                        : "N/A"
                    }
                  />

                  <MetricCard
                    label="Risk Level"
                    value={
                      explainability
                        .prediction
                        ?.risk_level ||
                      "N/A"
                    }
                  />

                  <MetricCard
                    label="Reference Date"
                    value={
                      explainability
                        .reference_date ||
                      "N/A"
                    }
                  />

                </div>


                {Array.isArray(
                  explainability.top_features
                ) && (
                  <div className="scenario-table-wrapper">

                    <table>

                      <thead>

                        <tr>

                          <th>
                            Feature
                          </th>

                          <th>
                            Importance
                          </th>

                          <th>
                            Current Value
                          </th>

                          <th>
                            Training Median
                          </th>

                        </tr>

                      </thead>


                      <tbody>

                        {explainability
                          .top_features
                          .map(
                            (
                              item,
                              index
                            ) => (

                              <tr
                                key={
                                  item.feature ||
                                  index
                                }
                              >

                                <td>
                                  {
                                    item.display_name ||
                                    item.feature
                                  }
                                </td>

                                <td>
                                  {typeof item.importance_percent ===
                                  "number"
                                    ? `${item.importance_percent.toFixed(
                                        2
                                      )}%`
                                    : "N/A"}
                                </td>

                                <td>
                                  {
                                    item.current_value == null
                                      ? "N/A"
                                      : round2(item.current_value)
                                  }
                                </td>

                                <td>
                                  {
                                    item.training_median == null
                                      ? "N/A"
                                      : round2(item.training_median)
                                  }
                                </td>

                              </tr>

                            )
                          )}

                      </tbody>

                    </table>

                  </div>
                )}


                {Array.isArray(
                  explainability.top_features
                ) &&
                explainability.top_features.length >
                  0 && (

                  <div
                    className="forecast-chart"
                    style={{
                      marginTop:
                        "24px",
                    }}
                  >

                    <ResponsiveContainer
                      width="100%"
                      height={320}
                    >

                      <BarChart
                        data={
                          explainability.top_features
                            .slice(
                              0,
                              8
                            )
                            .map(
                              (item) => ({
                                feature:
                                  item.display_name ||
                                  item.feature,
                                importance:
                                  Number(
                                    item.importance_percent ||
                                    0
                                  ),
                              })
                            )
                        }
                        layout="vertical"
                        margin={{
                          top: 20,
                          right: 30,
                          left: 100,
                          bottom: 10,
                        }}
                      >

                        <CartesianGrid
                          strokeDasharray="3 3"
                        />

                        <XAxis
                          type="number"
                          unit="%"
                        />

                        <YAxis
                          type="category"
                          dataKey="feature"
                          width={120}
                          tick={{
                            fontSize: 11,
                          }}
                        />

                        <Tooltip />

                        <Bar
                          dataKey="importance"
                          name="Importance"
                          fill="#2563eb"
                        />

                      </BarChart>

                    </ResponsiveContainer>

                  </div>

                )}


                <div className="scenario-note">

                  <strong>
                    Interpretation:
                  </strong>{" "}

                  {
                    explainability
                      .interpretation_note ||
                    "Feature importance represents overall model importance, not causation."
                  }

                </div>

              </div>

            </section>

          )}


        {/* ====================================================
            HISTORICAL FORECAST
        ==================================================== */}

        <section data-page="forecast" className="forecast-section">

          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  Historical Water-Level Forecast  -  Model Reference (2021)
                </h3>

                <p>
                  Historical observed and model-predicted
                  water levels.
                </p>

              </div>

            </div>


            <div className="forecast-content">


              <div className="next-forecast-card">

                <span>
                  Latest Next-Day Forecast
                </span>

                <strong>

                  {typeof nextForecastValue ===
                  "number"
                    ? `${nextForecastValue.toFixed(
                        2
                      )} m`
                    : "N/A"}

                </strong>

                <small>
                  Date:{" "}
                  {nextForecastDate}
                </small>

              </div>


              <div className="forecast-chart">

                {forecastData.length > 0 ? (

                  <ResponsiveContainer
                    width="100%"
                    height={400}
                  >

                    <LineChart
                      data={
                        forecastData
                      }
                      margin={{
                        top: 20,
                        right: 30,
                        left: 10,
                        bottom: 10,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="date"
                        tick={{
                          fontSize: 11,
                        }}
                        minTickGap={25}
                      />

                      <YAxis
                        tick={{
                          fontSize: 11,
                        }}
                        label={{
                          value:
                            "Water Level (m)",
                          angle:
                            -90,
                          position:
                            "insideLeft",
                        }}
                      />

                      <Tooltip />

                      <Legend />

                      <Line
                        type="monotone"
                        dataKey="actual"
                        name="Observed"
                        stroke="#2563eb"
                        strokeWidth={2}
                        dot={false}
                      />

                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="Predicted"
                        stroke="#dc2626"
                        strokeWidth={2}
                        dot={false}
                      />

                    </LineChart>

                  </ResponsiveContainer>

                ) : (

                  <div className="no-data">
                    No historical forecast records available.
                  </div>

                )}

              </div>

            </div>

          </div>

        </section>


        {/* ====================================================
            FEATURE 7 - HISTORICAL TREND
        ==================================================== */}

        <section data-page="forecast" className="forecast-section">

          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  Historical Trend Analysis
                </h3>

                <p>
                  Summary of the available observed
                  water-level history.
                </p>

              </div>

            </div>


            <div className="scenario-summary">


              <MetricCard
                label="Average Level"
                value={
                  historicalAverage !==
                  null
                    ? `${historicalAverage.toFixed(
                        2
                      )} m`
                    : "N/A"
                }
              />


              <MetricCard
                label="Minimum Level"
                value={
                  historicalMinimum !==
                  null
                    ? `${historicalMinimum.toFixed(
                        2
                      )} m`
                    : "N/A"
                }
              />


              <MetricCard
                label="Maximum Level"
                value={
                  historicalMaximum !==
                  null
                    ? `${historicalMaximum.toFixed(
                        2
                      )} m`
                    : "N/A"
                }
              />


              <MetricCard
                label="Overall Trend"
                value={
                  historicalTrend
                }
                valueColor={
                  getTrendColor(
                    historicalTrend
                  )
                }
              />

            </div>


            <div className="scenario-note">

              <strong>
                Net historical change:
              </strong>{" "}

              {historicalChange !==
              null
                ? `${historicalChange >= 0 ? "+" : ""}${historicalChange.toFixed(
                    2
                  )} m`
                : "N/A"}

            </div>


            {historicalActuals.length >
              0 && (

              <div className="forecast-chart">

                <ResponsiveContainer
                  width="100%"
                  height={360}
                >

                  <LineChart
                    data={
                      historicalActuals
                    }
                    margin={{
                      top: 20,
                      right: 30,
                      left: 10,
                      bottom: 10,
                    }}
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                    />

                    <XAxis
                      dataKey="date"
                      tick={{
                        fontSize: 11,
                      }}
                      minTickGap={25}
                    />

                    <YAxis
                      tick={{
                        fontSize: 11,
                      }}
                      label={{
                        value:
                          "Observed Water Level (m)",
                        angle:
                          -90,
                        position:
                          "insideLeft",
                      }}
                    />

                    <Tooltip />

                    <Line
                      type="monotone"
                      dataKey="water_level"
                      name="Observed Water Level"
                      stroke="#0f766e"
                      strokeWidth={3}
                      dot={false}
                    />

                  </LineChart>

                </ResponsiveContainer>

              </div>

            )}

          </div>

        </section>


        {/* ====================================================
            7-DAY FORECAST
        ==================================================== */}

        <section data-page="forecast" className="forecast-section">

          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  7-Day Water-Level Forecast  -  Model Reference (2021)
                </h3>

                <p>
                  Recursive model forecast for the next
                  seven days.
                </p>

              </div>

            </div>


            <div className="dynamic-what-if-content">


              <div className="scenario-summary">


                <MetricCard
                  label="Day 1"
                  value={
                    futureForecastData[0]
                      ?.predicted !==
                      null &&
                    futureForecastData[0]
                      ?.predicted !==
                      undefined
                      ? `${futureForecastData[0].predicted.toFixed(
                          2
                        )} m`
                      : "N/A"
                  }
                />


                <MetricCard
                  label="Day 3"
                  value={
                    futureForecastData[2]
                      ?.predicted !==
                      null &&
                    futureForecastData[2]
                      ?.predicted !==
                      undefined
                      ? `${futureForecastData[2].predicted.toFixed(
                          2
                        )} m`
                      : "N/A"
                  }
                />


                <MetricCard
                  label="Day 7"
                  value={
                    futureForecastData[6]
                      ?.predicted !==
                      null &&
                    futureForecastData[6]
                      ?.predicted !==
                      undefined
                      ? `${futureForecastData[6].predicted.toFixed(
                          2
                        )} m`
                      : "N/A"
                  }
                />


                <MetricCard
                  label="Forecast Trend"
                  value={
                    forecastTrend
                  }
                  valueColor={
                    getTrendColor(
                      forecastTrend
                    )
                  }
                />

              </div>


              {futureForecastData.length >
                0 ? (

                <div className="forecast-chart">

                  <ResponsiveContainer
                    width="100%"
                    height={360}
                  >

                    <LineChart
                      data={
                        futureForecastData
                      }
                      margin={{
                        top: 20,
                        right: 30,
                        left: 10,
                        bottom: 15,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="date"
                        tick={{
                          fontSize: 11,
                        }}
                        minTickGap={15}
                      />

                      <YAxis
                        tick={{
                          fontSize: 11,
                        }}
                        label={{
                          value:
                            "Forecast Water Level (m)",
                          angle:
                            -90,
                          position:
                            "insideLeft",
                        }}
                      />

                      <Tooltip />

                      <Legend />

                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="Predicted"
                        stroke="#1d4ed8"
                        strokeWidth={3}
                        dot={{
                          r: 4,
                        }}
                        activeDot={{
                          r: 6,
                        }}
                      />

                    </LineChart>

                  </ResponsiveContainer>

                </div>

              ) : (

                <div className="no-data">
                  No 7-day forecast records available.
                </div>

              )}


              {futureForecastData.length >
                0 && (

                <div className="scenario-table-wrapper">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Day
                        </th>

                        <th>
                          Forecast Date
                        </th>

                        <th>
                          Predicted Level
                        </th>

                        <th>
                          Change
                        </th>

                        <th>
                          Trend
                        </th>

                        <th>
                          Forecast Range
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {futureForecastData.map(
                        (
                          forecast,
                          index
                        ) => (

                          <tr
                            key={
                              `${forecast.date}-${index}`
                            }
                          >

                            <td>
                              Day{" "}
                              {
                                forecast.forecastDay ??
                                index + 1
                              }
                            </td>

                            <td>
                              {
                                forecast.date
                              }
                            </td>

                            <td>
                              {
                                forecast.predicted.toFixed(
                                  2
                                )
                              }{" "}
                              m
                            </td>

                            <td>

                              {forecast.change !==
                              null
                                ? `${forecast.change.toFixed(
                                    2
                                  )} m`
                                : "N/A"}

                            </td>

                            <td>

                              <span
                                className="risk-badge"
                                style={{
                                  backgroundColor:
                                    getTrendColor(
                                      forecast.trend
                                    ),
                                }}
                              >
                                {
                                  forecast.trend
                                }
                              </span>

                            </td>

                            <td>

                              {forecast.lower !==
                                null &&
                              forecast.upper !==
                                null
                                ? `${forecast.lower.toFixed(
                                    2
                                  )} - ${forecast.upper.toFixed(
                                    2
                                  )} m`
                                : "N/A"}

                            </td>

                          </tr>

                        )
                      )}

                    </tbody>

                  </table>

                </div>

              )}


              <div className="scenario-note">

                <strong>
                  Forecast note:
                </strong>{" "}

                The 7-day values are recursive model
                forecasts generated from the available
                project water-level data. The forecast
                range is an estimated model-error range,
                not a formally calibrated prediction interval.

              </div>

            </div>

          </div>

        </section>


        {/* ====================================================
            DYNAMIC WHAT-IF
        ==================================================== */}

        <section data-page="whatif" className="what-if-section">

          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  Dynamic What-If Emergency Analysis
                </h3>

                <p>
                  Change rainfall, water level, and drainage
                  conditions to calculate a new district-level
                  spatial risk scenario.
                </p>

              </div>

            </div>


            <form
              className="prediction-form"
              onSubmit={
                handleDynamicWhatIf
              }
            >

              <div className="prediction-grid">


                <div className="prediction-field">

                  <label>
                    Rainfall Change (%)
                  </label>

                  <input
                    type="number"
                    min="-100"
                    max="500"
                    step="1"
                    name="rainfall_change_percent"
                    value={
                      whatIfInput
                        .rainfall_change_percent
                    }
                    onChange={
                      handleWhatIfInputChange
                    }
                    placeholder="e.g. 25"
                  />

                </div>


                <div className="prediction-field">

                  <label>
                    Water-Level Change (m)
                  </label>

                  <input
                    type="number"
                    min="-10"
                    max="10"
                    step="0.1"
                    name="water_level_change_m"
                    value={
                      whatIfInput
                        .water_level_change_m
                    }
                    onChange={
                      handleWhatIfInputChange
                    }
                    placeholder="e.g. 0.5"
                  />

                </div>


                <div className="prediction-field">

                  <label>
                    Drainage Effectiveness Change (%)
                  </label>

                  <input
                    type="number"
                    min="-100"
                    max="100"
                    step="1"
                    name="drainage_effectiveness_change_percent"
                    value={
                      whatIfInput
                        .drainage_effectiveness_change_percent
                    }
                    onChange={
                      handleWhatIfInputChange
                    }
                    placeholder="e.g. -15"
                  />

                </div>

              </div>


              <button
                type="submit"
                className="prediction-button"
                disabled={
                  whatIfLoading
                }
              >
                {whatIfLoading
                  ? "Analyzing Scenario..."
                  : "Run What-If Analysis"}
              </button>

            </form>


            {dynamicWhatIfError && (

              <div className="prediction-error">

                {
                  dynamicWhatIfError
                }

              </div>

            )}


            {dynamicWhatIfData && (

              <>

                <div className="scenario-grid">


                  <MetricCard
                    label="Average Risk"
                    value={
                      formatNumber(
                        dynamicWhatIfData
                          .summary
                          ?.average_risk_score
                      )
                    }
                  />


                  <MetricCard
                    label="Minimum Risk"
                    value={
                      formatNumber(
                        dynamicWhatIfData
                          .summary
                          ?.minimum_risk_score
                      )
                    }
                  />


                  <MetricCard
                    label="Maximum Risk"
                    value={
                      formatNumber(
                        dynamicWhatIfData
                          .summary
                          ?.maximum_risk_score
                      )
                    }
                  />


                  <MetricCard
                    label="Very High Districts"
                    value={
                      dynamicWhatIfData
                        .summary
                        ?.very_high_districts ??
                      0
                    }
                  />

                </div>


                <div className="scenario-distribution">


                  <div>
                    <span>
                      Low
                    </span>
                    <strong>
                      {
                        dynamicWhatIfData
                          .summary
                          ?.low_districts ??
                        0
                      }
                    </strong>
                  </div>


                  <div>
                    <span>
                      Moderate
                    </span>
                    <strong>
                      {
                        dynamicWhatIfData
                          .summary
                          ?.moderate_districts ??
                        0
                      }
                    </strong>
                  </div>


                  <div>
                    <span>
                      High
                    </span>
                    <strong>
                      {
                        dynamicWhatIfData
                          .summary
                          ?.high_districts ??
                        0
                      }
                    </strong>
                  </div>


                  <div>
                    <span>
                      Very High
                    </span>
                    <strong>
                      {
                        dynamicWhatIfData
                          .summary
                          ?.very_high_districts ??
                        0
                      }
                    </strong>
                  </div>


                </div>


                <div className="scenario-note">

                  <strong>
                    Scenario:
                  </strong>{" "}

                  Rainfall{" "}
                  {
                    dynamicWhatIfData
                      .scenario
                      ?.rainfall_change_percent
                  }%{" | "}

                  Water level{" "}
                  {
                    dynamicWhatIfData
                      .scenario
                      ?.water_level_change_m
                  } m{" | "}

                  Drainage effectiveness{" "}
                  {
                    dynamicWhatIfData
                      .scenario
                      ?.drainage_effectiveness_change_percent
                  }%

                </div>


                <div className="scenario-table-wrapper">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          District
                        </th>

                        <th>
                          Base Risk
                        </th>

                        <th>
                          Scenario Risk
                        </th>

                        <th>
                          Risk Level
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {Array.isArray(
                        dynamicWhatIfData
                          .results
                      ) &&
                        dynamicWhatIfData
                          .results
                          .map(
                            (
                              district,
                              index
                            ) => (

                              <tr
                                key={
                                  district
                                    .district_id ??
                                  index
                                }
                              >

                                <td>
                                  {
                                    district
                                      .NAME_2 ??
                                    district
                                      .district ??
                                    "Unknown"
                                  }
                                </td>

                                <td>
                                  {formatNumber(
                                    district
                                      .spatial_risk_score
                                  )}
                                </td>

                                <td>
                                  {formatNumber(
                                    district
                                      .scenario_risk_score
                                  )}
                                </td>

                                <td>
                                  <RiskBadge
                                    level={
                                      district
                                        .risk_level
                                    }
                                  />
                                </td>

                              </tr>

                            )
                          )}

                    </tbody>

                  </table>

                </div>


                <div className="scenario-note">

                  <strong>
                    Decision-support note:
                  </strong>{" "}

                  This scenario engine applies
                  project-defined environmental adjustments
                  to the existing spatial risk score.
                  It is not a calibrated physical flood
                  probability model.

                </div>

              </>

            )}

          </div>

        </section>


        {/* ====================================================
            FEATURE 5 - EMERGENCY ACTION PANEL
        ==================================================== */}

        {emergencyActions && (

          <section data-page="emergency" className="status-section">

            <div className="panel status-panel">


              <div className="status-header">

                <h3>
                  Emergency Action Panel
                </h3>

                <span
                  className="prototype-badge"
                  style={{
                    backgroundColor:
                      getRiskColor(
                        emergencyActions
                          .overall_status
                      ),
                  }}
                >
                  {
                    emergencyActions
                      .alert_level
                  }
                </span>

              </div>


              <div className="scenario-note">

                <strong>
                  Current status:
                </strong>{" "}

                {
                  emergencyActions
                    .message
                }

              </div>


              <div className="scenario-table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Priority
                      </th>

                      <th>
                        Recommended Action
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {(
                      emergencyActions
                        .recommended_actions ||
                      []
                    ).map(
                      (
                        item,
                        index
                      ) => (

                        <tr
                          key={
                            index
                          }
                        >

                          <td>
                            <RiskBadge
                              level={
                                item.priority ===
                                "Immediate"
                                  ? "Very High"
                                  : item.priority ===
                                    "High"
                                  ? "High"
                                  : item.priority ===
                                    "Medium"
                                  ? "Moderate"
                                  : "Low"
                              }
                            />
                          </td>

                          <td>
                            {
                              item.action
                            }
                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>


              <div className="scenario-note">

                {
                  emergencyActions.note
                }

              </div>

            </div>

          </section>

        )}


        {/* ====================================================
            EXISTING PROJECT SCENARIOS
        ==================================================== */}

        <section data-page="whatif" className="what-if-section">

          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  Interactive What-If Scenario Explorer
                </h3>

                <p>
                  Select a predefined What-If scenario to inspect
                  its detailed district-level risk results. These
                  scenarios are screening analyses, not hydraulic
                  flood simulations or official warnings.
                </p>

              </div>


              <div className="scenario-selector">

                <label htmlFor="scenario">
                  Scenario
                </label>


                <select
                  id="scenario"
                  value={
                    selectedScenario
                  }
                  onChange={
                    (event) =>
                      setSelectedScenario(
                        event.target.value
                      )
                  }
                >

                  {availableScenarios.map(
                    (scenario) => (

                      <option
                        key={
                          scenario
                        }
                        value={
                          scenario
                        }
                      >
                        {
                          scenario
                        }
                      </option>

                    )
                  )}

                </select>

              </div>

            </div>


            <div className="scenario-grid">


              <MetricCard
                label="Average Risk"
                value={
                  formatNumber(
                    selectedScenarioSummary
                      ?.average_risk_score
                  )
                }
              />


              <MetricCard
                label="Minimum Risk"
                value={
                  formatNumber(
                    selectedScenarioSummary
                      ?.minimum_risk_score
                  )
                }
              />


              <MetricCard
                label="Maximum Risk"
                value={
                  formatNumber(
                    selectedScenarioSummary
                      ?.maximum_risk_score
                  )
                }
              />


              <MetricCard
                label="Very High Districts"
                value={
                  selectedScenarioSummary
                    ?.very_high_districts ??
                  0
                }
              />

            </div>


            <div className="scenario-distribution">


              <div>
                <span>
                  Moderate
                </span>
                <strong>
                  {
                    selectedScenarioSummary
                      ?.moderate_districts ??
                    0
                  }
                </strong>
              </div>


              <div>
                <span>
                  High
                </span>
                <strong>
                  {
                    selectedScenarioSummary
                      ?.high_districts ??
                    0
                  }
                </strong>
              </div>


              <div>
                <span>
                  Very High
                </span>
                <strong>
                  {
                    selectedScenarioSummary
                      ?.very_high_districts ??
                    0
                  }
                </strong>
              </div>

            </div>


            <div className="scenario-table-wrapper">

              <table>

                <thead>

                  <tr>

                    <th>
                      District
                    </th>

                    <th>
                      Scenario Risk Score
                    </th>

                    <th>
                      Risk Level
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {selectedScenarioResults.map(
                    (
                      district,
                      index
                    ) => (

                      <tr
                        key={
                          district
                            .district_id ??
                          index
                        }
                      >

                        <td>
                          {
                            district
                              .NAME_2 ??
                            "Unknown"
                          }
                        </td>

                        <td>
                          {
                            typeof district
                              .scenario_risk_score ===
                            "number"
                              ? district
                                  .scenario_risk_score
                                  .toFixed(
                                    3
                                  )
                              : "-"
                          }
                        </td>

                        <td>
                          <RiskBadge
                            level={
                              district
                                .risk_level
                            }
                          />
                        </td>

                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>


            <div className="scenario-note">

              <strong>
                Note:
              </strong>{" "}

              These are predefined project scenarios
              retained from the existing What-If pipeline.
              Use Dynamic What-If Analysis above for
              custom user-defined conditions.

            </div>

          </div>

        </section>


                {/* ====================================================
            FEATURE 8 - MODEL PERFORMANCE
        ==================================================== */}

        {modelPerformance && (
          <section data-page="models" className="forecast-section">

            <div className="panel">

              <div className="panel-heading">
                <div>
                  <h3>
                    Model Performance Dashboard
                  </h3>

                  <p>
                    Validation summary for the trained Random Forest
                    high-water event classifier.
                  </p>
                </div>
              </div>

              <div className="scenario-grid">

                <MetricCard
                  label="Model"
                  value={
                    modelPerformance.model_name || "N/A"
                  }
                />

                <MetricCard
                  label="Features"
                  value={
                    modelPerformance.feature_count ?? "N/A"
                  }
                />

                <MetricCard
                  label="Training Samples"
                  value={
                    modelPerformance.evaluation?.training_samples ??
                    "N/A"
                  }
                />

                <MetricCard
                  label="Test Samples"
                  value={
                    modelPerformance.evaluation?.test_samples ??
                    "N/A"
                  }
                />

              </div>

              {modelPerformance.evaluation?.available && (
                <>

                  <div className="scenario-distribution">

                    <div>
                      <span>Negative Test Events</span>
                      <strong>
                        {
                          Number(
                            modelPerformance.evaluation
                              .confusion_matrix?.[0]?.[0] ?? 0
                          ) +
                          Number(
                            modelPerformance.evaluation
                              .confusion_matrix?.[0]?.[1] ?? 0
                          )
                        }
                      </strong>
                    </div>

                    <div>
                      <span>Positive Test Events</span>
                      <strong>
                        {
                          Number(
                            modelPerformance.evaluation
                              .confusion_matrix?.[1]?.[0] ?? 0
                          ) +
                          Number(
                            modelPerformance.evaluation
                              .confusion_matrix?.[1]?.[1] ?? 0
                          )
                        }
                      </strong>
                    </div>

                    <div>
                      <span>Positive-Class Share</span>
                      <strong>
                        {
                          modelPerformance.evaluation.test_samples
                            ? (
                                (
                                  (
                                    Number(
                                      modelPerformance.evaluation
                                        .confusion_matrix?.[1]?.[0] ?? 0
                                    ) +
                                    Number(
                                      modelPerformance.evaluation
                                        .confusion_matrix?.[1]?.[1] ?? 0
                                    )
                                  ) /
                                  Number(
                                    modelPerformance.evaluation.test_samples
                                  )
                                ) * 100
                              ).toFixed(1)
                            : "N/A"
                        }%
                      </strong>
                    </div>

                    <div>
                      <span>Target</span>
                      <strong>
                        {
                          modelPerformance.evaluation.target_column ||
                          "N/A"
                        }
                      </strong>
                    </div>

                  </div>

                  <div className="scenario-grid">

                    <MetricCard
                      label="Accuracy"
                      value={
                        typeof modelPerformance.evaluation
                          .accuracy_percent === "number"
                          ? `${modelPerformance.evaluation.accuracy_percent.toFixed(2)}%`
                          : "N/A"
                      }
                    />

                    <MetricCard
                      label="Precision"
                      value={
                        typeof modelPerformance.evaluation
                          .precision_percent === "number"
                          ? `${modelPerformance.evaluation.precision_percent.toFixed(2)}%`
                          : "N/A"
                      }
                    />

                    <MetricCard
                      label="Recall"
                      value={
                        typeof modelPerformance.evaluation
                          .recall_percent === "number"
                          ? `${modelPerformance.evaluation.recall_percent.toFixed(2)}%`
                          : "N/A"
                      }
                    />

                    <MetricCard
                      label="F1 Score"
                      value={
                        typeof modelPerformance.evaluation
                          .f1_percent === "number"
                          ? `${modelPerformance.evaluation.f1_percent.toFixed(2)}%`
                          : "N/A"
                      }
                    />

                    <MetricCard
                      label="ROC-AUC"
                      value={
                        typeof modelPerformance.evaluation
                          .roc_auc === "number"
                          ? modelPerformance.evaluation.roc_auc.toFixed(3)
                          : "N/A"
                      }
                    />

                  </div>

                  {Array.isArray(
                    modelPerformance.evaluation.confusion_matrix
                  ) &&
                    modelPerformance.evaluation.confusion_matrix.length >= 2 && (

                    <div
                      className="scenario-card"
                      style={{ marginTop: "18px" }}
                    >

                      <h4>Confusion Matrix</h4>

                      <p
                        style={{
                          marginTop: "4px",
                          opacity: 0.75
                        }}
                      >
                        Actual class versus predicted class on the
                        stratified random 20% holdout set.
                      </p>

                      <div style={{ overflowX: "auto" }}>

                        <table
                          style={{
                            width: "100%",
                            maxWidth: "620px",
                            borderCollapse: "collapse",
                            marginTop: "12px"
                          }}
                        >

                          <thead>
                            <tr>
                              <th></th>
                              <th>Predicted Negative</th>
                              <th>Predicted Positive</th>
                            </tr>
                          </thead>

                          <tbody>

                            <tr>
                              <th style={{ textAlign: "left" }}>
                                Actual Negative
                              </th>

                              <td style={{ textAlign: "center" }}>
                                {
                                  modelPerformance.evaluation
                                    .confusion_matrix?.[0]?.[0] ?? 0
                                }
                              </td>

                              <td style={{ textAlign: "center" }}>
                                {
                                  modelPerformance.evaluation
                                    .confusion_matrix?.[0]?.[1] ?? 0
                                }
                              </td>
                            </tr>

                            <tr>
                              <th style={{ textAlign: "left" }}>
                                Actual Positive
                              </th>

                              <td style={{ textAlign: "center" }}>
                                {
                                  modelPerformance.evaluation
                                    .confusion_matrix?.[1]?.[0] ?? 0
                                }
                              </td>

                              <td style={{ textAlign: "center" }}>
                                {
                                  modelPerformance.evaluation
                                    .confusion_matrix?.[1]?.[1] ?? 0
                                }
                              </td>
                            </tr>

                          </tbody>

                        </table>

                      </div>

                    </div>
                  )}

                  <div
                    className="scenario-note"
                    style={{ marginTop: "18px" }}
                  >

                    <strong>Validation context:</strong>{" "}

                    The evaluation uses a stratified random 80/20 holdout matching the training script.
                    The test set contains{" "}
                    {modelPerformance.evaluation.test_samples ?? "N/A"}{" "}
                    samples, with only{" "}
                    {
                      Number(
                        modelPerformance.evaluation
                          .confusion_matrix?.[1]?.[0] ?? 0
                      ) +
                      Number(
                        modelPerformance.evaluation
                          .confusion_matrix?.[1]?.[1] ?? 0
                      )
                    }{" "}
                    positive event example.

                    <br />

                    Because the positive class is very limited,
                    the reported validation metrics should not be
                    interpreted as strong evidence of general
                    flood-event detection performance.

                  </div>

                  {modelPerformance.evaluation.target_note && (
                    <div
                      className="scenario-note"
                      style={{ marginTop: "10px" }}
                    >

                      <strong>Target limitation:</strong>{" "}

                      {modelPerformance.evaluation.target_note}

                    </div>
                  )}

                  <div
                    className="scenario-note"
                    style={{ marginTop: "10px" }}
                  >

                    <strong>Additional validation:</strong>{" "}

                   ROC-AUC is reported from held-out prediction
                   probabilities. PR-AUC is not currently reported
                   by the backend evaluation endpoint.
                  </div>

                  {modelPerformance.model_parameters && (
                    <div
                      className="scenario-card"
                      style={{ marginTop: "18px" }}
                    >

                      <h4>Model Configuration</h4>

                      <p>

                        Trees:{" "}
                        <strong>
                          {
                            modelPerformance.model_parameters
                              .n_estimators ?? "N/A"
                          }
                        </strong>

                        {"  -  "}

                        Max Depth:{" "}
                        <strong>
                          {
                            modelPerformance.model_parameters
                              .max_depth ?? "N/A"
                          }
                        </strong>

                        {"  -  "}

                        Min Samples Split:{" "}
                        <strong>
                          {
                            modelPerformance.model_parameters
                              .min_samples_split ?? "N/A"
                          }
                        </strong>

                        {"  -  "}

                        Min Samples Leaf:{" "}
                        <strong>
                          {
                            modelPerformance.model_parameters
                              .min_samples_leaf ?? "N/A"
                          }
                        </strong>

                      </p>

                      <p style={{ marginBottom: 0 }}>

                        Class Weight:{" "}
                        <strong>
                          {String(
                            modelPerformance.model_parameters
                              .class_weight ?? "N/A"
                          )}
                        </strong>

                        {"  -  "}

                        Random State:{" "}
                        <strong>
                          {
                            modelPerformance.model_parameters
                              .random_state ?? "N/A"
                          }
                        </strong>

                      </p>

                    </div>
                  )}

                </>
              )}

              {Array.isArray(
                modelPerformance.feature_importance
              ) &&
                modelPerformance.feature_importance.length > 0 && (

                <div className="forecast-chart">

                  <h4 style={{ marginBottom: "10px" }}>
                    Feature Importance
                  </h4>

                  <p style={{ opacity: 0.75 }}>
                    Relative importance reported by the trained
                    Random Forest. Importance indicates model
                    contribution and does not establish causation.
                  </p>

                  <ResponsiveContainer
                    width="100%"
                    height={420}
                  >

                    <BarChart
                      data={
                        modelPerformance.feature_importance
                          .slice(0, 12)
                          .map((item) => ({
                            feature:
                              item.display_name ||
                              item.feature,
                            importance:
                              Number(
                                item.importance_percent || 0
                              ),
                          }))
                      }
                      layout="vertical"
                      margin={{
                        top: 20,
                        right: 30,
                        left: 120,
                        bottom: 10,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        type="number"
                        unit="%"
                      />

                      <YAxis
                        type="category"
                        dataKey="feature"
                        width={140}
                        tick={{
                          fontSize: 11,
                        }}
                      />

                      <Tooltip />

                      <Bar
                        dataKey="importance"
                        name="Importance"
                        fill="#0f766e"
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>
              )}

              {!modelPerformance.evaluation?.available && (
                <div className="scenario-note">
                  {
                    modelPerformance.evaluation?.message ||
                    "Model evaluation is not available."
                  }
                </div>
              )}

            </div>

          </section>
        )}

{/* ====================================================
            FEATURE 10 - DATA QUALITY
        ==================================================== */}

        {dataQuality && (

          <section data-page="data" className="status-section">

            <div className="panel status-panel">


              <div className="status-header">

                <h3>
                  Data Quality & Freshness
                </h3>

                <span className="prototype-badge">
                  {
                    dataQuality
                      .overall_status
                  }
                </span>

              </div>


              <div className="scenario-grid">


                <MetricCard
                  label="CSV Datasets"
                  value={
                    dataQuality
                      .summary
                      ?.available_csv_datasets ??
                    0
                  }
                />


                <MetricCard
                  label="Total CSV Sources"
                  value={
                    dataQuality
                      .summary
                      ?.csv_datasets ??
                    0
                  }
                />


                <MetricCard
                  label="Supporting Files"
                  value={
                    dataQuality
                      .summary
                      ?.available_supporting_files ??
                    0
                  }
                />


                <MetricCard
                  label="Total Supporting Files"
                  value={
                    dataQuality
                      .summary
                      ?.supporting_files ??
                    0
                  }
                />

              </div>


              <div className="scenario-table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Dataset
                      </th>

                      <th>
                        Status
                      </th>

                      <th>
                        Rows
                      </th>

                      <th>
                        Columns
                      </th>

                      <th>
                        Missing %
                      </th>

                      <th>
                        Date Range
                      </th>

                      <th>
                        Last Modified
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {(
                      dataQuality
                        .datasets ||
                      []
                    ).map(
                      (
                        item,
                        index
                      ) => (

                        <tr
                          key={
                            `${item.name}-${index}`
                          }
                        >

                          <td>
                            {
                              item.name
                            }
                          </td>

                          <td>
                            <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, color: "#fff", fontWeight: 700, fontSize: 12,
                              background: item.status === "Available" ? "#16a34a" : item.status === "Read Error" ? "#dc2626" : "#ca8a04" }}>
                              {item.status === "Available" ? "✓ Available" : item.status}
                            </span>
                          </td>

                          <td>
                            {
                              item.row_count ??
                              "N/A"
                            }
                          </td>

                          <td>
                            {
                              item.column_count ??
                              "N/A"
                            }
                          </td>

                          <td>
                            {
                              item.missing_cells_percent !==
                              null &&
                              item.missing_cells_percent !==
                              undefined
                                ? `${item.missing_cells_percent}%`
                                : "N/A"
                            }
                          </td>

                          <td>

                            {item.date_range
                              ? `${
                                  item.date_range.start
                                } to ${
                                  item.date_range.end
                                }`
                              : "N/A"}

                          </td>

                          <td>
                            {
                              item.modified_at ??
                              "N/A"
                            }
                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>


              <div className="scenario-table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Supporting File
                      </th>

                      <th>
                        Status
                      </th>

                      <th>
                        Size
                      </th>

                      <th>
                        Last Modified
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {(
                      dataQuality
                        .supporting_files ||
                      []
                    ).map(
                      (
                        item,
                        index
                      ) => (

                        <tr
                          key={
                            `${item.name}-${index}`
                          }
                        >

                          <td>
                            {
                              item.name
                            }
                          </td>

                          <td>

                            <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, color: "#fff", fontWeight: 700, fontSize: 12,
                              background: item.status === "Available" ? "#16a34a" : item.status === "Read Error" ? "#dc2626" : "#ca8a04" }}>
                              {item.status === "Available" ? "✓ Available" : item.status}
                            </span>

                          </td>

                          <td>

                            {typeof item.size_mb ===
                            "number"
                              ? `${item.size_mb} MB`
                              : "N/A"}

                          </td>

                          <td>
                            {
                              item.modified_at ??
                              "N/A"
                            }
                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>


              <div className="scenario-note">

                <strong>
                  Freshness note:
                </strong>{" "}

                {
                  dataQuality
                    .freshness_note
                }

              </div>

            </div>

          </section>

        )}


        {/* ====================================================
            PROJECT STATUS
        ==================================================== */}

        <section data-page="data" className="status-section">

          <div className="panel status-panel">


            <div className="status-header">

              <h3>
                Flood Guard AI - Model & Data Status
              </h3>

              <span className="prototype-badge">
                Integrated Decision Support
              </span>

            </div>


            <div className="status-grid">


              <div className="status-item">

                <span>
                  Spatial Risk Map
                </span>

                <strong>
                  Available
                </strong>

              </div>


              <div className="status-item">

                <span>
                  District Coverage
                </span>

                <strong>
                  {
                    riskData?.district_count ||
                    0
                  }{" "}
                  Odisha Districts
                </strong>

              </div>


              <div className="status-item">

                <span>
                  Dynamic What-If
                </span>

                <strong>
                  Available
                </strong>

              </div>


              <div className="status-item">

                <span>
                  Flood Prediction
                </span>

                <strong>
                  Logistic Regression (live) + Random Forest
                </strong>

              </div>


              <div className="status-item">

                <span>
                  Explainability
                </span>

                <strong>
                  {explainability?.available
                    ? "Available"
                    : "Unavailable"}
                </strong>

              </div>


              <div className="status-item">

                <span>
                  Emergency Actions
                </span>

                <strong>
                  {emergencyActions?.available
                    ? "Available"
                    : "Unavailable"}
                </strong>

              </div>


            </div>


            <div className="status-note">

              <strong>
                Important:
              </strong>{" "}

              Flood Guard AI is an academic decision-support
              system using the available project datasets,
              trained machine-learning model, forecasting
              pipeline, and spatial-risk analysis. Results
              should not be interpreted as official real-time
              flood warnings.

            </div>

          </div>

        </section>


        {/* ====================================================
            RISK DISTRIBUTION
        ==================================================== */}

        <section data-page="map" className="bottom-grid">


          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  Risk Distribution
                </h3>

                <p>
                  Number of districts in each
                  risk category.
                </p>

              </div>

            </div>


            <div className="distribution-list">


              <div className="distribution-row">

                <span>
                  Very High
                </span>

                <strong>
                  {
                    distribution[
                      "Very High"
                    ] ||
                    0
                  }
                </strong>

              </div>


              <div className="distribution-row">

                <span>
                  High
                </span>

                <strong>
                  {
                    distribution.High ||
                    0
                  }
                </strong>

              </div>


              <div className="distribution-row">

                <span>
                  Moderate
                </span>

                <strong>
                  {
                    distribution.Moderate ||
                    0
                  }
                </strong>

              </div>


              <div className="distribution-row">

                <span>
                  Low
                </span>

                <strong>
                  {
                    distribution.Low ||
                    0
                  }
                </strong>

              </div>

            </div>

          </div>


          <div className="panel">


            <div className="panel-heading">

              <div>

                <h3>
                  District Risk Details
                </h3>

                <p>
                  District-wise calculated spatial
                  risk information.
                </p>

              </div>

            </div>


            <div className="table-wrapper">

              <table>

                <thead>

                  <tr>

                    <th>
                      District
                    </th>

                    <th>
                      Risk Score
                    </th>

                    <th>
                      Risk Level
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {districts.map(
                    (district) => (

                      <tr
                        key={
                          district
                            .district_id
                        }
                      >

                        <td>
                          {
                            district.NAME_2
                          }
                        </td>

                        <td>
                          {formatNumber(
                            district
                              .spatial_risk_score
                          )}
                        </td>

                        <td>
                          <RiskBadge
                            level={
                              district
                                .risk_level
                            }
                          />
                        </td>

                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>

          </div>

        </section>


      </main>

      </div>

    </div>
  );
}


export default App;







