import { useEffect, useState } from "react";

const API_BASE = "/api";

const riskColors = {
  Low: "#22c55e",
  Moderate: "#eab308",
  High: "#f97316",
  "Very High": "#dc2626",
};

function riskColor(level) {
  return riskColors[level] || "#64748b";
}

function firstArray(data, keys = []) {
  for (const key of keys) {
    if (Array.isArray(data?.[key])) {
      return data[key];
    }
  }
  return [];
}

function firstValue(data, keys = [], fallback = "N/A") {
  for (const key of keys) {
    if (
      data?.[key] !== undefined &&
      data?.[key] !== null &&
      data?.[key] !== ""
    ) {
      return data[key];
    }
  }
  return fallback;
}

function formatNumber(value, digits = 2) {
  return typeof value === "number"
    ? value.toFixed(digits)
    : "N/A";
}

function formatWholeNumber(value) {
  return typeof value === "number"
    ? value.toLocaleString("en-IN", {
        maximumFractionDigits: 0,
      })
    : "N/A";
}

export default function ExtendedIntelligence({
  selectedDistrictName,
}) {
  const [data, setData] = useState({
    priorityArea: null,
    forecastRisk: [],
    timeline: [],
    affectedDistricts: [],
    affectedCities: null,
    populationImpact: null,
    rainfallAnomaly: null,
    scenarioComparison: null,
    situation: null,
    alerts: null,
    vulnerability: null,
    impact: null,
    weather: null,
    areaAlert: null,
    dataSources: null,
    readiness: null,
    predictionHistory: [],
    alertHistory: [],
    scenarioHistory: [],
    reportHistory: [],
  });
  const [historicalWeather, setHistoricalWeather] = useState(null);

  useEffect(() => {
    let cancelled = false;

    fetch("/api/weather/historical-analysis")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Historical weather request failed");
        }
        return response.json();
      })
      .then((data) => {
        if (!cancelled) {
          const rows = Array.isArray(data.records) ? data.records : [];
          const firstDate = rows[0]?.date || "";
          const lastDate = rows[rows.length - 1]?.date || "";
          const trend = data.trend || {};
          const normalized = {
            ...data,
            period: data.period || {
              start_year: firstDate ? firstDate.slice(0, 4) : "N/A",
              end_year: lastDate ? lastDate.slice(0, 4) : "N/A",
            },
            reference_location: data.reference_location || "Bhubaneswar, Odisha, India",
            overall: data.overall || {
              temperature_trend: trend.temperature || "N/A",
              temperature_change_c: null,
              humidity_trend: trend.humidity || "N/A",
              humidity_change_pct: null,
              wind_trend: trend.wind || "N/A",
              wind_change_kmh: null,
              pressure_trend: trend.pressure || "N/A",
              pressure_change_hpa: null,
            },
            annual: Array.isArray(data.annual)
              ? data.annual
              : Object.values(
                  rows.reduce((acc, row) => {
                    const year = String(
                      row.year ||
                      (row.date ? String(row.date).slice(0, 4) : "")
                    );

                    if (!year || year === "undefined") {
                      return acc;
                    }

                    if (!acc[year]) {
                      acc[year] = {
                        year,
                        temperature: [],
                        humidity: [],
                        wind: [],
                        pressure: [],
                        count: 0,
                        humidity: [],
                        wind: [],
                        pressure: [],
                      };
                    }

                    acc[year].count += 1

                    const temperature = Number(
                      row.temperature_c ??
                      row.temperature_2m ??
                      row.avg_temperature_c
                    );

                    const humidity = Number(
                      row.humidity_pct ??
                      row.relative_humidity_2m ??
                      row.RH2M ??
                      row.avg_humidity_pct
                    );

                    const wind = Number(
                      row.wind_speed_kmh ??
                      row.wind_speed_10m ??
                      row.WS10M ??
                      row.avg_wind_speed_kmh
                    );

                    const pressure = Number(
                      row.pressure_hpa ??
                      row.surface_pressure ??
                      row.PS ??
                      row.avg_pressure_hpa
                    );

                    if (Number.isFinite(temperature)) {
                      acc[year].temperature.push(temperature);
                    }

                    if (Number.isFinite(humidity)) {
                      acc[year].humidity.push(humidity);
                    }

                    if (Number.isFinite(wind)) {
                      acc[year].wind.push(wind);
                    }

                    if (Number.isFinite(pressure)) {
                      acc[year].pressure.push(pressure);
                    }

                    return acc;
                  }, {})
                ).map((item) => ({
                  year: item.year,
                  avg_temperature_c:
                    item.temperature.length
                      ? Number(
                          (
                            item.temperature.reduce((a, b) => a + b, 0) /
                            item.temperature.length
                          ).toFixed(2)
                        )
                      : null,
                  avg_humidity_pct:
                    item.humidity.length
                      ? Number(
                          (
                            item.humidity.reduce((a, b) => a + b, 0) /
                            item.humidity.length
                          ).toFixed(2)
                        )
                      : null,
                  avg_wind_speed_kmh:
                    item.wind.length
                      ? Number(
                          (
                            item.wind.reduce((a, b) => a + b, 0) /
                            item.wind.length
                          ).toFixed(2)
                        )
                      : null,
                  avg_pressure_hpa:
                    item.pressure.length
                      ? Number(
                          (
                            item.pressure.reduce((a, b) => a + b, 0) /
                            item.pressure.length
                          ).toFixed(2)
                        )
                      : null,
                  records: item.count,
                })),

            interpretation: Array.isArray(data.interpretation) ? data.interpretation : [
              `Temperature trend: ${trend.temperature || "N/A"}.`,
              `Humidity trend: ${trend.humidity || "N/A"}.`,
              `Wind trend: ${trend.wind || "N/A"}.`,
              `Pressure trend: ${trend.pressure || "N/A"}.`,
            ],
          };
          setHistoricalWeather(normalized);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHistoricalWeather(null);
        }
      });


  return () => {
      cancelled = true;
    };
  }, []);


  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const load = async () => {
      setLoading(true);
      setError("");

      const district = selectedDistrictName || "Cuttack";

      const endpoints = [
        "priority-area",
        "forecast/risk",
        "forecast/timeline",
        "affected-districts",
        "situation-summary",
        "alerts",
        "vulnerability",
        "impact",
        `weather/district/${encodeURIComponent(district)}`,
        `area-alert/${encodeURIComponent(district)}`,
        "data-sources",
        "feature-readiness",
        "history/predictions",
        "history/alerts",
        "history/scenarios",
        "history/reports",
        `affected-cities?source=${encodeURIComponent(district)}`,
        `population-impact?source=${encodeURIComponent(district)}`,
        "rainfall-anomaly",
        "what-if/compare",
        `affected-districts?source=${encodeURIComponent(district)}`,
      ];

      const fetchJson = async (endpoint) => {
        try {
          const response = await fetch(`${API_BASE}/${endpoint}`);

          let result = {};
          try {
            result = await response.json();
          } catch {
            result = {};
          }

          return {
            ok: response.ok,
            result,
          };
        } catch (err) {
          return {
            ok: false,
            result: {
              _request_failed: true,
              _error: err?.message || "Request failed",
            },
          };
        }
      };

      try {
        const responses = await Promise.all(
          endpoints.map(fetchJson)
        );

        if (!active) {
          return;
        }

        const result = (index) =>
          responses[index]?.ok
            ? responses[index].result
            : {};

        setData({
          priorityArea:
            result(0)?.priority_area || null,

          forecastRisk:
            firstArray(result(1), [
              "forecast_risk",
              "results",
              "data",
            ]),

          timeline:
            firstArray(result(2), [
              "timeline",
              "results",
              "data",
            ]),

          affectedDistricts:
            firstArray(result(3), [
              "affected_districts",
              "potentially_affected_districts",
              "results",
              "data",
            ]),

          situation:
            result(4) || null,

          alerts:
            result(5) || null,

          vulnerability:
            result(6) || null,

          impact:
            result(7) || null,

          weather:
            result(8)?.weather ||
            result(8) ||
            null,

          areaAlert:
            result(9) || null,

          dataSources:
            result(10) || null,

          readiness:
            result(11) || null,

          predictionHistory:
            firstArray(result(12), [
              "predictions",
              "records",
              "history",
              "data",
            ]),

          alertHistory:
            firstArray(result(13), [
              "alerts",
              "records",
              "history",
              "data",
            ]),

          scenarioHistory:
            firstArray(result(14), [
              "scenarios",
              "records",
              "history",
              "data",
            ]),

          reportHistory:
            firstArray(result(15), [
              "reports",
              "records",
              "history",
              "data",
            ]),

          affectedCities:
            result(16) || null,

          populationImpact:
            result(17) || null,

          rainfallAnomaly:
            result(18) || null,

          scenarioComparison:
            result(19) || null,

          affectedFromSelected:
            firstArray(result(20), [
              "affected_districts",
              "potentially_affected_districts",
              "results",
              "data",
            ]),
        });
      } catch (err) {
        if (active) {
          setError(
            err?.message ||
              "Unable to load extended flood intelligence."
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    load();

    return () => {
      active = false;
    };
  }, [selectedDistrictName]);
  const timelinePeriods = [
    { label: "1 Day", day: 1 },
    { label: "3 Day", day: 3 },
    { label: "7 Day", day: 7 },
  ];

  const timeline =
    data.timeline.length > 0
      ? data.timeline
      : data.forecastRisk;

  const weatherRows =
    Array.isArray(data.weather?.daily)
      ? data.weather.daily.slice(0, 7)
      : [];
  const currentWeatherSnapshot = (() => {
    const times = Array.isArray(data.weather?.hourly?.time)
      ? data.weather.hourly.time
      : [];

    if (times.length === 0) {
      return null;
    }

    const now = Date.now();
    let bestIndex = -1;
    let bestDiff = Infinity;

    times.forEach((time, index) => {
      const timestamp = Date.parse(time);

      if (!Number.isFinite(timestamp)) {
        return;
      }

      const diff = Math.abs(timestamp - now);

      if (diff < bestDiff) {
        bestDiff = diff;
        bestIndex = index;
      }
    });

    if (bestIndex < 0) {
      return null;
    }

    return {
      time: times[bestIndex],
      temperature_c:
        data.weather?.hourly?.temperature_2m?.[bestIndex] ?? null,
      humidity_percent:
        data.weather?.hourly?.relative_humidity_2m?.[bestIndex] ?? null,
      rainfall_mm:
        data.weather?.hourly?.precipitation?.[bestIndex] ?? null,
      pressure_hpa:
        data.weather?.hourly?.surface_pressure?.[bestIndex] ?? null,
      wind_kmh:
        data.weather?.hourly?.wind_speed_10m?.[bestIndex] ?? null,
    };
  })();

  const scenarioComparisonRows =
    Array.isArray(data.scenarioComparison?.scenarios)
      ? data.scenarioComparison.scenarios
      : [];

  const baseScenario =
    scenarioComparisonRows.find(
      (item) => item.scenario === "Base"
    ) || null;

  const combinedScenario =
    scenarioComparisonRows.find(
      (item) =>
        item.scenario === "Combined Stress Scenario"
    ) || null;
  const priority =
    data.priorityArea || {};

  const selectedVulnerability =
    Array.isArray(
      data.vulnerability?.districts
    )
      ? data.vulnerability.districts.find(
          (item) =>
            String(
              item.district ||
                item.NAME_2 ||
                ""
            ).toLowerCase() ===
            String(
              selectedDistrictName || ""
            ).toLowerCase()
        )
      : null;

  const selectedImpact =
    Array.isArray(
      data.impact?.districts
    )
      ? data.impact.districts.find(
          (item) =>
            String(
              item.district ||
                item.NAME_2 ||
                ""
            ).toLowerCase() ===
            String(
              selectedDistrictName || ""
            ).toLowerCase()
        )
      : null;

  const selectedPopulationImpact =
    Array.isArray(
      data.populationImpact?.districts
    )
      ? data.populationImpact.districts.find(
          (item) =>
            String(
              item.district ||
                item.NAME_2 ||
                ""
            ).toLowerCase() ===
            String(
              selectedDistrictName ||
                ""
            ).toLowerCase()
        )
      : null;

  const affectedCityRows =
    firstArray(
      data.affectedCities,
      ["cities", "results", "data"]
    );

  const selectedRainfallAnomaly =
    Array.isArray(data.rainfallAnomaly?.anomalies)
      ? data.rainfallAnomaly.anomalies.find(
          (item) =>
            String(item.district || "").trim().toLowerCase() ===
            String(selectedDistrictName || "").trim().toLowerCase()
        )
      : null;

  const vulnerabilityScore =
    selectedVulnerability?.vulnerability_score ??
    selectedVulnerability?.score ??
    null;

  const getScoreRiskLevel = (score) => {
    if (typeof score !== "number") {
      return "N/A";
    }

    if (score < 0.25) {
      return "Low";
    }

    if (score < 0.50) {
      return "Moderate";
    }

    if (score < 0.75) {
      return "High";
    }

    return "Very High";
  };

  const vulnerabilityRiskLevel =
    getScoreRiskLevel(
      vulnerabilityScore
    );

  const impactScore =
    selectedImpact?.impact_score ??
    selectedImpact?.score ??
    null;

  const impactRiskLevel =
    getScoreRiskLevel(
      impactScore
    );

  const situationText =
    data.situation?.summary ||
    data.situation?.situation_summary ||
    data.situation?.message ||
    data.alerts?.message ||
    "Situation summary unavailable.";

  const alertLevel =
    data.areaAlert?.alert_level ||
    data.alerts?.alert_level ||
    "N/A";

  if (false) {
    return (
      <section
        className="scenario-card"
        style={{ marginBottom: "24px" }}
      >
        <h3>Extended Flood Intelligence</h3>
        <p>
          Loading future weather, risk forecast,
          impact assessment, alerts and history...
        </p>
      </section>
    );
  }

  // ==========================================================
  // POTENTIAL FLOOD SOURCE / PROPAGATION HOTSPOT SCREENING
  // ==========================================================

  const hotspotBaseRows =
    Array.isArray(data.affectedDistricts)
      ? data.affectedDistricts
          .map((item, index) => {
            const currentRiskRaw =
              item.current_risk_score ??
              item.risk_score ??
              item.spatial_risk_score ??
              null;

            const impactRaw =
              item.potential_impact_score ??
              item.impact_score ??
              null;

            const currentRisk = Number(currentRiskRaw);
            const potentialImpact = Number(impactRaw);

            let hotspotScore = null;

            if (
              Number.isFinite(currentRisk) &&
              Number.isFinite(potentialImpact)
            ) {
              hotspotScore =
                (currentRisk * 0.5) +
                (potentialImpact * 0.5);
            } else if (Number.isFinite(currentRisk)) {
              hotspotScore = currentRisk;
            } else if (Number.isFinite(potentialImpact)) {
              hotspotScore = potentialImpact;
            }

            return {
              district:
                item.district ||
                item.NAME_2 ||
                `Candidate ${index + 1}`,
              currentRisk:
                Number.isFinite(currentRisk)
                  ? currentRisk
                  : null,
              potentialImpact:
                Number.isFinite(potentialImpact)
                  ? potentialImpact
                  : null,
              hotspotScore,
              impactLevel:
                item.potential_impact_level ||
                item.risk_level ||
                "Screened",
            };
          })
          .filter(
            (item) =>
              item.district &&
              (
                item.currentRisk !== null ||
                item.potentialImpact !== null
              )
          )
          .sort(
            (a, b) =>
              (b.hotspotScore ?? -1) -
              (a.hotspotScore ?? -1)
          )
      : [];

  const priorityHotspotName =
    priority.NAME_2 ||
    selectedDistrictName ||
    null;

  const hotspotRows = priorityHotspotName
    ? [
        {
          district: priorityHotspotName,
          currentRisk: Number.isFinite(
            Number(priority.spatial_risk_score)
          )
            ? Number(priority.spatial_risk_score)
            : null,
          potentialImpact: null,
          hotspotScore: Number.isFinite(
            Number(priority.spatial_risk_score)
          )
            ? Number(priority.spatial_risk_score)
            : null,
          impactLevel:
            priority.risk_level ||
            "Priority Area",
        },
        ...hotspotBaseRows.filter(
          (item) =>
            String(item.district).trim().toLowerCase() !==
            String(priorityHotspotName).trim().toLowerCase()
        ),
      ]
    : hotspotBaseRows;

  const displayedHotspotRows =
    hotspotRows.slice(0, 8);

  const highestHotspot =
    displayedHotspotRows.find(
      (item) =>
        typeof item.hotspotScore === "number"
    ) || null;

  // ==========================================================
  // CITY IMPACT PRIORITY SCREENING
  // ==========================================================

  const cityPriorityRows =
    affectedCityRows
      .map((item, index) => {
        const population =
          Number(item.population_2011);

        const districtRisk =
          Number(item.district_risk_score);

        const safePopulation =
          Number.isFinite(population) &&
          population >= 0
            ? population
            : null;

        const safeDistrictRisk =
          Number.isFinite(districtRisk)
            ? districtRisk
            : null;

        const riskWeightedExposure =
          safePopulation !== null &&
          safeDistrictRisk !== null
            ? safePopulation * safeDistrictRisk
            : null;

        return {
          city:
            item.city_name ||
            `Urban Place ${index + 1}`,
          population: safePopulation,
          districtRisk: safeDistrictRisk,
          riskLevel:
            item.district_risk_level ||
            getScoreRiskLevel(safeDistrictRisk),
          riskWeightedExposure,
        };
      })
      .filter(
        (item) =>
          item.population !== null &&
          item.population > 0
      )
      .sort(
        (a, b) =>
          (b.riskWeightedExposure ?? 0) -
          (a.riskWeightedExposure ?? 0)
      );

  const maxCityExposure =
    cityPriorityRows.length > 0
      ? Math.max(
          ...cityPriorityRows.map(
            (item) =>
              item.riskWeightedExposure ?? 0
          )
        )
      : 0;

  const cityPriorityDisplayRows =
    cityPriorityRows
      .map((item) => ({
        ...item,
        priorityScore:
          maxCityExposure > 0 &&
          item.riskWeightedExposure !== null
            ? (
                item.riskWeightedExposure /
                maxCityExposure
              ) * 100
            : null,
      }))
      .slice(0, 10);

  const highestCityPriority =
    cityPriorityDisplayRows[0] || null;

  return (
    <section
      style={{
        marginBottom: "24px",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
      }}
    >
      {error && (
        <div className="error-card">
          <strong>Extended feature warning:</strong>
          <p>{error}</p>
        </div>
      )}

      {/* =====================================================
          CURRENT DECISION SUPPORT
      ===================================================== */}

      <div data-page="dashboard"
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "16px",
        }}
      >
        <div className="summary-card">
          <span>State Alert Level</span>
          <strong
            style={{
              color: riskColor(data.alerts?.alert_level || alertLevel),
            }}
          >
            {data.alerts?.alert_level || alertLevel}
          </strong>
        </div>

        <div className="summary-card">
          <span>Priority Area</span>
          <strong>
            {priority.NAME_2 || "N/A"}
          </strong>
        </div>

        <div className="summary-card">
          <span>Priority Risk</span>
          <strong>
            {priority.risk_level || "N/A"}
          </strong>
        </div>

        <div className="summary-card">
          <span>Selected Area Alert</span>
          <strong>
            {selectedDistrictName || "Cuttack"}
            {data.areaAlert?.alert_level && (
              <span style={{ color: riskColor(data.areaAlert.alert_level), fontSize: "0.8em" }}>
                {" · "}{data.areaAlert.alert_level}
              </span>
            )}
          </strong>
        </div>
      </div>

      {/* =====================================================
          AI SITUATION SUMMARY
      ===================================================== */}

      <div data-page="dashboard" className="scenario-card">
        <h3>AI Situation Summary</h3>

        <p style={{ lineHeight: 1.6 }}>
          {situationText}
        </p>

        {Array.isArray(
          data.situation?.highlights
        ) && (
          <ul>
            {data.situation.highlights
              .slice(0, 5)
              .map((item, index) => (
                <li key={index}>{String(item)}</li>
              ))}
          </ul>
        )}
      </div>

      {/* =====================================================
          FLOOD RISK FORECAST + TIMELINE
      ===================================================== */}

      <div data-page="forecast" className="scenario-card">
        <h3>Flood Risk Forecast — Model Reference (CWC gauge data, Jan 2021)</h3>

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(140px, 1fr))",
            gap: "12px",
            marginTop: "12px",
          }}
        >
          {timelinePeriods.map(
            ({ label, day }) => {
              const item = timeline.find(
                (record) =>
                  Number(
                    record.forecast_day
                  ) === day
              );

              return (
                <div
                  className="summary-card"
                  key={label}
                >
                  <span>{label}</span>

                  <strong
                    style={{
                      color: riskColor(
                        item?.risk_level
                      ),
                    }}
                  >
                    {item?.risk_level || "N/A"}
                  </strong>

                  <small>
                    {typeof item?.water_level_m ===
                    "number"
                      ? `${item.water_level_m.toFixed(
                          2
                        )} m`
                      : typeof item
                          ?.predicted_water_level_m ===
                        "number"
                      ? `${item.predicted_water_level_m.toFixed(
                          2
                        )} m`
                      : "N/A"}
                  </small>
                </div>
              );
            }
          )}
        </div>

        <h4 style={{ marginTop: "20px" }}>
          7-Day Risk Forecast
        </h4>

        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Day</th>
                <th>Date</th>
                <th>Water Level</th>
                <th>Risk</th>
              </tr>
            </thead>

            <tbody>
              {data.forecastRisk
                .slice(0, 7)
                .map((item, index) => (
                  <tr
                    key={`${item.forecast_date || index}`}
                  >
                    <td>
                      Day {item.forecast_day}
                    </td>

                    <td>
                      {item.forecast_date || "N/A"}
                    </td>

                    <td>
                      {formatNumber(
                        item.predicted_water_level_m
                      )}{" "}
                      m
                    </td>

                    <td>
                      <span
                        className="risk-badge"
                        style={{
                          backgroundColor:
                            riskColor(
                              item.risk_level
                            ),
                        }}
                      >
                        {item.risk_level || "N/A"}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>

        <p style={{ marginTop: "12px" }}>
          Forecast method:{" "}
          <strong>
            Empirical water-level percentile proxy
          </strong>
          . This is a project screening indicator,
          not a calibrated flood probability.
        </p>
      </div>

      {/* =====================================================
          FUTURE WEATHER
      ===================================================== */}

      {/* =====================================================
          CURRENT WEATHER SNAPSHOT
      ===================================================== */}

      <div data-page="weather" className="scenario-card">
        <h3>
          Current Weather Snapshot —{" "}
          {selectedDistrictName || "Selected District"}
        </h3>

        {!currentWeatherSnapshot ? (
          <p>Current weather snapshot is unavailable.</p>
        ) : (
          <>
            <div
              className="metric-grid"
              style={{ marginTop: "12px" }}
            >
              <div className="metric-card">
                <span>Temperature</span>
                <strong>
                  {typeof currentWeatherSnapshot.temperature_c === "number"
                    ? `${currentWeatherSnapshot.temperature_c.toFixed(1)} °C`
                    : "N/A"}
                </strong>
              </div>

              <div className="metric-card">
                <span>Humidity</span>
                <strong>
                  {typeof currentWeatherSnapshot.humidity_percent === "number"
                    ? `${currentWeatherSnapshot.humidity_percent.toFixed(0)}%`
                    : "N/A"}
                </strong>
              </div>

              <div className="metric-card">
                <span>Rainfall</span>
                <strong>
                  {typeof currentWeatherSnapshot.rainfall_mm === "number"
                    ? `${currentWeatherSnapshot.rainfall_mm.toFixed(1)} mm`
                    : "N/A"}
                </strong>
              </div>

              <div className="metric-card">
                <span>Wind Speed</span>
                <strong>
                  {typeof currentWeatherSnapshot.wind_kmh === "number"
                    ? `${currentWeatherSnapshot.wind_kmh.toFixed(1)} km/h`
                    : "N/A"}
                </strong>
              </div>

              <div className="metric-card">
                <span>Pressure</span>
                <strong>
                  {typeof currentWeatherSnapshot.pressure_hpa === "number"
                    ? `${currentWeatherSnapshot.pressure_hpa.toFixed(1)} hPa`
                    : "N/A"}
                </strong>
              </div>
            </div>

            <p style={{ marginTop: "12px" }}>
              Nearest available hourly forecast time:{" "}
              <strong>
                {currentWeatherSnapshot.time
                  ? currentWeatherSnapshot.time.replace("T", " ")
                  : "N/A"}
              </strong>
            </p>

            <p style={{ fontSize: "0.85rem", opacity: 0.8 }}>
              Source: {data.weather?.source || "Open-Meteo forecast API"}.
              This is the nearest available hourly forecast snapshot for
              the selected district centroid. It is not a station observation
              and is not itself a flood prediction.
            </p>
          </>
        )}
      </div>
      <div data-page="weather" className="scenario-card">

        <h3>
          Future Weather Forecast —{" "}
          {selectedDistrictName || "Cuttack"}
        </h3>

        {weatherRows.length === 0 ? (

          <p>
            Future weather data is currently
            unavailable.
          </p>

        ) : (

          <div style={{ overflowX: "auto" }}>

            <table>

              <thead>

                <tr>
                  <th>Date</th>
                  <th>Max Temp</th>
                  <th>Min Temp</th>
                  <th>Rainfall</th>
                  <th>Max Wind</th>
                </tr>

              </thead>

              <tbody>

                {weatherRows.map(
                  (item, index) => (

                    <tr
                      key={
                        item.date ||
                        `weather-${index}`
                      }
                    >

                      <td>
                        {item.date || "N/A"}
                      </td>

                      <td>
                        {formatNumber(
                          item.temperature_max_c
                        )} °C
                      </td>

                      <td>
                        {formatNumber(
                          item.temperature_min_c
                        )} °C
                      </td>

                      <td>
                        {formatNumber(
                          item.rainfall_mm
                        )} mm
                      </td>

                      <td>
                        {formatNumber(
                          item.wind_max_kmh
                        )} km/h
                      </td>

                    </tr>

                  )
                )}

              </tbody>

            </table>

          </div>

        )}

        <p style={{ marginTop: "12px" }}>
          Forecast source: Open-Meteo forecast API.
          Weather is provided for the selected
          district centroid and is not itself a
          flood prediction.
        </p>

      </div>
      {/* =====================================================
          POTENTIALLY AFFECTED DISTRICTS
      ===================================================== */}

      <div data-page="impact" className="scenario-card">
        <h3>
          Potentially Affected Districts — if flooding starts in{" "}
          {selectedDistrictName || "the selected district"}
        </h3>
        <p style={{ marginTop: 0 }}>
          Neighbouring districts that water could reach from the selected district
          (change the district in the sidebar to see another source).
        </p>

        {(data.affectedFromSelected && data.affectedFromSelected.length
          ? data.affectedFromSelected
          : data.affectedDistricts
        ).filter(
          (item) =>
            String(item.district || item.NAME_2 || "").toLowerCase() !==
            String(selectedDistrictName || "").toLowerCase()
        ).length === 0 ? (
          <p>
            No affected-district screening result
            is currently available.
          </p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>District</th>
                  <th>Current Risk</th>
                  <th>Potential Impact</th>
                  <th>Level</th>
                </tr>
              </thead>

              <tbody>
                {(data.affectedFromSelected && data.affectedFromSelected.length
                  ? data.affectedFromSelected
                  : data.affectedDistricts)
                  .filter(
                    (item) =>
                      String(item.district || item.NAME_2 || "").toLowerCase() !==
                      String(selectedDistrictName || "").toLowerCase()
                  )
                  .slice(0, 10)
                  .map((item, index) => (
                    <tr
                      key={`${item.district}-${index}`}
                    >
                      <td>
                        {item.district ||
                          item.NAME_2 ||
                          "N/A"}
                      </td>

                      <td>
                        {formatNumber(
                          item.current_risk_score
                        )}
                      </td>

                      <td>
                        {formatNumber(
                          item.potential_impact_score
                        )}
                      </td>

                      <td>
                        <span
                          className="risk-badge"
                          style={{
                            backgroundColor:
                              riskColor(
                                item.potential_impact_level
                              ),
                          }}
                        >
                          {item.potential_impact_level ||
                            "N/A"}
                        </span>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}

        <p style={{ marginTop: "12px" }}>
          Propagation is a GIS screening based on
          adjacent districts, spatial risk and
          relative elevation. It is not hydraulic
          flood routing.
        </p>
      </div>

      {/* =====================================================
          POTENTIAL FLOOD SOURCE / PROPAGATION HOTSPOT SCREENING
      ===================================================== */}

      <div data-page="impact" className="scenario-card">
        <h3>
          Potential Flood Source / Propagation Hotspot Screening
        </h3>

        <p>
          State-wide view: districts that combine higher spatial risk
          with higher potential propagation impact, starting from the
          state's highest-risk district (priority area).
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "12px",
            marginTop: "12px",
          }}
        >
          <div className="summary-card">
            <span>Priority Hotspot Area</span>
            <strong>
              {priorityHotspotName || "N/A"}
            </strong>
          </div>

          <div className="summary-card">
            <span>Districts Screened</span>
            <strong>
              {hotspotRows.length}
            </strong>
          </div>

          <div className="summary-card">
            <span>Highest Screening Score</span>
            <strong>
              {highestHotspot
                ? formatNumber(
                    highestHotspot.hotspotScore
                  )
                : "N/A"}
            </strong>
          </div>
        </div>

        {displayedHotspotRows.length === 0 ? (
          <p style={{ marginTop: "12px" }}>
            No propagation-hotspot screening records are
            currently available.
          </p>
        ) : (
          <div
            style={{
              overflowX: "auto",
              marginTop: "14px",
            }}
          >
            <table>
              <thead>
                <tr>
                  <th>District</th>
                  <th>Current Risk</th>
                  <th>Potential Impact</th>
                  <th>Hotspot Score</th>
                  <th>Screening Level</th>
                </tr>
              </thead>

              <tbody>
                {displayedHotspotRows.map(
                  (item, index) => (
                    <tr
                      key={`hotspot-${item.district}-${index}`}
                    >
                      <td>
                        {item.district}
                      </td>

                      <td>
                        {formatNumber(
                          item.currentRisk
                        )}
                      </td>

                      <td>
                        {formatNumber(
                          item.potentialImpact
                        )}
                      </td>

                      <td>
                        {formatNumber(
                          item.hotspotScore
                        )}
                      </td>

                      <td>
                        <span
                          className="risk-badge"
                          style={{
                            backgroundColor:
                              riskColor(
                                item.impactLevel
                              ),
                          }}
                        >
                          {item.impactLevel}
                        </span>
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        )}

        <p
          style={{
            marginTop: "12px",
            fontSize: "0.85rem",
            opacity: 0.8,
          }}
        >
          Method: 50% current spatial-risk score +
          50% potential propagation-impact score when both
          are available. This is a GIS screening indicator;
          it does not identify an exact physical flood source
          and does not perform hydraulic flood routing.
        </p>
      </div>

      {/* =====================================================
          VULNERABILITY + IMPACT
      ===================================================== */}

      <div data-page="impact"
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "16px",
        }}
      >
        <div className="scenario-card">
          <h3>Vulnerability Assessment</h3>

          <p>
            District:{" "}
            <strong>
              {selectedDistrictName || "N/A"}
            </strong>
          </p>

          <p>
            Long-term spatial risk score:{" "}
            <strong>
              {formatNumber(
                vulnerabilityScore
              )}
            </strong>
          </p>

          <p style={{ fontSize: "0.85rem", opacity: 0.8 }}>
            Based on elevation, drainage, distance to river and past
            floods. It changes only when the GIS data changes.
          </p>

          <p>
            Risk Level:{" "}
            <strong>
              {vulnerabilityRiskLevel}
            </strong>
          </p>
        </div>

        <div className="scenario-card">
          <h3>Impact Assessment</h3>

          <p>
            District:{" "}
            <strong>
              {selectedDistrictName || "N/A"}
            </strong>
          </p>

          <p>
            Live 7-day flood outlook:{" "}
            <strong
              style={{
                color: riskColor(
                  data.areaAlert?.live_outlook?.peak_level || impactRiskLevel
                ),
              }}
            >
              {data.areaAlert?.live_outlook?.peak_level || impactRiskLevel}
            </strong>
            {data.areaAlert?.live_outlook?.peak_date
              ? ` (highest on ${data.areaAlert.live_outlook.peak_date})`
              : ""}
          </p>

          <p>
            Today:{" "}
            <strong>
              {data.areaAlert?.live_outlook?.today_level || "N/A"}
            </strong>
          </p>

          <p>
            Estimated Population at Risk:{" "}
            <strong>
              {formatWholeNumber(
                selectedPopulationImpact
                  ?.estimated_population_at_risk
              )}
            </strong>
          </p>

          <p>
            WorldPop 2020 Population:{" "}
            <strong>
              {formatWholeNumber(
                selectedPopulationImpact
                  ?.population_worldpop_2020
              )}
            </strong>
          </p>

          <p>
            Population Exposure Basis:{" "}
            <strong>
              {selectedPopulationImpact
                ? "2020 WorldPop × spatial risk score"
                : "Dataset unavailable"}
            </strong>
          </p>

          <p>
            Infrastructure Exposure:{" "}
            <strong>
              {selectedImpact
                ?.infrastructure_exposure ??
                "See Emergency Response page (hospitals, schools, roads from OpenStreetMap)"}
            </strong>
          </p>
        </div>
      </div>

      {/* =====================================================
          POTENTIALLY AFFECTED CITIES (merged into the priority table below)
      ===================================================== */}

      <div data-page="merged-hidden" className="scenario-card">
        <h3>
          Potentially Affected Cities —{" "}
          {selectedDistrictName || "Cuttack"}
        </h3>

        <p>
          Census urban places linked to the selected district.
          Population values are from Census 2011.
        </p>

        <p>
          Urban Places:{" "}
          <strong>
            {data.affectedCities?.city_count ??
              affectedCityRows.length}
          </strong>
        </p>

        <p>
          Total Urban Population (2011):{" "}
          <strong>
            {formatWholeNumber(
              data.affectedCities
                ?.total_urban_population_2011
            )}
          </strong>
        </p>

        {affectedCityRows.length === 0 ? (
          <p>
            No city-level screening records are currently
            available for the selected district.
          </p>
        ) : (
          <div
            style={{
              overflowX: "auto",
              marginTop: "12px",
            }}
          >
            <table>
              <thead>
                <tr>
                  <th>City / Urban Place</th>
                  <th>Population (2011)</th>
                  <th>District Risk</th>
                  <th>Level</th>
                </tr>
              </thead>
              <tbody>
                {affectedCityRows
                  .slice(0, 12)
                  .map((item, index) => (
                    <tr
                      key={`${item.city_name}-${index}`}
                    >
                      <td>
                        {item.city_name || "N/A"}
                      </td>
                      <td>
                        {formatWholeNumber(
                          item.population_2011
                        )}
                      </td>
                      <td>
                        {formatNumber(
                          item.district_risk_score
                        )}
                      </td>
                      <td
                        style={{
                          color: riskColor(
                            item.district_risk_level
                          ),
                          fontWeight: 700,
                        }}
                      >
                        {item.district_risk_level ||
                          "N/A"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}

        <p
          style={{
            marginTop: "12px",
            fontSize: "0.85rem",
            opacity: 0.8,
          }}
        >
          Screening basis: the selected district's existing
          spatial risk score is linked to its Census urban
          places. This is not city-level hydraulic inundation
          modelling, and city coordinates are not inferred.
        </p>
      </div>

      {/* =====================================================

          {/* =====================================================
          CITY IMPACT PRIORITY SCREENING
      ===================================================== */}

      <div data-page="impact" className="scenario-card">
        <h3>
          Affected Towns &amp; Priority —{" "}
          {selectedDistrictName || "Cuttack"}
        </h3>

        <p>
          Urban places are screened using Census 2011
          population and the selected district's existing
          spatial-risk score.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "12px",
            marginTop: "12px",
          }}
        >
          <div className="summary-card">
            <span>Cities Screened</span>
            <strong>
              {cityPriorityRows.length}
            </strong>
          </div>

          <div className="summary-card">
            <span>Total Urban Population</span>
            <strong>
              {formatWholeNumber(
                data.affectedCities
                  ?.total_urban_population_2011
              )}
            </strong>
          </div>

          <div className="summary-card">
            <span>Highest Priority City</span>
            <strong>
              {highestCityPriority?.city ||
                "N/A"}
            </strong>
          </div>
        </div>

        {cityPriorityDisplayRows.length === 0 ? (
          <p style={{ marginTop: "12px" }}>
            No city-impact screening records are currently
            available for the selected district.
          </p>
        ) : (
          <div
            style={{
              overflowX: "auto",
              marginTop: "14px",
            }}
          >
            <table>
              <thead>
                <tr>
                  <th>City / Urban Place</th>
                  <th>Population (2011)</th>
                  <th>District Risk</th>
                  <th>Risk-Weighted Exposure</th>
                  <th>Priority Score</th>
                </tr>
              </thead>

              <tbody>
                {cityPriorityDisplayRows.map(
                  (item, index) => (
                    <tr
                      key={`city-priority-${item.city}-${index}`}
                    >
                      <td>
                        {item.city}
                      </td>

                      <td>
                        {formatWholeNumber(
                          item.population
                        )}
                      </td>

                      <td>
                        <span
                          className="risk-badge"
                          style={{
                            backgroundColor:
                              riskColor(
                                item.riskLevel
                              ),
                          }}
                        >
                          {formatNumber(
                            item.districtRisk
                          )}
                        </span>
                      </td>

                      <td>
                        {formatWholeNumber(
                          item.riskWeightedExposure
                        )}
                      </td>

                      <td>
                        {item.priorityScore !== null
                          ? `${item.priorityScore.toFixed(1)} / 100`
                          : "N/A"}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        )}

        <p
          style={{
            marginTop: "12px",
            fontSize: "0.85rem",
            opacity: 0.8,
          }}
        >
          Method: risk-weighted exposure =
          population × district spatial-risk score.
          Priority score is normalized against the highest
          screened city in the selected district. This is an
          exposure-priority indicator, not confirmed city-level
          inundation, flood depth, evacuation demand, or
          hydraulic modelling.
        </p>
      </div>

      {historicalWeather?.available && (
            <section data-page="weather"
              className="scenario-card"
              style={{ marginTop: "24px" }}
            >
              <div style={{ marginBottom: "18px" }}>
                <div
                  style={{
                    fontSize: "12px",
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    opacity: 0.75
                  }}
                >
                  HISTORICAL WEATHER ANALYSIS
                </div>

                <h3 style={{ margin: "6px 0 8px" }}>
                  Historical Weather Analysis
                </h3>

                <p style={{ margin: 0, opacity: 0.78, lineHeight: 1.5 }}>
                  {historicalWeather.period.start_year} - {historicalWeather.period.end_year}
                  {" "}historical weather reference from{" "}
                  {historicalWeather.reference_location}. This section
                  describes historical weather conditions and does not
                  replace the flood-risk model.
                </p>
              </div>

              <div
                className="metric-grid"
                style={{
                  marginBottom: "20px"
                }}
              >
                <div className="metric-card">
                  <span>Temperature Trend</span>
                  <strong>
                    {historicalWeather.overall.temperature_trend}
                  </strong>
                  <small>
                    {historicalWeather.overall.temperature_change_c > 0 ? "+" : ""}
                    {historicalWeather.overall.temperature_change_c} °C
                  </small>
                </div>

                <div className="metric-card">
                  <span>Humidity Trend</span>
                  <strong>
                    {historicalWeather.overall.humidity_trend}
                  </strong>
                  <small>
                    {historicalWeather.overall.humidity_change_pct > 0 ? "+" : ""}
                    {historicalWeather.overall.humidity_change_pct} %
                  </small>
                </div>

                <div className="metric-card">
                  <span>Wind Trend</span>
                  <strong>
                    {historicalWeather.overall.wind_trend}
                  </strong>
                  <small>
                    {historicalWeather.overall.wind_change_kmh > 0 ? "+" : ""}
                    {historicalWeather.overall.wind_change_kmh} km/h
                  </small>
                </div>

                <div className="metric-card">
                  <span>Pressure Trend</span>
                  <strong>
                    {historicalWeather.overall.pressure_trend}
                  </strong>
                  <small>
                    {historicalWeather.overall.pressure_change_hpa > 0 ? "+" : ""}
                    {historicalWeather.overall.pressure_change_hpa} hPa
                  </small>
                </div>
              </div>

              <h4>Weather Trend Interpretation</h4>

              <div
                style={{
                  display: "grid",
                  gap: "8px",
                  marginBottom: "20px"
                }}
              >
                {historicalWeather.interpretation.map(
                  (item, index) => (
                    <div
                      key={index}
                      style={{
                        padding: "10px 12px",
                        borderRadius: "10px",
                        background: "rgba(148,163,184,0.06)",
                        lineHeight: 1.45
                      }}
                    >
                      {item}
                    </div>
                  )
                )}
              </div>

              <h4>Annual Historical Summary</h4>

              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    minWidth: "720px"
                  }}
                >
                  <thead>
                    <tr>
                      <th>Year</th>
                      <th>Avg Temp °C</th>
                      <th>Avg Humidity %</th>
                      <th>Avg Wind km/h</th>
                      <th>Avg Pressure hPa</th>
                      <th>Records</th>
                    </tr>
                  </thead>

                  <tbody>
                    {historicalWeather.annual.map((row) => (
                      <tr key={row.year}>
                        <td>{row.year}</td>
                        <td style={{ textAlign: "right" }}>
                          {row.avg_temperature_c ?? row.average_temperature_c ?? "N/A"}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {row.avg_humidity_pct ?? row.avg_humidity_percent ?? row.average_humidity_percent ?? "N/A"}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {row.avg_wind_speed_kmh ?? row.avg_wind_speed ?? row.average_wind_speed ?? "N/A"}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {row.avg_pressure_hpa ?? row.avg_pressure ?? row.average_pressure ?? "N/A"}
                        </td>
                        <td style={{ textAlign: "right" }}>
                          {row.records}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p
                style={{
                  marginTop: "16px",
                  marginBottom: 0,
                  fontSize: "0.85rem",
                  opacity: 0.7,
                  lineHeight: 1.5
                }}
              >
                Source: {historicalWeather.source}. Historical rainfall
                is not present in this dataset. The weather series is a
                Bhubaneswar reference-point dataset and should not be
                interpreted as district-station observations.
              </p>
            </section>
          )}

      {/* =====================================================
          RAINFALL ANOMALY DETECTION
      ===================================================== */}
      <div data-page="weather" className="scenario-card">
        <h3>Rainfall Anomaly Detection — {selectedDistrictName || "Selected District"}</h3>

        {!data.rainfallAnomaly?.available ? (
          <p>
            Rainfall anomaly analysis is currently unavailable.
            {data.rainfallAnomaly?.message
              ? ` ${data.rainfallAnomaly.message}`
              : ""}
          </p>
        ) : !selectedRainfallAnomaly ? (
          <p>No rainfall anomaly record is available for the selected district.</p>
        ) : (
          <>
            <div className="metric-grid" style={{ marginTop: "12px" }}>
              <div className="metric-card"><span>Latest Rainfall</span><strong>{formatNumber(selectedRainfallAnomaly.rainfall_mm)} mm</strong></div>
              <div className="metric-card"><span>Historical Median</span><strong>{formatNumber(selectedRainfallAnomaly.median)} mm</strong></div>
              <div className="metric-card"><span>Anomaly</span><strong>{formatNumber(selectedRainfallAnomaly.anomaly_mm)} mm</strong></div>
              <div className="metric-card"><span>Anomaly Level</span><strong style={{ color: riskColor(selectedRainfallAnomaly.anomaly_level) }}>{selectedRainfallAnomaly.anomaly_level || "Normal"}</strong></div>
            </div>

            <p style={{ marginTop: "12px" }}>Latest available rainfall date: {data.rainfallAnomaly.latest_date || "N/A"}</p>
            <p>Anomaly percentage: {typeof selectedRainfallAnomaly.anomaly_percent === "number" ? `${selectedRainfallAnomaly.anomaly_percent.toFixed(1)}%` : "N/A"}</p>
            <p>Z-score: {typeof selectedRainfallAnomaly.z_score === "number" ? selectedRainfallAnomaly.z_score.toFixed(2) : "N/A"}</p>
            <p style={{ fontSize: "0.85rem", opacity: 0.8 }}>
              Method: latest daily district rainfall compared with the district historical median and variation in the available IMD dataset. This is an anomaly indicator, not a flood probability or official warning.
            </p>
          </>
        )}
      </div>

      {/* =====================================================
          AREA ALERT
      ===================================================== */}

      <div data-page="weather" className="scenario-card">
        <h3>
          Area-Specific Alert —{" "}
          {selectedDistrictName || "Cuttack"}
        </h3>

        <p>
          Alert Level:{" "}
          <strong
            style={{
              color: riskColor(
                data.areaAlert?.alert_level
              ),
            }}
          >
            {data.areaAlert?.alert_level || alertLevel}
          </strong>
        </p>

        <p>
          {data.areaAlert?.message ||
            data.areaAlert?.summary ||
            "Area-specific alert information is unavailable."}
        </p>
      </div>

      {/* =====================================================
          HISTORY
      ===================================================== */}

      {/* =====================================================
          SCENARIO COMPARISON
      ===================================================== */}

      <div data-page="whatif" className="scenario-card">
        <h3>Scenario Comparison</h3>

        {scenarioComparisonRows.length === 0 ? (
          <p>
            Scenario comparison data is currently unavailable.
          </p>
        ) : (
          <>
            <p>
              Comparison of the predefined What-If scenarios
              using the project's spatial-risk framework.
            </p>

            <div style={{ overflowX: "auto", marginTop: "12px" }}>
              <table>
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Average Risk</th>
                    <th>Maximum Risk (1.0 = highest possible)</th>
                    <th>Moderate</th>
                    <th>High</th>
                    <th>Very High</th>
                  </tr>
                </thead>

                <tbody>
                  {scenarioComparisonRows.map(
                    (item, index) => (
                      <tr
                        key={
                          item.scenario ||
                          `comparison-${index}`
                        }
                      >
                        <td>
                          <strong>
                            {item.scenario || "Scenario"}
                          </strong>
                        </td>

                        <td>
                          {typeof item.average_risk_score ===
                          "number"
                            ? item.average_risk_score.toFixed(3)
                            : "N/A"}
                        </td>

                        <td>
                          {typeof item.maximum_risk_score ===
                          "number"
                            ? item.maximum_risk_score.toFixed(3)
                            : "N/A"}
                        </td>

                        <td>
                          {item.moderate_districts ?? "N/A"}
                        </td>

                        <td>
                          {item.high_districts ?? "N/A"}
                        </td>

                        <td>
                          {item.very_high_districts ?? "N/A"}
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>

            {baseScenario && combinedScenario && (
              <div
                className="metric-grid"
                style={{ marginTop: "16px" }}
              >
                <div className="metric-card">
                  <span>Average Risk Change</span>
                  <strong>
                    {(
                      combinedScenario.average_risk_score -
                      baseScenario.average_risk_score
                    ).toFixed(3)}
                  </strong>
                </div>

                <div className="metric-card">
                  <span>Maximum Risk Change</span>
                  <strong>
                    {(
                      combinedScenario.maximum_risk_score -
                      baseScenario.maximum_risk_score
                    ).toFixed(3)}
                  </strong>
                </div>

                <div className="metric-card">
                  <span>Change in "High" districts (they moved up to Very High)</span>
                  <strong>
                    {combinedScenario.high_districts -
                      baseScenario.high_districts >= 0
                      ? "+"
                      : ""}
                    {combinedScenario.high_districts -
                      baseScenario.high_districts}
                  </strong>
                </div>

                <div className="metric-card">
                  <span>Change in "Very High" districts</span>
                  <strong>
                    {combinedScenario.very_high_districts -
                      baseScenario.very_high_districts >= 0
                      ? "+"
                      : ""}
                    {combinedScenario.very_high_districts -
                      baseScenario.very_high_districts}
                  </strong>
                </div>
              </div>
            )}

            <p
              style={{
                marginTop: "12px",
                fontSize: "0.85rem",
                opacity: 0.8,
              }}
            >
              These are predefined project scenarios.
              They are comparative screening indicators,
              not hydraulic flood simulations or official
              warnings.
            </p>
          </>
        )}
      </div>
      {/* =====================================================
          SAVED ANALYSIS RECORDS
      ===================================================== */}

      <div data-page="data" className="scenario-card">
        <h3>Saved Analysis Records</h3>

        <div
          className="metric-grid"
          style={{ marginTop: "12px" }}
        >
          <div className="metric-card">
            <span>Predictions Saved</span>
            <strong>{data.predictionHistory.length}</strong>
          </div>

          <div className="metric-card">
            <span>Alerts Saved</span>
            <strong>{data.alertHistory.length}</strong>
          </div>

          <div className="metric-card">
            <span>Scenarios Saved</span>
            <strong>{data.scenarioHistory.length}</strong>
          </div>

          <div className="metric-card">
            <span>Reports Saved</span>
            <strong>{data.reportHistory.length}</strong>
          </div>
        </div>

        <h4 style={{ marginTop: "20px" }}>
          Prediction History
        </h4>

        {data.predictionHistory.length === 0 ? (
          <p>No saved prediction records.</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Date / Time</th>
                  <th>Source</th>
                  <th>Reference Date</th>
                  <th>Prediction</th>
                  <th>Risk</th>
                  <th>Probability</th>
                </tr>
              </thead>

              <tbody>
                {data.predictionHistory.map(
                  (item, index) => (
                    <tr key={item.id || index}>
                      <td>
                        {item.created_at || "N/A"}
                      </td>

                      <td>
                        {item.source || "N/A"}
                      </td>

                      <td>
                        {item.reference_date || "N/A"}
                      </td>

                      <td>
                        {item.prediction_label || "N/A"}
                      </td>

                      <td>
                        <strong>
                          {item.risk_level || "N/A"}
                        </strong>
                      </td>

                      <td>
                        {typeof item.flood_probability_percent ===
                        "number"
                          ? `${item.flood_probability_percent.toFixed(2)}%`
                          : "N/A"}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        )}

        <h4 style={{ marginTop: "20px" }}>
          Alert History
        </h4>

        {data.alertHistory.length === 0 ? (
          <p>No saved alert records.</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Date / Time</th>
                  <th>Status</th>
                  <th>Alert Level</th>
                  <th>Priority Districts</th>
                  <th>Message</th>
                </tr>
              </thead>

              <tbody>
                {data.alertHistory.map(
                  (item, index) => (
                    <tr key={item.id || index}>
                      <td>
                        {item.created_at || "N/A"}
                      </td>

                      <td>
                        <strong>
                          {item.overall_status || "N/A"}
                        </strong>
                      </td>

                      <td>
                        {item.alert_level || "N/A"}
                      </td>

                      <td>
                        {item.priority_district_count ??
                          item.priority_districts?.length ??
                          0}
                      </td>

                      <td>
                        {item.message || "N/A"}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        )}

        <h4 style={{ marginTop: "20px" }}>
          Scenario History
        </h4>

        {data.scenarioHistory.length === 0 ? (
          <p>
            No saved custom What-If scenario records yet.
            Predefined scenario results are shown in
            Scenario Comparison above.
          </p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Date / Time</th>
                  <th>Rainfall Change</th>
                  <th>Water-Level Change</th>
                  <th>Drainage Change</th>
                  <th>Average Risk</th>
                </tr>
              </thead>

              <tbody>
                {data.scenarioHistory.map(
                  (item, index) => (
                    <tr key={item.id || index}>
                      <td>
                        {item.created_at || "N/A"}
                      </td>

                      <td>
                        {item.rainfall_change_percent ?? "N/A"}%
                      </td>

                      <td>
                        {item.water_level_change_m ?? "N/A"} m
                      </td>

                      <td>
                        {item.drainage_effectiveness_change_percent ??
                          "N/A"}%
                      </td>

                      <td>
                        {typeof item.average_risk_score ===
                        "number"
                          ? item.average_risk_score.toFixed(3)
                          : "N/A"}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        )}

        <h4 style={{ marginTop: "20px" }}>
          Report History
        </h4>

        {data.reportHistory.length === 0 ? (
          <p>
            No saved report records yet. Generate a risk
            summary report to create a report-history entry.
          </p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Date / Time</th>
                  <th>Report Type</th>
                </tr>
              </thead>

              <tbody>
                {data.reportHistory.map(
                  (item, index) => (
                    <tr key={item.id || index}>
                      <td>
                        {item.created_at || "N/A"}
                      </td>

                      <td>
                        {item.report_type || "N/A"}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        )}

        <p
          style={{
            marginTop: "12px",
            fontSize: "0.85rem",
            opacity: 0.8,
          }}
        >
          History shows records actually persisted by the
          Flood Guard AI SQLite database. Empty sections are
          not populated with synthetic records.
        </p>
      </div>

{/* =====================================================
          DATA / FEATURE STATUS
      ===================================================== */}

      <div data-page="data" className="scenario-card">
        <h3>Data Sources & Feature Readiness</h3>

        {data.readiness?.features && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(220px, 1fr))",
              gap: "8px",
              marginBottom: "16px",
            }}
          >
            {Object.entries(
              data.readiness.features
            ).map(([name, available]) => (
              <div key={name}>
                <strong>
                  {available ? "✓" : "—"}{" "}
                </strong>
                {name.replaceAll("_", " ")}
              </div>
            ))}
          </div>
        )}

        {Array.isArray(
          data.dataSources?.sources
        ) && (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Status</th>
                </tr>
              </thead>

              <tbody>
                {data.dataSources.sources
                  .slice(0, 12)
                  .map((source, index) => (
                    <tr key={index}>
                      <td>
                        {source.name ||
                          source.dataset ||
                          source.source ||
                          "N/A"}
                      </td>

                      <td>
                        {source.status ||
                          (source.available
                            ? "Available"
                            : "Unavailable")}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}

        <p style={{ marginTop: "12px" }}>
          City and population screening are available.
          Emergency GIS intelligence is integrated with live GIS
          services when available. Infrastructure, shelter, road
          and evacuation results are live screening outputs and
          may show no records when verified facilities or routing
          data are unavailable.
        </p>
      </div>
    </section>
  );
}





















