import { useEffect, useMemo, useState } from "react";

import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
} from "react-leaflet";

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

function score(value) {
  return typeof value === "number"
    ? value.toFixed(3)
    : "N/A";
}

function number(value, digits = 2) {
  return typeof value === "number"
    ? value.toFixed(digits)
    : "N/A";
}

function Metric({ label, value }) {
  return (
    <div className="scenario-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Badge({ level }) {
  return (
    <span
      className="risk-badge"
      style={{
        backgroundColor: riskColor(level),
      }}
    >
      {level || "N/A"}
    </span>
  );
}

export default function EmergencyGISIntelligence({
  selectedDistrictName,
}) {
  const [data, setData] = useState({
    infrastructure: null,
    shelters: null,
    roads: null,
    evacuation: null,
    cities: null,
    inundation: null,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [finalInundationResult, setFinalInundationResult] = useState(null);
  const [inundationResult, setInundationResult] = useState(null);

  useEffect(() => {
    if (!selectedDistrictName) {
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");

      const endpoints = [
        `infrastructure?district=${encodeURIComponent(
          selectedDistrictName
        )}`,
        `shelters?district=${encodeURIComponent(
          selectedDistrictName
        )}`,
        `roads?district=${encodeURIComponent(
          selectedDistrictName
        )}`,
        `evacuation-route?district=${encodeURIComponent(
          selectedDistrictName
        )}`,
        `city-risk?district=${encodeURIComponent(
          selectedDistrictName
        )}`,
        `inundation-screening?district=${encodeURIComponent(
          selectedDistrictName
        )}`,
      ];

      const endpointKeys = [
        "infrastructure",
        "shelters",
        "roads",
        "evacuation",
        "cities",
        "inundation",
      ];

      const fetchGIS = async (endpoint, index) => {
        const controller = new AbortController();

        const timeoutId = setTimeout(() => {
          controller.abort();
        }, 60000);

        try {
          const response = await fetch(
            `${API_BASE}/${endpoint}`,
            {
              signal: controller.signal,
            }
          );

          let payload;

          try {
            payload = await response.json();
          } catch {
            payload = {
              status: "error",
              available: false,
              message: "Invalid backend response.",
            };
          }

          if (!response.ok) {
            payload = {
              ...payload,
              status: "error",
              available: false,
              message:
                payload?.message ||
                `GIS request failed (${response.status}).`,
            };
          }

          if (!cancelled) {
            if (index === 5) {               setInundationResult(payload);             } 
            setData((previous) => ({
              ...previous,
              [endpointKeys[index]]: payload,
            }));
          }

          return payload;
        } catch (err) {
          const payload = {
            status: "error",
            available: false,
            message:
              err?.name === "AbortError"
                ? "Live GIS source timed out (OpenStreetMap was slow). Refresh in a minute; results are saved once loaded."
                : err?.message ||
                  "Live GIS source could not be reached.",
          };

          if (!cancelled) {
            setData((previous) => ({
              ...previous,
              [endpointKeys[index]]: payload,
            }));
          }

          return payload;
        } finally {
          clearTimeout(timeoutId);
        }
      };

      try {
        /*
         * IMPORTANT:
         * Inundation screening is a fast local calculation.
         * Run it first so it is not blocked by OSM/OSRM services.
         *
         * The remaining GIS services are requested sequentially
         * because several backend GIS operations use blocking I/O.
         */
        const requestOrder = [5, 0, 1, 2, 3, 4];

        const failed = [];

        for (const index of requestOrder) {
          if (cancelled) {
            return;
          }

          const payload = await fetchGIS(
            endpoints[index],
            index
          );

          /*
           * Inundation screening is the fast local calculation.
           * Stop the main loading indicator immediately after it
           * succeeds so slower external GIS services cannot hide
           * an already-available result.
           */
          if (index === 5 && !cancelled) {
            setLoading(false);
          }

          if (payload?.status === "error") {
            failed.push(payload);
          }
        }

        if (cancelled) {
          return;
        }

        if (failed.length > 0) {
          setError(
            failed[0]?.message ||
              "Some live GIS services were unavailable; available screening results are shown."
          );
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err?.message ||
              "Emergency GIS data could not be loaded."
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }    load();

    return () => {
      cancelled = true;
    };
  }, [selectedDistrictName]);


  // FINAL DIRECT INUNDATION FETCH
  useEffect(() => {
    if (!selectedDistrictName) {
      return;
    }

    let active = true;

    fetch(`${API_BASE}/inundation-screening?district=${encodeURIComponent(selectedDistrictName)}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(HTTP );
        }
        return response.json();
      })
      .then((payload) => {
        if (active && payload?.status === 'success') {
          setFinalInundationResult(payload);
        }
      })
      .catch((err) => {
        console.warn('Final inundation fetch failed:', err);
      });

    return () => {
      active = false;
    };
  }, [selectedDistrictName]);

  const infrastructure = data.infrastructure;
  const shelters = data.shelters;
  const roads = data.roads;
  const evacuation = data.evacuation;
  const cities = data.cities;
  const inundation = finalInundationResult || inundationResult || data.inundation;

  const assets = infrastructure?.assets || [];
  const facilities = shelters?.facilities || [];
  const roadSegments = roads?.roads || [];
  const cityRows = cities?.cities || [];

  const mapCenter = useMemo(() => {
    if (infrastructure?.assets?.[0]) {
      return [
        infrastructure.assets[0].latitude,
        infrastructure.assets[0].longitude,
      ];
    }

    if (facilities?.[0]) {
      return [
        facilities[0].latitude,
        facilities[0].longitude,
      ];
    }

    if (evacuation?.origin) {
      return [
        evacuation.origin.latitude,
        evacuation.origin.longitude,
      ];
    }

    return [20.2961, 85.8245];
  }, [infrastructure, facilities, evacuation]);

  if (!selectedDistrictName) {
    return null;
  }

  return (
    <section className="status-section">
      <div className="panel status-panel">
        <div className="status-header">
          <div>
            <h3>
              Emergency GIS Intelligence —{" "}
              {selectedDistrictName}
            </h3>
            <p>
              Infrastructure, shelters, roads, evacuation
              planning, city spatial risk and inundation
              screening.
            </p>
          </div>

          <span className="prototype-badge">
            {loading ? "Loading GIS" : "Rapid GIS Screening"}
          </span>
        </div>

        {error && (
          <div className="scenario-note">
            <strong>Live-data notice:</strong>{" "}
            {error}
          </div>
        )}

        <div className="scenario-grid">
          <Metric
            label="Infrastructure Assets"
            value={
              infrastructure?.asset_count ?? "N/A"
            }
          />

          <Metric
            label="High-Risk Assets"
            value={
              infrastructure?.high_risk_asset_count ?? "N/A"
            }
          />

          <Metric
            label="Infrastructure Damage Risk"
            value={
              infrastructure?.damage_risk != null
                ? `${score(infrastructure.damage_risk)} ${infrastructure.damage_risk_level || ""}`
                : "N/A"
            }
          />

          <Metric
            label="Mapped Shelters / Facilities"
            value={
              shelters?.facility_count ?? "N/A"
            }
          />

          <Metric
            label="Major Roads"
            value={
              roads?.major_road_km != null
                ? `${number(roads.major_road_km)} km`
                : "N/A"
            }
          />

          <Metric
            label="Road Exposure Risk"
            value={
              roads?.road_exposure_index != null
                ? `${score(roads.road_exposure_index)} ${roads.road_risk_level || ""}`
                : "N/A"
            }
          />

          <Metric
            label="Best Safe-Facility Score"
            value={
              shelters?.best_facility?.suitability_score != null
                ? score(
                    shelters.best_facility.suitability_score
                  )
                : "N/A"
            }
          />

          <Metric
            label="Inundation Screening"
            value={
              inundation?.water_level_adjusted_index != null
                ? `${score(inundation.water_level_adjusted_index)} ${inundation.risk_level || ""}`
                : "N/A"
            }
          />
        </div>

        <div className="scenario-note">
          <strong>Decision-support scope:</strong>{" "}
          These emergency GIS outputs are planning/screening
          indicators. Infrastructure damage is not an
          engineering loss estimate, mapped facilities are
          not automatically verified emergency shelters, and
          routes do not account for live road closures.
        </div>

        <div
          style={{
            height: 420,
            width: "100%",
            marginTop: 20,
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          <MapContainer
            center={mapCenter}
            zoom={10}
            scrollWheelZoom
            style={{
              height: "100%",
              width: "100%",
            }}
          >
            <TileLayer
              attribution="&copy; OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {roadSegments.map((road) => (
              <Polyline
                key={`road-${road.osm_id}`}
                positions={road.coordinates || []}
                pathOptions={{
                  color: riskColor(road.risk_level),
                  weight: 3,
                  opacity: 0.65,
                }}
              >
                <Popup>
                  <strong>{road.name}</strong>
                  <br />
                  Class: {road.road_class}
                  <br />
                  Risk: {score(road.risk_score)}
                  <br />
                  Level: {road.risk_level}
                </Popup>
              </Polyline>
            ))}

            {assets.map((asset) => (
              <CircleMarker
                key={`asset-${asset.osm_id}`}
                center={[
                  asset.latitude,
                  asset.longitude,
                ]}
                radius={7}
                pathOptions={{
                  color: riskColor(asset.risk_level),
                  fillColor: riskColor(
                    asset.risk_level
                  ),
                  fillOpacity: 0.85,
                }}
              >
                <Popup>
                  <strong>{asset.name}</strong>
                  <br />
                  Type: {asset.category}
                  <br />
                  Risk: {score(asset.risk_score)}
                  <br />
                  River distance:{" "}
                  {number(
                    asset.distance_to_river_m,
                    0
                  )} m
                </Popup>
              </CircleMarker>
            ))}

            {facilities.map((facility) => (
              <CircleMarker
                key={`facility-${facility.osm_id}`}
                center={[
                  facility.latitude,
                  facility.longitude,
                ]}
                radius={8}
                pathOptions={{
                  color: "#2563eb",
                  fillColor: "#2563eb",
                  fillOpacity: 0.85,
                }}
              >
                <Popup>
                  <strong>{facility.name}</strong>
                  <br />
                  Type: {facility.facility_type}
                  <br />
                  Suitability:{" "}
                  {score(
                    facility.suitability_score
                  )}
                  <br />
                  Verified OSM shelter:{" "}
                  {facility.verified_osm_shelter
                    ? "Yes"
                    : "No"}
                </Popup>
              </CircleMarker>
            ))}

            {cityRows.map((city) => (
              <CircleMarker
                key={`city-${city.city_name}-${city.latitude}`}
                center={[
                  city.latitude,
                  city.longitude,
                ]}
                radius={5}
                pathOptions={{
                  color: riskColor(city.risk_level),
                  fillColor: riskColor(
                    city.risk_level
                  ),
                  fillOpacity: 0.7,
                }}
              >
                <Popup>
                  <strong>{city.city_name}</strong>
                  <br />
                  Spatial risk:{" "}
                  {score(
                    city.city_spatial_risk_score
                  )}
                  <br />
                  Level: {city.risk_level}
                  <br />
                  River distance:{" "}
                  {number(
                    city.distance_to_river_m,
                    0
                  )} m
                </Popup>
              </CircleMarker>
            ))}

            {evacuation?.route_coordinates?.length > 1 && (
              <Polyline
                positions={evacuation.route_coordinates.map(
                  ([longitude, latitude]) => [
                    latitude,
                    longitude,
                  ]
                )}
                pathOptions={{
                  color: "#111827",
                  weight: 6,
                  opacity: 0.9,
                }}
              >
                <Popup>
                  Evacuation planning route
                  <br />
                  Distance:{" "}
                  {number(
                    evacuation.distance_km
                  )} km
                  <br />
                  Duration:{" "}
                  {number(
                    evacuation.duration_minutes,
                    1
                  )} min
                  <br />
                  Route risk:{" "}
                  {score(
                    evacuation.route_risk_score
                  )}
                </Popup>
              </Polyline>
            )}
          </MapContainer>
        </div>

        <div className="scenario-grid">
          <Metric
            label="Evacuation Route"
            value={
              evacuation?.available
                ? `${number(
                    evacuation.distance_km
                  )} km`
                : "Unavailable"
            }
          />

          <Metric
            label="Route Risk"
            value={
              evacuation?.route_risk_score != null
                ? `${score(
                    evacuation.route_risk_score
                  )} ${evacuation.route_risk_level || ""}`
                : "N/A"
            }
          />

          <Metric
            label="Highest-Risk City"
            value={
              cities?.highest_risk_city?.city_name ||
              "N/A"
            }
          />

          <Metric
            label="City Risk"
            value={
              cities?.highest_risk_city?.city_spatial_risk_score != null
                ? score(
                    cities.highest_risk_city
                      .city_spatial_risk_score
                  )
                : "N/A"
            }
          />
        </div>

        <div className="scenario-table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Infrastructure</th>
                <th>Type</th>
                <th>River Distance</th>
                <th>Risk</th>
                <th>Level</th>
              </tr>
            </thead>

            <tbody>
              {assets.slice(0, 20).map((item) => (
                <tr key={`asset-row-${item.osm_id}`}>
                  <td>{item.name}</td>
                  <td>{item.category}</td>
                  <td>
                    {number(
                      item.distance_to_river_m,
                      0
                    )}{" "}
                    m
                  </td>
                  <td>{score(item.risk_score)}</td>
                  <td>
                    <Badge level={item.risk_level} />
                  </td>
                </tr>
              ))}

              {assets.length === 0 && (
                <tr>
                  <td colSpan={5}>
                    No live infrastructure records
                    were returned.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="scenario-table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Facility</th>
                <th>Type</th>
                <th>Suitability</th>
                <th>Verified Shelter</th>
              </tr>
            </thead>

            <tbody>
              {facilities.slice(0, 15).map((item) => (
                <tr key={`facility-row-${item.osm_id}`}>
                  <td>{item.name}</td>
                  <td>{item.facility_type}</td>
                  <td>
                    {score(
                      item.suitability_score
                    )}
                  </td>
                  <td>
                    {item.verified_osm_shelter
                      ? "Yes"
                      : "Potential"}
                  </td>
                </tr>
              ))}

              {facilities.length === 0 && (
                <tr>
                  <td colSpan={4}>
                    No shelter/safe-facility
                    records were returned.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="scenario-table-wrapper">
          <table>
            <thead>
              <tr>
                <th>City / Town</th>
                <th>Place Type</th>
                <th>Spatial Risk</th>
                <th>Risk Level</th>
                <th>River Distance</th>
              </tr>
            </thead>

            <tbody>
              {cityRows.slice(0, 15).map((item, index) => (
                <tr
                  key={`city-row-${item.city_name}-${index}`}
                >
                  <td>{item.city_name}</td>
                  <td>{item.place_type}</td>
                  <td>
                    {score(
                      item.city_spatial_risk_score
                    )}
                  </td>
                  <td>
                    <Badge level={item.risk_level} />
                  </td>
                  <td>
                    {number(
                      item.distance_to_river_m,
                      0
                    )}{" "}
                    m
                  </td>
                </tr>
              ))}

              {cityRows.length === 0 && (
                <tr>
                  <td colSpan={5}>
                    No live city/town records were
                    returned.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {inundation && (
          <div className="scenario-note">
            <strong>Inundation screening:</strong>{" "}
            base index{" "}
            {score(
              inundation.base_inundation_screening_index
            )}, adjusted index{" "}
            {score(
              inundation.water_level_adjusted_index
            )}, level{" "}
            {inundation.risk_level || "N/A"}.
            <br />
            Components — spatial risk:{" "}
            {score(
              inundation.components?.spatial_risk
            )}
            , drainage:{" "}
            {score(
              inundation.components?.drainage_risk
            )}
            , elevation:{" "}
            {score(
              inundation.components?.elevation_risk
            )}
            .
          </div>
        )}

        {evacuation?.available && (
          <div className="scenario-note">
            <strong>Evacuation destination:</strong>{" "}
            {evacuation.destination?.name || "N/A"}.
            Route generation uses the district
            representative point as the origin and
            the highest-ranked mapped facility as the
            destination.
          </div>
        )}
      </div>
    </section>
  );
}






