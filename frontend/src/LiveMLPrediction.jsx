import { useEffect, useState } from "react";
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

// Live ML flood prediction: the trained v2 model run on today's rainfall and weather.

const LEVEL_COLORS = { Normal: "#16a34a", Watch: "#ca8a04", Warning: "#ea580c", Severe: "#dc2626" };
const card = { background: "#ffffff", border: "1px solid #dbe3ea", borderRadius: 14, padding: 20, marginBottom: 18 };
const badge = (lv) => ({
  display: "inline-block", padding: "3px 10px", borderRadius: 999, color: "#fff", fontWeight: 700, fontSize: 13,
  background: LEVEL_COLORS[lv] || "#64748b",
});
const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "–");
const fmtDate = (d) => {
  try {
    return new Date(d + "T00:00:00").toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
  } catch {
    return d;
  }
};
const FEATURE_NAMES = {
  rain_mm: "Rain today", rain_3d: "Rain, last 3 days", rain_7d: "Rain, last 7 days", rain_15d: "Rain, last 15 days",
  rain_30d: "Rain, last 30 days", rain_max_3d: "Heaviest day (3 days)", rain_api: "Soil wetness index",
  rain_7d_anomaly: "Rain above normal (7 days)", rain_7d_normal: "Normal 7-day rain",
  district_flood_freq: "Past flood frequency", humidity: "Humidity", humidity_3d: "Humidity (3 days)",
  temp_c: "Temperature", wind_ms: "Wind", doy_sin: "Season", doy_cos: "Season", monsoon: "Monsoon season",
  static_elevation_m: "Elevation", static_distance_to_river_m: "Distance to river",
};
const reasonText = (drivers) =>
  (drivers || []).map((d) => FEATURE_NAMES[d.feature] || d.feature).filter((v, i, a) => a.indexOf(v) === i).join(", ");

async function getJson(url) {
  const r = await fetch(url);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
  return j;
}

export default function LiveMLPrediction({ selectedDistrictName }) {
  const district = selectedDistrictName || "Cuttack";
  const [all, setAll] = useState(null);
  const [allErr, setAllErr] = useState("");
  const [one, setOne] = useState(null);
  const [oneErr, setOneErr] = useState("");

  useEffect(() => {
    let off = false;
    getJson("/api/live-ml/warnings")
      .then((j) => !off && setAll(j))
      .catch((e) => !off && setAllErr(String(e.message || e)));
    return () => { off = true; };
  }, []);

  useEffect(() => {
    let off = false;
    setOne(null);
    setOneErr("");
    getJson(`/api/live-ml/district/${encodeURIComponent(district)}`)
      .then((j) => !off && setOne(j))
      .catch((e) => !off && setOneErr(String(e.message || e)));
    return () => { off = true; };
  }, [district]);

  const chart = (one?.series || []).map((d) => ({ ...d, label: fmtDate(d.date), probability_pct: d.probability * 100 }));
  const todayLabel = chart.find((d) => d.is_today)?.label;
  const thr = (all?.threshold ?? one?.threshold ?? 0) * 100;

  return (
    <div>
      <section style={card}>
        <h3 style={{ margin: 0 }}>Live AI Flood Prediction — {district}</h3>
        <p style={{ margin: "4px 0 12px", color: "#5b6b80" }}>
          The trained machine-learning model (v2, {one?.model || all?.model || "Logistic Regression"}) run on today's
          live rainfall and weather. Question: <i>will a flood start in this district within the next 7 days?</i>
        </p>

        {!one && !oneErr && <p style={{ color: "#5b6b80" }}>Running the model on live data…</p>}
        {oneErr && <p style={{ color: "#b91c1c" }}>Live prediction unavailable: {oneErr}</p>}

        {one && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 12 }}>
              <div className="summary-card">
                <span>Today ({fmtDate(one.today)})</span>
                <strong><span style={badge(one.today_level)}>{one.today_level}</span></strong>
                <small>Flood probability {pct(one.today_probability)}</small>
              </div>
              <div className="summary-card">
                <span>Highest in next 7 days</span>
                <strong><span style={badge(one.peak_level)}>{one.peak_level}</span></strong>
                <small>{pct(one.peak_probability)} on {fmtDate(one.peak_date)}</small>
              </div>
              <div className="summary-card">
                <span>Rain: last 7 days / next 7 days</span>
                <strong>{one.rain_last7_mm} / {one.rain_next7_mm} mm</strong>
                <small>Live Open-Meteo rainfall</small>
              </div>
              <div className="summary-card">
                <span>Main reasons today</span>
                <strong style={{ fontSize: 15 }}>{reasonText(one.drivers) || "No strong flood signal"}</strong>
                <small>Features pushing the probability up</small>
              </div>
            </div>

            <div style={{ width: "100%", height: 300, marginTop: 16 }}>
              <ResponsiveContainer>
                <ComposedChart data={chart} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="label" minTickGap={14} />
                  <YAxis yAxisId="p" domain={[0, 100]} label={{ value: "Flood probability (%)", angle: -90, position: "insideLeft" }} />
                  <YAxis yAxisId="r" orientation="right" label={{ value: "Rain (mm)", angle: 90, position: "insideRight" }} />
                  <Tooltip formatter={(v, n) => (n === "Flood probability" ? `${Number(v).toFixed(1)}%` : v)} />
                  <Legend />
                  <Bar yAxisId="r" dataKey="rain_mm" name="Rain (mm)" fill="#93c5fd" />
                  <Line yAxisId="p" type="monotone" dataKey="probability_pct" name="Flood probability" stroke="#b91c1c" strokeWidth={3} dot={false} />
                  {thr > 0 && <ReferenceLine yAxisId="p" y={thr} stroke="#ea580c" strokeDasharray="4 4" label={{ value: "Warning level", fill: "#ea580c", position: "right" }} />}
                  {todayLabel && <ReferenceLine yAxisId="p" x={todayLabel} stroke="#0f2a3d" label={{ value: "Today", position: "top" }} />}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <p style={{ fontSize: 13, color: "#5b6b80", marginTop: 8, lineHeight: 1.5 }}>
              <b>How:</b> the last 60 days of live rainfall and weather plus the 8-day forecast are turned into the same
              features the model was trained on (rain over 3/7/15/30 days, heaviest day, soil-wetness index, rain vs normal
              for this time of year, season, humidity, temperature, wind, past flood frequency). {one.note}
            </p>
          </>
        )}
      </section>

      <section style={card}>
        <h3 style={{ margin: 0 }}>Live AI Flood Warnings — All Districts</h3>
        <p style={{ margin: "4px 0 12px", color: "#5b6b80" }}>Sorted by the highest flood probability in the next 7 days.</p>
        {!all && !allErr && <p style={{ color: "#5b6b80" }}>Running the model for all districts…</p>}
        {allErr && <p style={{ color: "#b91c1c" }}>Live warnings unavailable: {allErr}</p>}
        {all && (
          <div style={{ overflowX: "auto", maxHeight: 460, overflowY: "auto" }}>
            <table>
              <thead>
                <tr><th>#</th><th>District</th><th>Today</th><th>Probability today</th><th>Peak (7 days)</th><th>Peak probability</th><th>Peak day</th><th>Rain next 7 days</th><th>Main reasons</th></tr>
              </thead>
              <tbody>
                {all.districts.map((d, i) => (
                  <tr key={d.district}>
                    <td>{i + 1}</td>
                    <td style={{ fontWeight: 600 }}>{d.district}</td>
                    <td><span style={badge(d.today_level)}>{d.today_level}</span></td>
                    <td>{pct(d.today_probability)}</td>
                    <td><span style={badge(d.peak_level)}>{d.peak_level}</span></td>
                    <td>{pct(d.peak_probability)}</td>
                    <td>{fmtDate(d.peak_date)}</td>
                    <td>{d.rain_next7_mm} mm</td>
                    <td style={{ fontSize: 13 }}>{reasonText(d.drivers) || "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
