import { useEffect, useState } from "react";
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

// Live 7-day flood forecast (current dates) from GloFAS river discharge + Open-Meteo rainfall.

const COLORS = { Low: "#16a34a", Moderate: "#ca8a04", High: "#ea580c", "Very High": "#dc2626" };
const card = { background: "#ffffff", border: "1px solid #dbe3ea", borderRadius: 14, padding: 20, marginBottom: 18 };
const badge = (lv) => ({
  display: "inline-block", padding: "3px 10px", borderRadius: 999, color: "#fff", fontWeight: 700, fontSize: 13,
  background: COLORS[lv] || "#64748b",
});
const fmtDate = (d) => {
  try {
    return new Date(d + "T00:00:00").toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
  } catch {
    return d;
  }
};
const n0 = (v) => (typeof v === "number" ? v.toLocaleString("en-IN", { maximumFractionDigits: 0 }) : "–");

export default function LiveFloodForecast({ selectedDistrictName }) {
  const district = selectedDistrictName || "Cuttack";
  const [fc, setFc] = useState(null);
  const [fcErr, setFcErr] = useState("");
  const [sum, setSum] = useState(null);
  const [sumErr, setSumErr] = useState("");

  useEffect(() => {
    let off = false;
    setFc(null);
    setFcErr("");
    fetch(`/api/live/forecast/${encodeURIComponent(district)}`)
      .then(async (r) => {
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
        return j;
      })
      .then((j) => !off && setFc(j))
      .catch((e) => !off && setFcErr(String(e.message || e)));
    return () => { off = true; };
  }, [district]);

  const [tick, setTick] = useState(0);

  useEffect(() => {
    let off = false;
    let timer = null;
    fetch("/api/live/summary")
      .then(async (r) => {
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
        return j;
      })
      .then((j) => {
        if (off) return;
        setSum(j);
        setSumErr("");
        if (j.preparing && j.preparing.length > 0) timer = setTimeout(() => setTick((t) => t + 1), 30000);
      })
      .catch((e) => {
        if (off) return;
        setSumErr(String(e.message || e));
        timer = setTimeout(() => setTick((t) => t + 1), 60000);
      });
    return () => { off = true; if (timer) clearTimeout(timer); };
  }, [tick]);

  const th = fc?.thresholds_m3s;
  const chart = (fc?.days || []).map((d) => ({ ...d, label: fmtDate(d.date) }));
  const todayLabel = chart.find((d) => d.is_today)?.label;
  const future = (fc?.days || []).filter((d) => !d.is_past);

  return (
    <div>
      <section style={card}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "flex-start" }}>
          <div>
            <h3 style={{ margin: 0 }}>Live 7-Day Flood Forecast — {district}</h3>
            <p style={{ margin: "4px 0 0", color: "#5b6b80" }}>
              Current dates. River discharge from GloFAS (Open-Meteo Flood API) and rainfall forecast, compared with
              the district river's own history.
            </p>
          </div>
          {fc && <div style={{ fontSize: 13, color: "#5b6b80" }}>Updated for {fmtDate(fc.today)} {fc.today?.slice(0, 4)}</div>}
        </div>

        {!fc && !fcErr && (
          <p style={{ color: "#5b6b80" }}>
            Loading live data… The first time a district is opened, its river reference (2010–2022) is downloaded
            once, which can take up to a minute.
          </p>
        )}
        {fcErr && (
          <p style={{ color: "#b91c1c" }}>
            Live forecast unavailable: {fcErr}. Check the internet connection and refresh in a minute.
          </p>
        )}

        {fc && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 12, marginTop: 14 }}>
              <div className="summary-card">
                <span>Today's flood risk</span>
                <strong><span style={badge(fc.today_level)}>{fc.today_level || "–"}</span></strong>
                <small>{fc.today_reason}</small>
              </div>
              <div className="summary-card">
                <span>Highest in next 7 days</span>
                <strong><span style={badge(fc.peak?.risk_level)}>{fc.peak?.risk_level || "–"}</span></strong>
                <small>{fc.peak ? `on ${fmtDate(fc.peak.date)} · ${fc.peak.reason}` : ""}</small>
              </div>
              <div className="summary-card">
                <span>Peak river discharge</span>
                <strong>{n0(Math.max(...future.map((d) => d.discharge_m3s || 0)))} m³/s</strong>
                <small>High-flow level (P95): {n0(th?.p95)} m³/s</small>
              </div>
              <div className="summary-card">
                <span>Heaviest rain day</span>
                <strong>{Math.max(...future.map((d) => d.rain_mm || 0)).toFixed(1)} mm</strong>
                <small>Heavy rain starts at 64.5 mm (IMD)</small>
              </div>
            </div>

            <div style={{ width: "100%", height: 320, marginTop: 18 }}>
              <ResponsiveContainer>
                <ComposedChart data={chart} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="label" />
                  <YAxis yAxisId="q" label={{ value: "Discharge (m³/s)", angle: -90, position: "insideLeft", offset: -2 }} />
                  <YAxis yAxisId="r" orientation="right" label={{ value: "Rain (mm)", angle: 90, position: "insideRight" }} />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="r" dataKey="rain_mm" name="Rain (mm)" fill="#93c5fd" />
                  <Line yAxisId="q" type="monotone" dataKey="discharge_m3s" name="River discharge" stroke="#1e5a85" strokeWidth={3} dot />
                  <Line yAxisId="q" type="monotone" dataKey="discharge_max_m3s" name="Upper estimate" stroke="#94a3b8" strokeDasharray="5 4" dot={false} />
                  {th && <ReferenceLine yAxisId="q" y={th.p80} stroke={COLORS.Moderate} strokeDasharray="4 4" label={{ value: "P80", fill: COLORS.Moderate, position: "right" }} />}
                  {th && <ReferenceLine yAxisId="q" y={th.p95} stroke={COLORS.High} strokeDasharray="4 4" label={{ value: "P95", fill: COLORS.High, position: "right" }} />}
                  {th && <ReferenceLine yAxisId="q" y={th.p99} stroke={COLORS["Very High"]} strokeDasharray="4 4" label={{ value: "P99", fill: COLORS["Very High"], position: "right" }} />}
                  {todayLabel && <ReferenceLine yAxisId="q" x={todayLabel} stroke="#0f2a3d" label={{ value: "Today", position: "top" }} />}
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div style={{ overflowX: "auto", marginTop: 14 }}>
              <table>
                <thead>
                  <tr>
                    <th>Date</th><th>River discharge</th><th>River level</th><th>Rain</th><th>Rain level</th><th>Flood risk</th><th>Why</th>
                  </tr>
                </thead>
                <tbody>
                  {future.map((d) => (
                    <tr key={d.date} style={d.is_today ? { background: "#eef6fb", fontWeight: 600 } : undefined}>
                      <td>{fmtDate(d.date)}{d.is_today ? " (today)" : ""}</td>
                      <td>{n0(d.discharge_m3s)} m³/s</td>
                      <td>{d.river_level || "–"}</td>
                      <td>{d.rain_mm == null ? "–" : `${d.rain_mm} mm`}</td>
                      <td>{d.rain_level || "–"}</td>
                      <td><span style={badge(d.risk_level)}>{d.risk_level || "–"}</span></td>
                      <td style={{ fontSize: 13 }}>{d.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p style={{ marginTop: 12, fontSize: 13, color: "#5b6b80", lineHeight: 1.5 }}>
              <b>Method:</b> river level compares the forecast discharge with this river's own daily history
              ({fc.reference_period}): below P80 Low, P80–P95 Moderate, P95–P99 High, above P99 Very High. Rain level
              uses IMD categories (heavy 64.5 mm, very heavy 115.6 mm, extremely heavy 204.5 mm). Daily risk is the
              higher of the two, raised one step for districts with spatial risk ≥ 0.70. River grid cell used:
              {" "}{fc.river_point?.lat}, {fc.river_point?.lon}. Source: {fc.source}. Screening forecast, not an official warning.
            </p>
          </>
        )}
      </section>

      <section style={card}>
        <h3 style={{ margin: 0 }}>Live Flood Outlook — All Districts (next 7 days)</h3>
        <p style={{ margin: "4px 0 12px", color: "#5b6b80" }}>Districts sorted by their highest forecast risk in the next 7 days.</p>
        {!sum && !sumErr && <p style={{ color: "#5b6b80" }}>Loading all districts… (first run can take 1–2 minutes)</p>}
        {sumErr && <p style={{ color: "#b91c1c" }}>Live outlook unavailable: {sumErr}</p>}
        {sum && sum.preparing?.length > 0 && (
          <p style={{ background: "#fff7e6", border: "1px solid #f5d38a", borderRadius: 8, padding: "8px 12px", fontSize: 14 }}>
            Preparing river references: <b>{sum.ready_count} of {sum.total}</b> districts ready. The list updates
            automatically every 30 seconds (each new district is set up once, about 15 seconds each).
            {sum.build_status?.error ? ` Note: ${sum.build_status.error}` : ""}
          </p>
        )}
        {sum && (
          <div style={{ overflowX: "auto", maxHeight: 460, overflowY: "auto" }}>
            <table>
              <thead>
                <tr><th>#</th><th>District</th><th>Today</th><th>Peak (7 days)</th><th>Peak day</th><th>Peak discharge</th><th>P95 level</th><th>Max rain</th></tr>
              </thead>
              <tbody>
                {sum.districts.map((d, i) => (
                  <tr key={d.district}>
                    <td>{i + 1}</td>
                    <td style={{ fontWeight: 600 }}>{d.district}</td>
                    <td><span style={badge(d.today_level)}>{d.today_level || "–"}</span></td>
                    <td><span style={badge(d.peak_level)}>{d.peak_level || "–"}</span></td>
                    <td>{d.peak_date ? fmtDate(d.peak_date) : "–"}</td>
                    <td>{n0(d.peak_discharge_m3s)} m³/s</td>
                    <td>{n0(d.p95_m3s)} m³/s</td>
                    <td>{typeof d.max_rain_mm === "number" ? `${d.max_rain_mm.toFixed(1)} mm` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {sum.unavailable?.length > 0 && (
              <p style={{ fontSize: 13, color: "#b45309" }}>No live data for: {sum.unavailable.map((u) => u.split(":")[0]).join(", ")}</p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
