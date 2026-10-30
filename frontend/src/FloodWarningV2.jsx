import { useEffect, useState } from "react";
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API = "/api/v2";
const LEVEL = { Severe: "#b91c1c", Warning: "#ea580c", Watch: "#ca8a04", Normal: "#15803d" };
const card = { background: "#ffffff", border: "1px solid #dbe3ea", borderRadius: 12, padding: 16, marginTop: 16 };
const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);

export default function FloodWarningV2() {
  const [date, setDate] = useState("");
  const [w, setW] = useState(null);
  const [sel, setSel] = useState(null);
  const [hist, setHist] = useState(null);
  const [report, setReport] = useState(null);
  const [wi, setWi] = useState({ rain_change_pct: 50, extra_rain_today_mm: 0 });
  const [wiRes, setWiRes] = useState(null);
  const [err, setErr] = useState("");
  const [events, setEvents] = useState([]);

  const load = (d) => {
    setErr("");
    fetch(`${API}/warnings${d ? `?date=${d}` : ""}`)
      .then((r) => (r.ok ? r.json() : r.json().then((j) => Promise.reject(j.detail || r.statusText))))
      .then((j) => { setW(j); setDate(j.date); if (j.districts.length) setSel(j.districts[0].district); })
      .catch((e) => setErr(String(e)));
  };
  useEffect(() => {
    // open on the most recent real flood event (falls back to the latest data day)
    fetch(`${API}/flood-dates`)
      .then((r) => (r.ok ? r.json() : { dates: [] }))
      .then((j) => {
        const list = j.dates || [];
        setEvents(list);
        load(list.length ? list[0].date : "");
      })
      .catch(() => load(""));
    fetch(`${API}/model-report`).then((r) => (r.ok ? r.json() : null)).then(setReport).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!sel || !w) return;
    fetch(`${API}/district/${encodeURIComponent(sel)}?date=${w.date}&days=90`).then((r) => r.json()).then(setHist).catch(() => {});
    setWiRes(null);
  }, [sel, w]);

  const runWhatIf = () =>
    fetch(`${API}/what-if`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ district: sel, date: w.date, ...wi }) })
      .then((r) => r.json()).then(setWiRes).catch((e) => setErr(String(e)));

  const best = report?.metrics_v2;
  return (
    <section style={card}>
      <h2 style={{ marginTop: 0 }}>Past-Event Replay — Flood Early Warning v2 (trained on real flood events)</h2>
      <p style={{ color: "#475569", marginTop: 0 }}>
        {w?.question || "Will a flood start in this district soon?"} Model: <b>{w?.model}</b>, trained on India Flood Inventory events
        2000–2016 and tested on 2017–2023 using 24 years of IMD rainfall. Pick any date to replay a past situation.
      </p>
      {events.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "6px 0 10px", alignItems: "center" }}>
          <span style={{ fontSize: 13, color: "#475569" }}>Past flood events:</span>
          {events.map((ev) => (
            <button key={ev.date} type="button" title={`${ev.districts} districts: ${ev.names}`}
              onClick={() => { setDate(ev.date); load(ev.date); }}
              style={{ padding: "4px 10px", borderRadius: 999, cursor: "pointer", fontSize: 13,
                       border: "1px solid #cbd5e1", background: date === ev.date ? "#0f2a3d" : "#fff",
                       color: date === ev.date ? "#fff" : "#0f2a3d" }}>
              {ev.date} · {ev.districts} districts
            </button>
          ))}
        </div>
      )}
      <label>Date: <input type="date" value={date} onChange={(e) => { setDate(e.target.value); load(e.target.value); }} /></label>
      {err && <p style={{ color: "#b91c1c" }}>{err}</p>}
      {w?.levels && (
        <p style={{ fontSize: 13, color: "#5b6b80", margin: "6px 0 0" }}>
          Levels use the model's tuned alert thresholds: Watch from {(w.levels.Watch * 100).toFixed(1)}%, Warning from{" "}
          {(w.levels.Warning * 100).toFixed(1)}%, Severe from {(w.levels.Severe * 100).toFixed(1)}%. In the monsoon every district
          has a high probability, so the level shows which districts stand out, not the probability alone.
        </p>
      )}

      {w && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1fr) minmax(320px, 1.3fr)", gap: 16, marginTop: 12 }}>
          <div style={{ maxHeight: 520, overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead><tr><th align="left">District</th><th>Level</th><th>Probability</th><th>Rain 3 days</th></tr></thead>
              <tbody>
                {w.districts.map((d) => (
                  <tr key={d.district} onClick={() => setSel(d.district)}
                    style={{ cursor: "pointer", background: sel === d.district ? "#eef4fb" : "transparent", borderTop: "1px solid #eef2f6" }}>
                    <td>{d.district}</td>
                    <td align="center"><span style={{ background: LEVEL[d.level], color: "#fff", borderRadius: 6, padding: "1px 8px" }}>{d.level}</span></td>
                    <td align="center">{pct(d.probability)}</td>
                    <td align="center">{d.rain_3d_mm} mm</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            {sel && <h3 style={{ marginTop: 0 }}>{sel}: last 90 days</h3>}
            {sel && (w.districts.find((d) => d.district === sel)?.drivers || []).length > 0 && (
              <p style={{ fontSize: 13 }}>Main reasons: {[...new Set(w.districts.find((d) => d.district === sel).drivers.map((x) => x.feature))].join(", ")}</p>
            )}
            {hist && (
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={hist.series.map((s) => ({ ...s, flood: s.flood_reported ? 1 : null, probability: s.probability * 100 }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={10} minTickGap={30} />
                  <YAxis yAxisId="r" fontSize={10} unit=" mm" />
                  <YAxis yAxisId="p" orientation="right" fontSize={10} unit="%" />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar yAxisId="r" dataKey="rain_mm" name="Rain (mm)" fill="#60a5fa" />
                  <Line yAxisId="p" dataKey="probability" name="Flood probability (%)" stroke="#ea580c" dot={false} strokeWidth={2} />
                  <Bar yAxisId="p" dataKey="flood" name="Flood reported" fill="#b91c1c" barSize={4} />
                </ComposedChart>
              </ResponsiveContainer>
            )}
            <div style={{ borderTop: "1px solid #eef2f6", marginTop: 8, paddingTop: 8 }}>
              <b>What-if (the ML model is re-run)</b>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 6, fontSize: 13 }}>
                <label>Rain change %: <input type="number" value={wi.rain_change_pct} style={{ width: 70 }}
                  onChange={(e) => setWi({ ...wi, rain_change_pct: Number(e.target.value) })} /></label>
                <label>Extra rain today (mm): <input type="number" value={wi.extra_rain_today_mm} style={{ width: 70 }}
                  onChange={(e) => setWi({ ...wi, extra_rain_today_mm: Number(e.target.value) })} /></label>
                <button onClick={runWhatIf}>Run</button>
              </div>
              {wiRes && (
                <p style={{ fontSize: 13 }}>
                  Probability {pct(wiRes.baseline_probability)} ({wiRes.baseline_level}) → <b>{pct(wiRes.scenario_probability)} ({wiRes.scenario_level})</b>
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {best && (
        <div style={{ marginTop: 16 }}>
          <h3>How good is it? (unseen years {best.test_years})</h3>
          <table style={{ borderCollapse: "collapse", fontSize: 13 }}>
            <thead><tr><th align="left">Model</th><th>ROC-AUC</th><th>Floods warned</th><th>Alerts per district-year</th></tr></thead>
            <tbody>
              {best.results.map((r) => (
                <tr key={r.model} style={{ borderTop: "1px solid #eef2f6", fontWeight: r.model === best.selected_model ? 700 : 400 }}>
                  <td>{r.model}</td><td align="center">{r.roc_auc}</td>
                  <td align="center">{r.event_hit_rate == null ? "–" : pct(r.event_hit_rate)}</td>
                  <td align="center">{r.alerts_per_district_year}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
