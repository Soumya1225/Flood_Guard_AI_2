// Project Overview page for Flood Guard AI (like the Marine Trace AI welcome page).
// Explains what the project does, how it works, which models are used and the tested results.

const pct = (v, fallback) => (typeof v === "number" ? `${v.toFixed(1)}%` : fallback);
const num = (v, d = 3, fallback = "–") => (typeof v === "number" ? v.toFixed(d) : fallback);

const STEPS = [
  ["Collect", "IMD district rainfall, river water-level and weather records, live Open-Meteo weather, Odisha GIS layers (district boundaries, river network, historical floods), WorldPop and Census population."],
  ["Map the risk", "A GIS spatial-risk index scores all 30 Odisha districts from 0 to 1 using elevation, drainage, distance to river and past flood events, and colours them on the risk map."],
  ["Predict", "A Logistic Regression model (v2, trained on real flood events 2000–2016, tested on 2017–2023) runs every 3 hours on live rainfall and weather and gives each district's flood probability for the next 7 days. A Random Forest Classifier on 2020 water-level data is kept as a reference model with feature importance."],
  ["Forecast", "Live GloFAS river-discharge forecasts and rain forecasts give each district a daily flood risk for the next 7 days. A Random Forest Regressor shows the 7-day water-level forecast on historical gauge data (reference)."],
  ["Spread & impact", "GIS screening finds neighbouring, lower-lying districts that may be affected next, ranks hotspots, estimates the population at risk (WorldPop × risk) and the towns to prioritise."],
  ["Act", "The response plan sets the response level (Normal / Watch / Warning / Severe), estimates people to shelter and relief needs, lists priority towns, contacts and ready-to-send warnings in English, Hindi and Odia, and tracks a checklist. Emergency GIS screens shelters, roads and evacuation."],
];

const BUILT = ["Python", "scikit-learn", "Logistic Regression", "Random Forest", "GloFAS (Copernicus)", "FastAPI", "SQLite", "React", "Leaflet", "Recharts",
  "GeoPackage GIS", "IMD rainfall", "Open-Meteo API", "WorldPop", "Census 2011", "OpenStreetMap"];

const METHODS = [
  ["Will a flood start in the next 7 days? (live)", "Logistic Regression v2 (live) + Random Forest (2020 reference)", "Machine learning",
   "The live model reads the last 60 days of rain and the 8-day forecast and gives a flood probability (ROC-AUC 0.782 on unseen years). The Random Forest votes 'flood' / 'no flood' on 2020 water-level data."],
  ["Why did it predict that?", "Feature importance", "Explainable AI",
   "Shows which inputs mattered most, e.g. maximum water level counts for about 36% of the decision."],
  ["Will the rivers rise this week? (live)", "GloFAS river forecast + rain thresholds", "Live data + rules",
   "Compares each district's forecast river flow with its own history (P80 / P95 / P99) and heavy-rain levels (IMD) to give Low → Very High risk for each day."],
  ["How high will the river be in the next 7 days?", "Random Forest Regressor", "Machine learning",
   "Predicts tomorrow's water level, then uses that answer to predict the next day, up to 7 days (historical reference)."],
  ["Is the forecast water level dangerous?", "Percentile rule", "Statistics",
   "Compares the predicted level with past levels: higher than most past days means higher risk."],
  ["Which districts are most at risk?", "Spatial risk index", "GIS",
   "Gives every district a score from 0 to 1 using low elevation, poor drainage, nearness to rivers and past floods."],
  ["Which nearby districts may flood next?", "Propagation & hotspot screening", "GIS",
   "Checks neighbouring districts and their height (water flows downhill). Hotspot score = 50% own risk + 50% chance of receiving water."],
  ["How many people are at risk?", "Population exposure", "Calculation",
   "District population (WorldPop 2020) × risk score. Towns are ranked by population × risk."],
  ["Is today's rain unusual?", "Rainfall anomaly (z-score)", "Statistics",
   "Compares today's rain with the district's normal rain; a big difference means unusual rain."],
  ["What if conditions get worse?", "What-if scenario engine", "Simulation",
   "Increase rain or water level, or reduce drainage, and the risk of every district is recalculated."],
  ["What should we do now?", "Alert engine & action rules", "Decision rules",
   "Counts high-risk districts to set the alert level and lists the recommended emergency actions."],
  ["How many people need help, and what do we tell them?", "Response plan engine", "Decision rules + standards",
   "Takes the higher of the live river outlook and the live AI prediction, then estimates shelter, water, food, toilets and boats (Sphere standards) and writes warnings in English, Hindi and Odia."],
  ["What is the weather now and next week?", "Open-Meteo API", "Live data",
   "Fetches live and 7-day forecast weather for the selected district (data source, not a model)."],
];

const S = {
  hero: { background: "linear-gradient(135deg, #0f2a3d 0%, #173f5f 55%, #1e5a85 100%)", color: "#fff", borderRadius: 16, padding: "40px 36px" },
  tag: { fontSize: 13, letterSpacing: "0.12em", textTransform: "uppercase", color: "#9fc4dd" },
  h1: { fontSize: 44, lineHeight: 1.1, margin: "12px 0 14px", fontWeight: 800 },
  lead: { fontSize: 17, lineHeight: 1.6, maxWidth: 760, color: "#dbe9f2", margin: 0 },
  btn: { background: "#f0a91c", color: "#0f2a3d", border: 0, borderRadius: 8, padding: "12px 20px", fontWeight: 700, fontSize: 15, cursor: "pointer" },
  btn2: { background: "transparent", color: "#fff", border: "1px solid #9fc4dd", borderRadius: 8, padding: "12px 20px", fontWeight: 600, fontSize: 15, cursor: "pointer" },
  card: { background: "#fff", border: "1px solid #dbe3ea", borderRadius: 14, padding: 22, marginTop: 20 },
  h2: { margin: "0 0 14px", fontSize: 22 },
  grid: (min) => ({ display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))`, gap: 14 }),
  stat: { background: "#f3f7fa", borderRadius: 12, padding: 16 },
  big: { fontSize: 30, fontWeight: 800, color: "#0f2a3d" },
  small: { fontSize: 13, color: "#5b6b80" },
  chip: { background: "#e8f1f7", color: "#1e4a66", borderRadius: 999, padding: "6px 12px", fontSize: 13, fontWeight: 600 },
};

export default function ProjectOverview({ riskData, highRiskCount, decisionSupport, modelPerformance, onOpen }) {
  const ev = modelPerformance?.evaluation || {};
  const live = [
    [riskData?.district_count ?? "–", "Districts monitored"],
    [decisionSupport?.alert_level || "–", "Current alert level"],
    [num(riskData?.average_risk_score), "Average risk score"],
    [highRiskCount ?? "–", "High / Very High districts"],
    [Array.isArray(decisionSupport?.priority_districts) ? decisionSupport.priority_districts.length : "–", "Priority districts"],
  ];
  const results = [
    [pct(ev.accuracy_percent, "98.4%"), "flood prediction accuracy", `Random Forest, ${ev.test_samples ?? 63} test samples`],
    [pct(ev.precision_percent, "100%"), "precision", "no false flood alarms on the test set"],
    [pct(ev.recall_percent, "66.7%"), "recall", "real flood events caught (only 3 in the test set)"],
    ["7 days", "water-level forecast", "recursive Random Forest Regressor with error range"],
  ];

  return (
    <div>
      <section style={S.hero}>
        <div style={S.tag}>Machine Learning for Predictive Analytics · Odisha</div>
        <h1 style={S.h1}>Predict the flood.<br />Protect the district.</h1>
        <p style={S.lead}>
          Flood Guard AI predicts flood risk with machine learning, maps the risk of all 30 Odisha districts,
          shows which neighbouring areas and towns may be affected next, and lets officials test what-if
          situations before deciding where to send help.
        </p>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 24 }}>
          <button type="button" style={S.btn} onClick={() => onOpen("dashboard")}>Open the dashboard</button>
          <button type="button" style={S.btn2} onClick={() => document.getElementById("fg-how")?.scrollIntoView({ behavior: "smooth" })}>How it works</button>
        </div>
      </section>

      <section style={S.card}>
        <h2 style={S.h2}>Live system</h2>
        <div style={S.grid(160)}>
          {live.map(([v, l]) => (
            <div key={l} style={S.stat}><div style={S.big}>{v}</div><div style={S.small}>{l}</div></div>
          ))}
        </div>
      </section>

      <section style={S.card} id="fg-how">
        <h2 style={S.h2}>How it works</h2>
        <div style={S.grid(280)}>
          {STEPS.map(([t, d], i) => (
            <div key={t} style={{ ...S.stat, display: "flex", gap: 14 }}>
              <div style={{ flex: "none", width: 36, height: 36, borderRadius: "50%", background: "#1e5a85", color: "#fff", display: "grid", placeItems: "center", fontWeight: 800 }}>{i + 1}</div>
              <div><div style={{ fontWeight: 700, fontSize: 16 }}>{t}</div><div style={{ fontSize: 14, color: "#334155", lineHeight: 1.5, marginTop: 4 }}>{d}</div></div>
            </div>
          ))}
        </div>
      </section>

      <section style={S.card}>
        <h2 style={S.h2}>Tested results</h2>
        <div style={S.grid(200)}>
          {results.map(([v, l, n]) => (
            <div key={l} style={S.stat}>
              <div style={S.big}>{v}</div>
              <div style={{ fontWeight: 600 }}>{l}</div>
              <div style={S.small}>{n}</div>
            </div>
          ))}
        </div>
      </section>

      <section style={S.card}>
        <h2 style={S.h2}>What-if example: combined stress</h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#334155" }}>
          With rainfall +25%, water level +0.5 m and drainage effectiveness −15% together, the average district risk
          rises from <b>0.573 to 0.753</b> and the number of <b>Very High</b> districts jumps from <b>2 to 16</b>.
          Cuttack, Malkangiri and Jajpur remain the top priorities, and the system recommends monitoring the
          high-risk districts, checking drainage and preparing response resources in advance.
        </p>
      </section>

      <section style={S.card}>
        <h2 style={S.h2}>Which method is used for what</h2>
        <p style={{ margin: "-6px 0 14px", color: "#5b6b80" }}>Every question the system answers, the method behind it, in simple words.</p>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ textAlign: "left", background: "#0f2a3d", color: "#fff" }}>
                {["#", "Question", "Method used", "Type", "How it works (simple)"].map((h) => (
                  <th key={h} style={{ padding: "10px 12px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METHODS.map(([q, m, t, d], i) => (
                <tr key={q} style={{ background: i % 2 ? "#ffffff" : "#f6f9fb", verticalAlign: "top" }}>
                  <td style={{ padding: "10px 12px", color: "#5b6b80" }}>{i + 1}</td>
                  <td style={{ padding: "10px 12px", fontWeight: 600 }}>{q}</td>
                  <td style={{ padding: "10px 12px", fontWeight: 700, color: "#1e5a85" }}>{m}</td>
                  <td style={{ padding: "10px 12px" }}><span style={{ ...S.chip, display: "inline-block", whiteSpace: "nowrap" }}>{t}</span></td>
                  <td style={{ padding: "10px 12px", color: "#334155", lineHeight: 1.5 }}>{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button type="button" style={{ ...S.btn, marginTop: 16 }} onClick={() => onOpen("models")}>See model details &amp; performance</button>
      </section>

      <section style={S.card}>
        <h2 style={S.h2}>Built with</h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {BUILT.map((b) => <span key={b} style={S.chip}>{b}</span>)}
        </div>
      </section>

      <p style={{ textAlign: "center", color: "#64748b", fontSize: 13, margin: "22px 0 4px" }}>
        MCA project · Centurion University of Technology and Management · Risk scores are decision-support
        indicators, not official flood warnings.
      </p>
    </div>
  );
}
