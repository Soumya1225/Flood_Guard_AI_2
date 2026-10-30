// "Models & Methods Used" section for Flood Guard AI.
// Explains which model / method is used for each part of the dashboard and what it does.
// Static content (no backend call), so it always loads.

const card = { background: "#ffffff", border: "1px solid #dbe3ea", borderRadius: 12, padding: 18, marginTop: 18 };
const cell = { padding: "8px 10px", borderBottom: "1px solid #eef2f6", verticalAlign: "top" };

const MODELS = [
  {
    name: "Logistic Regression (v2) — Live AI Prediction",
    kind: "Machine Learning",
    color: "#b91c1c",
    job: "Will a flood start in the next 7 days? (today, live)",
    work: "Chosen as the best of 4 candidates (Logistic Regression, Random Forest, XGBoost, rainfall rule). It is run every 3 hours on live Open-Meteo rainfall and weather for all 30 districts.",
    input: "Rain over 1/3/7/15/30 days, heaviest day, soil-wetness index, rain vs normal for the season, season, humidity, temperature, wind, past flood frequency",
    output: "Flood-start probability → Normal / Watch / Warning / Severe, with the main reasons",
    details: "Trained on India Flood Inventory events 2000–2016, tested on unseen years 2017–2023: ROC-AUC 0.782 vs 0.668 for the rainfall-only rule; catches 36.4% of flood events at the same alert budget (Random Forest 12.4%, XGBoost 13.2%).",
    where: "AI Prediction (Live AI Flood Prediction, Past-Event Replay), Response Plan",
  },
  {
    name: "Live River Outlook (GloFAS)",
    kind: "Live hydrology data + rules",
    color: "#1e5a85",
    job: "Will the rivers rise above normal this week?",
    work: "Takes the GloFAS river-discharge forecast (Copernicus, via Open-Meteo Flood API) for each district's main river and compares it with that river's own history.",
    input: "Forecast river discharge (m³/s), 7-day rainfall forecast, district spatial risk",
    output: "Daily flood risk Low / Moderate / High / Very High for today and the next 7 days",
    details: "Below P80 Low, P80–P95 Moderate, P95–P99 High, above P99 Very High; heavy-rain levels from IMD; small streams (flow below 50 m³/s, or a usual high flow P95 below 100 m³/s) capped at Moderate; raised one step for districts with spatial risk ≥ 0.70.",
    where: "Flood Forecast (live), Area alert, Impact, Response Plan",
  },
  {
    name: "Response Plan Engine",
    kind: "Decision rules + standards",
    color: "#7c2d12",
    job: "What to do, for how many people, and what to tell them",
    work: "Takes the higher of the live river outlook and the live AI prediction to set the response level, then estimates people to shelter and relief needs and lists priority towns, contacts, warning messages and a checklist.",
    input: "Live outlook, live ML probability, WorldPop population, spatial risk, Census towns",
    output: "Response level (Normal / Watch / Warning / Severe), relief needs, warnings in English / Hindi / Odia, checklist",
    details: "Water 15 L/person/day, 3.5 m² shelter space and 1 toilet per 20 people follow Sphere standards; other numbers are stated planning assumptions.",
    where: "Emergency Response (Flood Response Plan)",
  },
  {
    name: "Random Forest Classifier",
    kind: "Machine Learning",
    color: "#15803d",
    job: "Flood event prediction",
    work: "Predicts whether a high-water (flood) event will happen and gives its probability. Many decision trees vote; the share of trees voting 'flood' is the probability.",
    input: "12 features: water level mean / max / min / change, 3-day and 7-day water-level mean, temperature, humidity, wind speed, pressure, month, day of year",
    output: "Flood probability (%) → Low / Moderate / High / Very High risk",
    details: "300 trees, max depth 8, class weight = balanced, stratified 80/20 split. Accuracy 98.41%, Precision 100%, Recall 66.67%, F1 80% (63 test samples).",
    where: "AI Flood Prediction, Model Performance Dashboard",
  },
  {
    name: "Feature Importance (Explainable AI)",
    kind: "Explainability",
    color: "#0f766e",
    job: "Why the model predicted this",
    work: "Shows which inputs the Random Forest relies on most (e.g. maximum water level ≈ 36%, mean water level ≈ 19%) and compares current values with the training median.",
    input: "The trained Random Forest Classifier",
    output: "Ranked list of features with importance %",
    details: "Importance shows model contribution, not cause and effect.",
    where: "AI Explainability, Feature Importance chart",
  },
  {
    name: "Random Forest Regressor",
    kind: "Machine Learning",
    color: "#2563eb",
    job: "Water-level forecasting",
    work: "Predicts the next day's river water level. The prediction is fed back in to forecast day 2, day 3 … up to day 7 (recursive forecasting).",
    input: "Past water levels (lags, 3-day / 7-day means), weather, date features",
    output: "Water level (m) for the next 1–7 days, trend (rising / falling / stable) and an error range",
    details: "Forecast range = model error band, not a calibrated interval. Risk for 1 / 3 / 7 days uses a water-level percentile rule.",
    where: "Flood Risk Forecast, Historical & 7-Day Water-Level Forecast",
  },
  {
    name: "Spatial Risk Index (GIS)",
    kind: "GIS multi-criteria analysis",
    color: "#b45309",
    job: "District flood-risk score",
    work: "Combines several risk factors for every Odisha district into one score between 0 and 1.",
    input: "Elevation risk, drainage risk, distance to river, historical flood events (from the boundary, river network and historical flood layers)",
    output: "Risk score 0–1 → Moderate / High / Very High for 30 districts",
    details: "Lower elevation, poor drainage, nearness to rivers and more past floods give a higher score.",
    where: "Risk map, District Explorer, District Comparison, Risk Distribution",
  },
  {
    name: "Flood Propagation & Hotspot Screening",
    kind: "GIS spatial analysis",
    color: "#c2410c",
    job: "Which nearby districts may be affected",
    work: "Looks at districts next to the high-risk one and estimates potential impact from their own risk and relative elevation (water flows to lower areas). Hotspot score combines current risk and potential impact.",
    input: "District adjacency, spatial risk score, elevation",
    output: "Potentially affected districts, potential impact score, priority hotspot",
    details: "Screening only; not hydraulic flood routing.",
    where: "Potentially Affected Districts, Hotspot Screening, AI Situation Summary",
  },
  {
    name: "City Impact Priority (Exposure)",
    kind: "Risk-weighted exposure",
    color: "#7c3aed",
    job: "Which towns to prioritise",
    work: "Ranks the towns of the selected district by how many people are exposed to its flood risk.",
    input: "Census 2011 town population, district risk score",
    output: "Risk-weighted exposure = population × risk score; priority score out of 100",
    details: "Shows exposure priority, not actual inundation.",
    where: "City Impact Priority Screening",
  },
  {
    name: "What-If Scenario Analysis",
    kind: "Scenario simulation",
    color: "#db2777",
    job: "What happens if conditions get worse",
    work: "Recalculates every district's risk after changing rainfall, water level or drainage, so emergency teams can plan for worse conditions.",
    input: "Rainfall change (%), water-level change (m), drainage effectiveness change (%)",
    output: "New district risk scores and levels; comparison with the base case",
    details: "Predefined: Rainfall +25%, Water level +0.5 m, Drainage −15%, Combined stress; plus custom Dynamic What-If.",
    where: "Dynamic What-If, Scenario Comparison, Scenario Explorer",
  },
  {
    name: "Rainfall Anomaly Detection",
    kind: "Statistical method",
    color: "#0891b2",
    job: "Is today's rain unusual?",
    work: "Compares the latest district rainfall with its historical median and spread.",
    input: "IMD district daily rainfall",
    output: "Anomaly (mm and %), z-score, anomaly level",
    details: "z-score = (latest − mean) / standard deviation.",
    where: "Rainfall Anomaly Detection",
  },
  {
    name: "Rule-based Emergency Actions",
    kind: "Decision rules",
    color: "#475569",
    job: "What to do now",
    work: "Turns the current risk level into recommended actions (monitor high-risk districts, check drainage, prepare resources).",
    input: "Overall and district alert levels",
    output: "Prioritised action list",
    details: "Academic decision support, not official instructions.",
    where: "Emergency Action Panel, Area-Specific Alert",
  },
];

const DATA = [
  ["IMD gridded rainfall 2000–2023 + NASA POWER weather + India Flood Inventory", "Training and testing the v2 Logistic Regression flood model"],
  ["Open-Meteo live weather + GloFAS river discharge (Flood API)", "Live AI prediction, live river outlook, rainfall anomaly, response plan"],
  ["Water-level & weather data (2020)", "Training the Random Forest models"],
  ["IMD district & state rainfall", "Rainfall anomaly, area alerts"],
  ["OpenStreetMap (Overpass, cached)", "Hospitals, schools, shelters, roads for Emergency GIS"],
  ["Odisha boundary, river network, historical flood layers (GeoPackage)", "Spatial risk index, propagation, risk map"],
  ["Census 2011 town populations", "City impact priority"],
];

const FLOW = [
  ["Data", "Live rain & river forecasts, water level, weather, rainfall, GIS layers, population"],
  ["Features", "Water-level means & changes, weather, date; risk factors per district"],
  ["Models", "Logistic Regression (live) + Random Forest Classifier & Regressor"],
  ["Spatial analysis", "Risk index, propagation, hotspots, city exposure"],
  ["Decision support", "What-if scenarios, alerts, emergency actions"],
  ["Dashboard", "Maps, forecasts, explanations, reports"],
];

export default function ModelsUsed() {
  return (
    <div style={card}>
      <h2 style={{ margin: 0 }}>Models &amp; Methods Used</h2>
      <div style={{ color: "#5b6b80", fontSize: 14, marginTop: 4 }}>
        Which model or method powers each part of Flood Guard AI, what it takes in and what it gives out.
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 14 }}>
        {FLOW.map(([t, d], i) => (
          <div key={t} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ background: "#f3f6fa", borderRadius: 10, padding: "8px 12px", minWidth: 140, maxWidth: 210 }}>
              <div style={{ fontWeight: 700, fontSize: 14 }}><span style={{ color: "#2563eb" }}>{i + 1}.</span> {t}</div>
              <div style={{ fontSize: 12, color: "#475569" }}>{d}</div>
            </div>
            {i < FLOW.length - 1 && <span style={{ color: "#94a3b8", fontSize: 18 }}>→</span>}
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12, marginTop: 16 }}>
        {MODELS.map((m) => (
          <div key={m.name} style={{ border: "1px solid #dbe3ea", borderRadius: 10, padding: 14, background: "#fbfcfe" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "flex-start" }}>
              <b style={{ color: m.color, fontSize: 15 }}>{m.name}</b>
              <span style={{ fontSize: 11, background: m.color, color: "#fff", borderRadius: 999, padding: "2px 8px", whiteSpace: "nowrap" }}>{m.kind}</span>
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, margin: "4px 0 8px" }}>{m.job}</div>
            <div style={{ fontSize: 13, lineHeight: 1.5 }}>
              <div><b>Work:</b> {m.work}</div>
              <div><b>Input:</b> {m.input}</div>
              <div><b>Output:</b> {m.output}</div>
              <div style={{ color: "#475569" }}><b>Details:</b> {m.details}</div>
              <div style={{ color: "#475569" }}><b>Shown in:</b> {m.where}</div>
            </div>
          </div>
        ))}
      </div>

      <h3 style={{ margin: "18px 0 6px", fontSize: 16 }}>Summary</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "#5b6b80" }}>
              {["Model / Method", "Type", "Job in the project", "Output"].map((h) => (
                <th key={h} style={{ ...cell, borderBottom: "1px solid #dbe3ea" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODELS.map((m) => (
              <tr key={m.name}>
                <td style={{ ...cell, fontWeight: 600 }}>{m.name}</td>
                <td style={cell}>{m.kind}</td>
                <td style={cell}>{m.job}</td>
                <td style={cell}>{m.output}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 style={{ margin: "18px 0 6px", fontSize: 16 }}>Data used by these models</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <tbody>
            {DATA.map(([d, u]) => (
              <tr key={d}><td style={{ ...cell, fontWeight: 600, width: "45%" }}>{d}</td><td style={cell}>{u}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ fontSize: 12, color: "#5b6b80", marginTop: 10 }}>
        Why two kinds of model? Random Forest handles the non-linear water-level data and gives feature importance;
        for predicting real flood events from rainfall, Logistic Regression won a fair test on unseen years (2017–2023)
        against Random Forest, XGBoost and a rainfall rule, and its coefficients make each warning easy to explain.
      </div>
    </div>
  );
}
