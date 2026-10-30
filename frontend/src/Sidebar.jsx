// Sidebar navigation for Flood Guard AI (grouped like SpillWatch).

export const PAGES = [
  { id: "about", num: 0, group: "Home", label: "Project Overview", hint: "What Flood Guard AI does, how it works, models used and tested results." },
  { id: "dashboard", num: 1, group: "1. Dashboard", label: "Overview", hint: "Overall alert status, AI situation summary, state KPIs and priority districts." },
  { id: "map", num: 2, group: "1. Dashboard", label: "Risk Map", hint: "Odisha spatial risk map, district explorer, district comparison and risk distribution." },
  { id: "forecast", num: 3, group: "2. Forecasting", label: "Flood Forecast", hint: "Live 7-day flood forecast for every district (GloFAS river discharge + rainfall), plus the Random Forest water-level model reference." },
  { id: "weather", num: 4, group: "2. Forecasting", label: "Weather & Rainfall", hint: "Live weather, 7-day weather forecast, historical weather, rainfall anomaly and area alert." },
  { id: "prediction", num: 5, group: "3. AI Models", label: "AI Prediction", hint: "Live AI flood prediction for every district (trained model on live rainfall), past-event replay, Random Forest model and explainability." },
  { id: "models", num: 6, group: "3. AI Models", label: "Models & Performance", hint: "Models and methods used, model performance, configuration and feature importance." },
  { id: "impact", num: 7, group: "4. Impact & Response", label: "Impact & Spread", hint: "Affected neighbouring districts, hotspots, population at risk and city priority." },
  { id: "emergency", num: 8, group: "4. Impact & Response", label: "Emergency Response", hint: "District response plan (response level, people to shelter, relief needs, contacts, warning messages, checklist) and emergency GIS." },
  { id: "whatif", num: 9, group: "4. Impact & Response", label: "What-If Analysis", hint: "Change rainfall, water level and drainage to see how district risk changes." },
  { id: "data", num: 10, group: "5. Data", label: "Data & History", hint: "Data quality, feature readiness, project status and saved prediction / alert / scenario history." },
];

const GROUPS = [...new Set(PAGES.map((p) => p.group))];

export default function Sidebar({ page, setPage, districts = [], selectedDistrictId, setSelectedDistrictId }) {
  return (
    <aside className="fg-sidebar">
      <div className="fg-brand">
        <div className="fg-logo">FG</div>
        <div>
          <div className="fg-brand-name">Flood Guard AI</div>
          <div className="fg-brand-sub">Flood forecasting &amp; decision support</div>
        </div>
      </div>

      <div className="fg-group">District</div>
      <select
        className="fg-side-select"
        value={selectedDistrictId}
        onChange={(e) => setSelectedDistrictId(e.target.value)}
      >
        {districts.map((d) => (
          <option key={d.district_id} value={d.district_id}>{d.NAME_2}</option>
        ))}
      </select>

      <nav>
        {GROUPS.map((g) => (
          <div key={g}>
            <div className="fg-group">{g}</div>
            {PAGES.filter((p) => p.group === g).map((p) => (
              <button
                key={p.id}
                type="button"
                title={p.hint}
                className={"fg-nav" + (page === p.id ? " active" : "")}
                onClick={() => setPage(p.id)}
              >
                {p.label}
              </button>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
