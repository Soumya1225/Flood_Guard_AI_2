import { useEffect, useState } from "react";

// District flood response plan: tier, people to shelter, relief needs, priorities, contacts,
// public warning messages (English / Hindi / Odia) and a saved response checklist.

const card = { background: "#ffffff", border: "1px solid #dbe3ea", borderRadius: 14, padding: 20, marginBottom: 18 };
const n0 = (v) => (typeof v === "number" ? v.toLocaleString("en-IN", { maximumFractionDigits: 0 }) : "–");
const LANGS = [["en", "English"], ["hi", "हिन्दी (Hindi)"], ["or", "ଓଡ଼ିଆ (Odia)"]];

export default function ResponsePlan({ selectedDistrictName }) {
  const district = selectedDistrictName || "Cuttack";
  const [plan, setPlan] = useState(null);
  const [err, setErr] = useState("");
  const [lang, setLang] = useState("or");
  const [copied, setCopied] = useState(false);

  const load = () => {
    setErr("");
    fetch(`/api/response/${encodeURIComponent(district)}`)
      .then(async (r) => {
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
        return j;
      })
      .then(setPlan)
      .catch((e) => setErr(String(e.message || e)));
  };

  useEffect(() => {
    setPlan(null);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [district]);

  const toggle = (task) => {
    fetch(`/api/response/${encodeURIComponent(district)}/task/${task.id}?done=${!task.done}`, { method: "POST" })
      .then(() => setPlan((p) => ({
        ...p,
        checklist: p.checklist.map((t) => (t.id === task.id ? { ...t, done: !t.done, updated_at: new Date().toISOString().slice(0, 19) } : t)),
      })))
      .catch(() => {});
  };

  const copy = () => {
    const text = plan?.messages?.[lang] || "";
    try {
      navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  if (err) return <section style={card}><h3 style={{ marginTop: 0 }}>Flood Response Plan — {district}</h3><p style={{ color: "#b91c1c" }}>Response plan unavailable: {err}</p></section>;
  if (!plan) return <section style={card}><h3 style={{ marginTop: 0 }}>Flood Response Plan — {district}</h3><p style={{ color: "#5b6b80" }}>Preparing the response plan from live data…</p></section>;

  const t = plan.tier;
  const r = plan.relief;
  const required = plan.checklist.filter((c) => c.required);
  const doneCount = required.filter((c) => c.done).length;

  return (
    <div>
      <section style={{ ...card, borderLeft: "none", background: "#fff" }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ background: t.color, color: "#fff", borderRadius: 12, padding: "14px 20px", minWidth: 180, textAlign: "center" }}>
            <div style={{ fontSize: 13, opacity: 0.9 }}>Response level</div>
            <div style={{ fontSize: 26, fontWeight: 800 }}>{t.name}</div>
            <div style={{ fontSize: 15, fontWeight: 700 }}>{t.action}</div>
          </div>
          <div style={{ flex: 1, minWidth: 260 }}>
            <h3 style={{ margin: 0 }}>Flood Response Plan — {plan.district}</h3>
            <p style={{ margin: "6px 0 0", color: "#334155", lineHeight: 1.5 }}>
              {plan.reasons.length ? <>Why: {plan.reasons.join("; ")}.</> : "No flood signal in the live 7-day outlook."}
              {plan.peak_date ? <> Highest risk around <b>{plan.peak_date}</b>.</> : null}
              {" "}Long-term spatial risk: {plan.spatial_risk_score?.toFixed?.(3) ?? "–"}.
            </p>
            {plan.notes?.length > 0 && <p style={{ margin: "6px 0 0", fontSize: 13, color: "#b45309" }}>{plan.notes.join(" ")}</p>}
          </div>
        </div>
      </section>

      <section style={card}>
        <h3 style={{ marginTop: 0 }}>People and relief needs</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
          {[
            ["Population (WorldPop 2020)", n0(plan.population_worldpop_2020)],
            ["Population at risk", n0(plan.population_at_risk)],
            ["People to shelter", n0(r.people_to_shelter)],
            ["Relief camps", n0(r.relief_camps)],
            ["Drinking water / day", `${n0(r.drinking_water_litres_per_day)} L`],
            ["Dry rations / day", `${n0(r.dry_rations_kg_per_day)} kg`],
            ["Shelter space", `${n0(r.shelter_space_m2)} m²`],
            ["Toilets", n0(r.toilets)],
            ["Rescue boats", n0(r.rescue_boats)],
          ].map(([l, v]) => (
            <div className="summary-card" key={l}><span>{l}</span><strong>{v}</strong></div>
          ))}
        </div>
        <p style={{ fontSize: 13, color: "#5b6b80", marginTop: 10, lineHeight: 1.5 }}>{r.basis}</p>
      </section>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 18 }}>
        <section style={{ ...card, marginBottom: 0 }}>
          <h3 style={{ marginTop: 0 }}>Evacuation priority (towns)</h3>
          {plan.evacuation_priority.length === 0 ? (
            <p style={{ color: "#5b6b80" }}>No town list available for this district.</p>
          ) : (
            <table>
              <thead><tr><th>#</th><th>Town</th><th>Population (2011)</th></tr></thead>
              <tbody>
                {plan.evacuation_priority.map((c, i) => (
                  <tr key={c.name}><td>{i + 1}</td><td>{c.name}</td><td>{n0(c.population_2011)}</td></tr>
                ))}
              </tbody>
            </table>
          )}
          <p style={{ fontSize: 13, color: "#5b6b80" }}>Within each town, start with river-bank and low-lying wards, then the elderly, children, pregnant women and disabled people.</p>
        </section>

        <section style={{ ...card, marginBottom: 0 }}>
          <h3 style={{ marginTop: 0 }}>Rescue resources and contacts</h3>
          <p style={{ margin: "0 0 8px" }}>
            <b>{plan.resources.ndrf.name}</b>: about {plan.resources.ndrf.distance_km} km away
            (≈ {plan.resources.ndrf.travel_hours_by_road} h by road).
          </p>
          <p style={{ margin: "0 0 10px", fontSize: 14, color: "#334155" }}>{plan.resources.district_teams}</p>
          <table>
            <tbody>
              {plan.resources.contacts.map((c) => (
                <tr key={c.number}><td>{c.name}</td><td style={{ fontWeight: 700, fontSize: 16 }}>{c.number}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section style={{ ...card, marginTop: 18 }}>
        <h3 style={{ marginTop: 0 }}>Public warning message</h3>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
          {LANGS.map(([k, label]) => (
            <button key={k} type="button" onClick={() => setLang(k)}
              style={{ padding: "7px 14px", borderRadius: 8, border: "1px solid #cbd5e1", cursor: "pointer",
                       background: lang === k ? "#0f2a3d" : "#fff", color: lang === k ? "#fff" : "#0f2a3d", fontWeight: 600 }}>
              {label}
            </button>
          ))}
          <button type="button" onClick={copy}
            style={{ padding: "7px 14px", borderRadius: 8, border: 0, cursor: "pointer", background: "#f0a91c", color: "#0f2a3d", fontWeight: 700 }}>
            {copied ? "Copied ✓" : "Copy message"}
          </button>
        </div>
        <div style={{ background: "#f6f9fb", border: "1px solid #dbe3ea", borderRadius: 10, padding: 14, fontSize: 16, lineHeight: 1.7 }}>
          {plan.messages[lang]}
        </div>
        <p style={{ fontSize: 13, color: "#5b6b80" }}>Ready to send by SMS, WhatsApp, radio or loudspeaker. Hindi and Odia texts should be checked by a native speaker before official use.</p>
      </section>

      <section style={card}>
        <h3 style={{ marginTop: 0 }}>Response checklist <span style={{ fontSize: 14, color: "#5b6b80", fontWeight: 500 }}>({doneCount} of {required.length} required tasks done)</span></h3>
        <div style={{ display: "grid", gap: 6 }}>
          {plan.checklist.map((c) => (
            <label key={c.id} style={{ display: "flex", gap: 10, alignItems: "flex-start", padding: "8px 10px", borderRadius: 8,
                                        background: c.required ? "#f6f9fb" : "transparent", opacity: c.required ? 1 : 0.55, cursor: "pointer" }}>
              <input type="checkbox" checked={c.done} onChange={() => toggle(c)} style={{ marginTop: 3 }} />
              <span style={{ flex: 1, textDecoration: c.done ? "line-through" : "none" }}>{c.text}</span>
              <span style={{ fontSize: 12, color: "#5b6b80", whiteSpace: "nowrap" }}>
                {["Normal", "Watch", "Warning", "Severe"][c.tier]}{c.done && c.updated_at ? ` · ${c.updated_at.replace("T", " ")}` : ""}
              </span>
            </label>
          ))}
        </div>
        <p style={{ fontSize: 13, color: "#5b6b80" }}>Tasks for the current response level are highlighted; ticks are saved in the Flood Guard AI database. {plan.note}</p>
      </section>
    </div>
  );
}
