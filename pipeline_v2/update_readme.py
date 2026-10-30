"""Add the v2 flood early-warning results to README.md and show the new Datasets folder in the project tree.
The numbers are read from models/v2_reports/metrics_v2.json, so they always match the saved model.
A backup of the old README is kept in _backups/. Safe to run twice (the v2 section is replaced, not duplicated).

    python -m pipeline_v2.update_readme
"""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "models" / "v2_reports" / "metrics_v2.json"
START, END = "<!-- v2-start -->", "<!-- v2-end -->"


def section(m):
    rows = "\n".join(
        f"| {r['model']} | {r['roc_auc']:.3f} | {r['pr_auc']:.4f} | {r['lift']:.1f}x | {r['event_hit_rate'] * 100:.0f}% |"
        for r in m["results"])
    best = next(r for r in m["results"] if r["model"] == m["selected_model"])
    rule = next((r for r in m["results"] if "rule" in r["model"].lower()), None)
    vs = f" (simple rainfall rule: {rule['event_hit_rate'] * 100:.0f}%)" if rule else ""
    return f"""{START}
## Flood Early Warning v2 (trained on real flood events)

**Question:** {m['task']}

| | |
|---|---|
| Labels | {m['label_source']}: {best['test_events']} real flood starts in the test years |
| Inputs | IMD gridded daily rainfall (0.25 deg), NASA POWER weather, season, elevation, distance to river |
| Not used | district ID and land-cover class code (they are labels, not measurements, and let models memorise districts) |
| Split | train {m['train_years']}, test {m['test_years']} (time split: the model never sees the future) |
| Alerts | {m['alert_rule']} |

**Results on the unseen years {m['test_years']}** (every model raises the same number of alerts):

| Model | ROC-AUC | PR-AUC | Lift over random | Real floods warned |
|---|---|---|---|---|
{rows}

Selected model: **{m['selected_model']}**: warns of **{best['event_hit_rate'] * 100:.0f}%** of real flood starts
on the day or up to 7 days before{vs}, with about {best['alerts_per_district_year']:.0f} alert days per district per year.

Floods are rare (about {m['positive_rate_test'] * 100:.1f}% of district-days), so PR-AUC values are low in absolute terms;
the lift column shows how much better than random alerts each model is. Logistic Regression beat the tree models
because only {best['test_events']} test events (and a few hundred training events) exist: the simpler model generalises better.
Three problem set-ups were compared first (`pipeline_v2/s04_experiments.py`, results in `models/v2_reports/experiments.csv`).

**Dashboard:** the "Flood Early Warning v2" panel shows the warning level of every district for any past date,
the main reasons for each warning, the last 90 days of rainfall against the probability, and a what-if (more rain).
API: `/api/v2/warnings`, `/api/v2/district/{{name}}`, `/api/v2/what-if`, `/api/v2/model-report`.

**Run again:** `python -m pipeline_v2.run_all` (download, dataset, model, risk map).
All data is in `Datasets/` (see `Datasets/README.md`).
{END}"""


def main():
    m = json.loads(METRICS.read_text())
    text = README.read_text(encoding="utf-8")
    bk = ROOT / "_backups" / f"README_before_v2_{datetime.now():%Y%m%d_%H%M}.md"
    bk.parent.mkdir(exist_ok=True)
    shutil.copy2(README, bk)

    new = section(m)
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: new, text, flags=re.S)
    else:
        anchor = re.search(r"^## Datasets\s*$", text, flags=re.M)
        text = (text[:anchor.start()] + new + "\n\n" + text[anchor.start():]) if anchor else (text.rstrip() + "\n\n" + new + "\n")

    # project tree: replace the separate data folders with one Datasets line
    lines, out, added = text.splitlines(), [], False
    for ln in lines:
        t = re.match(r"^(\s*[│ ]*[├└]──\s*)(data|processed_data|DEM|LULC)/?\s*$", ln)
        if t:
            if not added:
                out.append(f"{t.group(1)}Datasets          (data, processed_data, DEM, LULC)")
                added = True
            continue
        out.append(ln)
    text = "\n".join(out) + "\n"
    if "pipeline_v2" not in "\n".join(l for l in out if "──" in l):
        text = text.replace(f"Datasets          (data, processed_data, DEM, LULC)",
                            "Datasets          (data, processed_data, DEM, LULC)\n├── pipeline_v2       (v2 flood early warning)", 1)
    README.write_text(text, encoding="utf-8")
    print("README.md updated. Old version:", bk)


if __name__ == "__main__":
    main()
