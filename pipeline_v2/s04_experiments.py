"""Step 4: compare problem set-ups honestly, then keep the best one.

Same data, same time split (train 2000-2016, test 2017-2023). Variants of the question asked:
  A  flood starts in the next 1-3 days            (first version)
  B  flood starts today or in the next 3 days     (today's observed rain is used: a same-day nowcast + 3-day warning)
  C  flood starts today or in the next 7 days     (weekly warning window)
Features: rainfall, weather, season, terrain; the district ID and the land-cover class code are removed
(they are labels, not measurements, and let trees memorise districts).
Fair comparison: every model may raise alerts on the same share of district-days (ALERT_BUDGET);
we report how many real floods got an alert (event hit rate) and the lift over random alerts.

    python -m pipeline_v2.s04_experiments
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_v2.common import DATASET, REPORTS, TRAIN_END

ALERT_BUDGET = 0.03          # alerts on 3% of district-days (about 11 alert days per district per year)
DROP = {"static_district_id", "static_lulc_class"}
VARIANTS = {"A: start in next 1-3 days": (1, 3), "B: start today..+3 days": (0, 3), "C: start today..+7 days": (0, 7)}


def target(df, lo, hi):
    g = df.groupby("district")["flood_start"]
    s = sum(g.shift(-h).fillna(0) for h in range(lo, hi + 1))
    return (s > 0).astype(int)


def models(pos_w):
    out = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.3)),
           "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=200, max_depth=10, max_features=0.5,
                                                   class_weight="balanced_subsample", n_jobs=-1, random_state=42)}
    try:
        from xgboost import XGBClassifier
        out["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                                       min_child_weight=50, reg_lambda=5, scale_pos_weight=pos_w, n_jobs=-1, random_state=42)
    except ImportError:
        pass
    return out


def hit_rate(test, p, lo, hi, budget=ALERT_BUDGET):
    thr = np.quantile(p, 1 - budget)
    al = pd.Series(p >= thr, index=pd.MultiIndex.from_frame(test[["district", "date"]]))
    starts = test.loc[test["flood_start"] == 1, ["district", "date"]]
    hits = sum(any(al.get((s.district, s.date - pd.Timedelta(days=h)), False) for h in range(lo, hi + 1))
               for _, s in starts.iterrows())
    return hits / max(len(starts), 1), len(starts)


def main():
    df = pd.read_csv(DATASET, parse_dates=["date"]).sort_values(["district", "date"]).reset_index(drop=True)
    feats = [c for c in ["rain_mm", "rain_3d", "rain_7d", "rain_15d", "rain_30d", "rain_max_3d", "rain_api", "rain_7d_normal",
                         "rain_7d_anomaly", "doy_sin", "doy_cos", "monsoon", "district_flood_freq", "humidity",
                         "humidity_3d", "temp_c", "wind_ms"] if c in df] + \
            [c for c in df.columns if c.startswith("static_") and c not in DROP]
    df[feats] = df[feats].fillna(df[feats].median())
    rows = []
    for vname, (lo, hi) in VARIANTS.items():
        y = target(df, lo, hi)
        d = df.assign(y=y)
        # rows with an ongoing flood are excluded unless the variant includes today (then only the start day counts)
        d = d[(d["flood_active"] == 0) | (d["flood_start"] == 1)] if lo == 0 else d[d["flood_active"] == 0]
        tr, te = d[d.date.dt.year <= TRAIN_END], d[d.date.dt.year > TRAIN_END]
        base = te["y"].mean()
        rule = te["rain_3d"].to_numpy()
        full_te = df[df.date.dt.year > TRAIN_END]
        hr, n = hit_rate(full_te, full_te["rain_3d"].to_numpy(), lo, hi)
        rows.append({"variant": vname, "model": "Rainfall rule (3-day rain)", "roc_auc": round(roc_auc_score(te["y"], rule), 3),
                     "pr_auc": round(average_precision_score(te["y"], rule), 4), "lift": round(average_precision_score(te["y"], rule) / base, 1),
                     "event_hit_rate": round(hr, 3), "test_events": n, "positive_rate": round(base, 4)})
        pos_w = float((tr["y"] == 0).sum() / max(tr["y"].sum(), 1))
        for mname, m in models(pos_w).items():
            m.fit(tr[feats], tr["y"])
            p = m.predict_proba(te[feats])[:, 1]
            hr, n = hit_rate(full_te, m.predict_proba(full_te[feats])[:, 1], lo, hi)
            ap = average_precision_score(te["y"], p)
            rows.append({"variant": vname, "model": mname, "roc_auc": round(roc_auc_score(te["y"], p), 3),
                         "pr_auc": round(ap, 4), "lift": round(ap / base, 1), "event_hit_rate": round(hr, 3),
                         "test_events": n, "positive_rate": round(base, 4)})
            print(rows[-1])
    res = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS / "experiments.csv", index=False)
    best = res[res.model != "Rainfall rule (3-day rain)"].sort_values(["event_hit_rate", "roc_auc"], ascending=False).iloc[0]
    (REPORTS / "experiments_best.json").write_text(json.dumps(best.to_dict(), indent=2, default=str))
    print(f"\nAll models raise alerts on the same {ALERT_BUDGET:.0%} of district-days.\n")
    print(res.to_string(index=False))
    print(f"\nBest: {best['variant']} with {best['model']} -> catches {best['event_hit_rate'] * 100:.0f}% of real floods, "
          f"ROC-AUC {best['roc_auc']}")


if __name__ == "__main__":
    main()
