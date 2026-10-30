"""Step 3: train the final flood early-warning model (the set-up that won in s04_experiments).

  Question: will a flood start in this district today or in the next 7 days?   (variant C)
  Train 2000-2016, test 2017-2023 (never seen). Alert levels are set on 2013-2016 inside training.
  District ID and land-cover class code are NOT used (they are labels, not measurements).
  Fair comparison: every model raises alerts on the same 3% of district-days; we report the share
  of real floods that got an alert (event hit rate) and the lift over random alerts.

    python -m pipeline_v2.s03_train_models
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_v2.common import DATASET, MODELS, REPORTS, TRAIN_END, VALID_FROM

LO, HI = 0, 7                         # flood start today .. +7 days
TARGET = "flood_within_7d"
ALERT_BUDGET = 0.03                   # Warning = top 3% of district-days
LEVEL_SHARES = {"Watch": 0.06, "Warning": 0.03, "Severe": 0.01}
DROP = {"static_district_id", "static_lulc_class"}
BASE_FEATURES = ["rain_mm", "rain_3d", "rain_7d", "rain_15d", "rain_30d", "rain_max_3d", "rain_api",
                 "rain_7d_normal", "rain_7d_anomaly", "doy_sin", "doy_cos", "monsoon", "district_flood_freq"]
OPTIONAL = ["humidity", "humidity_3d", "temp_c", "wind_ms"]


class RainRule:
    """Baseline: the more rain in the last 3 days, the higher the alert (what a simple warning system does)."""
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.clip(X["rain_3d"].to_numpy() / 300.0, 0, 1)
        return np.column_stack([1 - p, p])


def models(pos_weight):
    out = {"Rainfall rule (baseline)": RainRule(),
           "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.3)),
           "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=200, max_depth=10, max_features=0.5,
                                                   class_weight="balanced_subsample", n_jobs=-1, random_state=42)}
    try:
        from xgboost import XGBClassifier
        out["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                                       min_child_weight=50, reg_lambda=5, scale_pos_weight=pos_weight, n_jobs=-1, random_state=42)
    except ImportError:
        print("   (xgboost not installed: skipped)")
    return out


def add_target(df):
    g = df.groupby("district")["flood_start"]
    s = sum(g.shift(-h).fillna(0) for h in range(LO, HI + 1))
    return df.assign(**{TARGET: (s > 0).astype(int)})


def event_hit_rate(full, p, thr):
    """Share of real flood starts that had an alert on the start day or up to 7 days before."""
    al = pd.Series(p >= thr, index=pd.MultiIndex.from_frame(full[["district", "date"]]))
    starts = full.loc[full["flood_start"] == 1, ["district", "date"]]
    hits = sum(any(al.get((s.district, s.date - pd.Timedelta(days=h)), False) for h in range(LO, HI + 1))
               for _, s in starts.iterrows())
    return round(hits / max(len(starts), 1), 3), int(len(starts))


def main():
    print("[1] Loading", DATASET)
    df = pd.read_csv(DATASET, parse_dates=["date"]).sort_values(["district", "date"]).reset_index(drop=True)
    df = add_target(df)
    feats = BASE_FEATURES + [c for c in OPTIONAL if c in df and df[c].notna().mean() > 0.8] + \
        [c for c in df.columns if c.startswith("static_") and c not in DROP]
    feats = [f for f in feats if f in df]
    med = df[feats].median()
    df[feats] = df[feats].fillna(med)
    rows = df[(df["flood_active"] == 0) | (df["flood_start"] == 1)]      # ongoing flood days: nothing to predict
    yr = rows.date.dt.year
    train, test = rows[yr <= TRAIN_END], rows[yr > TRAIN_END]
    fit_part, val_part = train[train.date.dt.year < VALID_FROM], train[train.date.dt.year >= VALID_FROM]
    full_test = df[df.date.dt.year > TRAIN_END]
    base = test[TARGET].mean()
    print(f"   features ({len(feats)}): {feats}")
    print(f"   train {len(train):,} rows ({int(train[TARGET].sum())} positive) | test {len(test):,} rows ({int(test[TARGET].sum())} positive)")
    pos_w = float((train[TARGET] == 0).sum() / max(train[TARGET].sum(), 1))

    results, fitted = [], {}
    for name, m in models(pos_w).items():
        print(f"[2] {name}")
        m.fit(fit_part[feats], fit_part[TARGET])                       # 1) alert levels from validation years
        pv = m.predict_proba(val_part[feats])[:, 1]
        levels = {k: float(np.quantile(pv, 1 - s)) for k, s in LEVEL_SHARES.items()}
        m.fit(train[feats], train[TARGET])                             # 2) refit on all training years
        p = m.predict_proba(test[feats])[:, 1]
        thr = levels["Warning"]
        hr, n_ev = event_hit_rate(full_test, m.predict_proba(full_test[feats])[:, 1], thr)
        a, y = p >= thr, test[TARGET].to_numpy()
        ap = average_precision_score(y, p)
        r = {"model": name, "roc_auc": round(roc_auc_score(y, p), 3), "pr_auc": round(ap, 4), "lift": round(ap / base, 1),
             "precision": round(precision_score(y, a, zero_division=0), 3), "recall": round(recall_score(y, a, zero_division=0), 3),
             "f1": round(f1_score(y, a, zero_division=0), 3), "brier": round(brier_score_loss(y, np.clip(p, 0, 1)), 4),
             "event_hit_rate": hr, "test_events": n_ev,
             "alerts_per_district_year": round(float(a.sum()) / max(test.district.nunique() * test.date.dt.year.nunique(), 1), 1)}
        results.append(r)
        fitted[name] = (m, levels, p)
        print("   ", r)

    ml = [r for r in results if r["model"] != "Rainfall rule (baseline)"]
    best = max(ml, key=lambda r: (r["event_hit_rate"], r["roc_auc"]))
    model, levels, p_best = fitted[best["model"]]
    question = "Will a flood start in this district today or in the next 7 days?"
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats, "threshold": levels["Warning"], "levels": levels, "name": best["model"],
                 "horizon_days": HI, "question": question, "medians": med.to_dict()}, MODELS / "flood_onset_model_v2.pkl")

    imp = None
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feats)
    elif hasattr(model, "named_steps"):
        imp = pd.Series(np.abs(model[-1].coef_[0]), index=feats)
    if imp is not None:
        imp.sort_values(ascending=False).rename("importance").to_csv(REPORTS / "feature_importance.csv")

    out = test[["date", "district", "district_name", "rain_3d", TARGET, "flood_start"]].copy()
    out["probability"] = p_best
    out["alert"] = p_best >= levels["Warning"]
    out.to_csv(REPORTS / "test_predictions.csv.gz", index=False, compression="gzip")

    summary = {"task": question, "train_years": f"{df.date.dt.year.min()}-{TRAIN_END}",
               "test_years": f"{TRAIN_END + 1}-{df.date.dt.year.max()}",
               "label_source": "India Flood Inventory v3 (real reported flood events)",
               "alert_rule": f"Warning = top {ALERT_BUDGET:.0%} of district-days (levels set on {VALID_FROM}-{TRAIN_END})",
               "features": feats, "results": results, "selected_model": best["model"],
               "selection_rule": "highest event hit rate at the same alert budget (then ROC-AUC)",
               "positive_rate_test": round(float(base), 4)}
    (REPORTS / "metrics_v2.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(results).to_csv(REPORTS / "model_comparison.csv", index=False)
    plots(test[TARGET].to_numpy(), {n: v[2] for n, v in fitted.items()})
    print("\nModel comparison on unseen years 2017-2023:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved model: {MODELS / 'flood_onset_model_v2.pkl'} ({best['model']}): "
          f"catches {best['event_hit_rate'] * 100:.0f}% of real floods, ROC-AUC {best['roc_auc']}")


def plots(y, probs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for n, p in probs.items():
        f, t, _ = roc_curve(y, p)
        ax[0].plot(f, t, label=f"{n} (AUC {roc_auc_score(y, p):.2f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{n} (AP {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(title="ROC curve (test years 2017-2023)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(y.mean(), color="k", ls=":", lw=0.8)
    ax[1].set(title="Precision-recall curve (test years)", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS / "roc_pr_curves.png", dpi=130)


if __name__ == "__main__":
    main()
