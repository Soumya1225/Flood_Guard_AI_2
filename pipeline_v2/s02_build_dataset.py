"""Step 2: build the district-day dataset (Odisha, 2000-2023) with real flood labels.

Each row = one district on one day. Features use ONLY information available on that day:
  rainfall today / last 3, 7, 15, 30 days, wettest day of last 3 days, antecedent precipitation index,
  rainfall anomaly vs the district's normal for that time of year (normal computed from training years only),
  humidity / temperature (NASA POWER), season, and static terrain indicators from the GIS layer.
Target: flood_next_3d = 1 if a flood (India Flood Inventory) STARTS in this district in the next 3 days.
Days while a flood is already going on are left out (nothing to predict there).

    python -m pipeline_v2.s02_build_dataset
"""
import io
import json

import numpy as np
import pandas as pd

from pipeline_v2.common import (BBOX, DATASET, END_YEAR, HORIZON, IFI_CSV, IMD_DIR, POWER_DIR, ROOT, START_YEAR,
                                TRAIN_END, load_districts, match_district, norm, split_districts)

NLAT, NLON, LAT0, LON0, RES = 129, 135, 6.5, 66.5, 0.25
STATIC_KEEP = ("elev", "slope", "river", "drain", "lulc", "built", "urban", "water", "crop", "tree", "forest",
               "pop", "dist")
STATIC_DROP = ("hist", "risk", "score", "level", "flood", "category", "rank", "prob")


# ---------------------------------------------------------------- rainfall
def read_imd_year(year):
    f = IMD_DIR / "rain" / f"{year}.grd"
    days = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
    a = np.fromfile(f, dtype="<f4")
    a = a[: days * NLAT * NLON].reshape(days, NLAT, NLON)
    a = np.where(a < 0, np.nan, a)                              # -999 = missing
    return pd.date_range(f"{year}-01-01", periods=days), a


def district_cells(districts):
    """Grid cells (row, col) whose centre is inside each district; nearest cells for very small districts."""
    from shapely.geometry import Point
    lats = LAT0 + RES * np.arange(NLAT)
    lons = LON0 + RES * np.arange(NLON)
    rows = np.where((lats >= BBOX[1] - 0.5) & (lats <= BBOX[3] + 0.5))[0]
    cols = np.where((lons >= BBOX[0] - 0.5) & (lons <= BBOX[2] + 0.5))[0]
    cells = {}
    for _, d in districts.iterrows():
        inside = [(r, c) for r in rows for c in cols if d.geometry.contains(Point(lons[c], lats[r]))]
        if len(inside) < 2:
            ctr = d.geometry.representative_point()
            dist = sorted(((lats[r] - ctr.y) ** 2 + (lons[c] - ctr.x) ** 2, r, c) for r in rows for c in cols)
            inside = [(r, c) for _, r, c in dist[:4]]
        cells[d.district] = inside
    return cells


def district_rain(districts):
    cells = district_cells(districts)
    frames = []
    for y in range(START_YEAR, END_YEAR + 1):
        f = IMD_DIR / "rain" / f"{y}.grd"
        if not f.exists():
            print(f"   {y}: no IMD file, skipped")
            continue
        dates, a = read_imd_year(y)
        for d, cl in cells.items():
            r, c = zip(*cl)
            vals = np.nanmean(a[:, list(r), list(c)], axis=1)
            frames.append(pd.DataFrame({"date": dates, "district": d, "rain_mm": vals}))
        print(f"   {y} done")
    df = pd.concat(frames, ignore_index=True)
    df["rain_mm"] = df["rain_mm"].fillna(0.0)
    return df


# ---------------------------------------------------------------- weather (optional)
def district_weather():
    out = []
    for f in sorted(POWER_DIR.glob("*.csv")):
        txt = f.read_text(errors="ignore")
        if "-END HEADER-" in txt:
            txt = txt.split("-END HEADER-", 1)[1]
        w = pd.read_csv(io.StringIO(txt))
        w["date"] = pd.to_datetime(w["YEAR"].astype(str) + w["DOY"].astype(str).str.zfill(3), format="%Y%j")
        w = w.replace(-999, np.nan)
        w = w.rename(columns={"T2M": "temp_c", "RH2M": "humidity", "WS2M": "wind_ms"})
        w["district"] = f.stem
        out.append(w[["date", "district", "temp_c", "humidity", "wind_ms"]])
    return pd.concat(out, ignore_index=True) if out else None


# ---------------------------------------------------------------- flood labels
def flood_events(known):
    raw = pd.read_csv(IFI_CSV, encoding_errors="ignore", low_memory=False)
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in raw.columns}
    pick = lambda *keys: next((cols[k] for k in cols if all(x in k for x in keys)), None)
    c_start, c_end = pick("start", "date") or pick("start"), pick("end", "date") or pick("end")
    c_state, c_dist = pick("state"), pick("district")
    if not all([c_start, c_end, c_state, c_dist]):
        raise SystemExit(f"Unexpected flood inventory columns: {list(raw.columns)}")
    od = raw[raw[c_state].astype(str).str.contains("odisha|orissa", case=False, na=False)].copy()
    od["start"] = pd.to_datetime(od[c_start], errors="coerce", dayfirst=True)
    od["end"] = pd.to_datetime(od[c_end], errors="coerce", dayfirst=True).fillna(od["start"])
    rows, unmatched = [], set()
    for _, e in od.dropna(subset=["start"]).iterrows():
        for name in split_districts(e[c_dist]):
            d = match_district(name, known)
            if d:
                rows.append({"district": d, "start": e["start"].normalize(), "end": max(e["end"], e["start"]).normalize()})
            else:
                unmatched.add(name)
    ev = pd.DataFrame(rows).drop_duplicates()
    ev = ev[(ev.start.dt.year >= START_YEAR) & (ev.start.dt.year <= END_YEAR)]
    if unmatched:
        print("   district names not matched (ignored):", sorted(unmatched)[:15])
    return ev


def add_labels(df, events):
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    df["flood_start"] = 0
    df["flood_active"] = 0
    key = pd.MultiIndex.from_frame(df[["district", "date"]])
    pos = pd.Series(np.arange(len(df)), index=key)
    for _, e in events.iterrows():
        k = (e.district, e.start)
        if k in pos.index:
            df.loc[pos[k], "flood_start"] = 1
        days = pd.date_range(e.start, e.end)
        idx = [pos[(e.district, d)] for d in days if (e.district, d) in pos.index]
        df.loc[idx, "flood_active"] = 1
    g = df.groupby("district")["flood_start"]
    nxt = sum(g.shift(-h) for h in range(1, HORIZON + 1))
    df[f"flood_next_{HORIZON}d"] = (nxt.fillna(0) > 0).astype(int)
    return df


# ---------------------------------------------------------------- features
def add_features(df):
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    g = df.groupby("district")["rain_mm"]
    for w in (3, 7, 15, 30):
        df[f"rain_{w}d"] = g.transform(lambda s: s.rolling(w, min_periods=1).sum())
    df["rain_max_3d"] = g.transform(lambda s: s.rolling(3, min_periods=1).max())
    def api(s, k=0.9):
        out, acc = np.empty(len(s)), 0.0
        for i, v in enumerate(s.to_numpy()):
            acc = acc * k + v
            out[i] = acc
        return pd.Series(out, index=s.index)
    df["rain_api"] = g.transform(api)
    df["doy"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * df.doy / 365.25), np.cos(2 * np.pi * df.doy / 365.25)
    df["monsoon"] = df["month"].between(6, 10).astype(int)
    # normal rainfall for this district and time of year, from TRAINING years only (no look-ahead)
    tr = df[df.date.dt.year <= TRAIN_END]
    clim = tr.groupby(["district", "doy"])["rain_7d"].mean().rename("rain_7d_normal").reset_index()
    clim["rain_7d_normal"] = clim.groupby("district")["rain_7d_normal"].transform(
        lambda s: s.rolling(15, center=True, min_periods=1).mean())
    df = df.merge(clim, on=["district", "doy"], how="left")
    df["rain_7d_anomaly"] = df["rain_7d"] - df["rain_7d_normal"]
    if "humidity" in df:
        h = df.groupby("district")["humidity"]
        df["humidity_3d"] = h.transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df


def static_features(known):
    p = ROOT / "gis" / "odisha_spatial_risk.csv"
    if not p.exists():
        return None, []
    s = pd.read_csv(p)
    name_col = next((c for c in s.columns if "district" in c.lower() or c.lower() in ("name", "name_2")), None)
    if name_col is None:
        return None, []
    s["district"] = s[name_col].map(lambda x: match_district(x, known))
    keep = [c for c in s.select_dtypes("number").columns
            if any(k in c.lower() for k in STATIC_KEEP) and not any(k in c.lower() for k in STATIC_DROP)]
    s = s.dropna(subset=["district"]).groupby("district")[keep].mean().reset_index()
    return s.rename(columns={c: f"static_{c}" for c in keep}), [f"static_{c}" for c in keep]


def main():
    print("[1] Districts (GADM)...")
    districts = load_districts()
    known = set(districts.district)
    print(f"   {len(known)} Odisha districts")
    print("[2] District rainfall from IMD grids...")
    df = district_rain(districts)
    print("[3] Weather (NASA POWER)...")
    w = district_weather()
    if w is not None:
        df = df.merge(w, on=["date", "district"], how="left")
        print(f"   weather for {w.district.nunique()} districts")
    else:
        print("   none found (optional)")
    print("[4] Flood events (India Flood Inventory)...")
    ev = flood_events(known)
    print(f"   {len(ev)} district flood events {START_YEAR}-{END_YEAR} "
          f"({(ev.start.dt.year <= TRAIN_END).sum()} train, {(ev.start.dt.year > TRAIN_END).sum()} test)")
    ev.to_csv(DATASET.parent / "flood_events_odisha_v2.csv", index=False)
    print("[5] Features and labels...")
    df = add_features(df)
    df = add_labels(df, ev)
    # flood frequency per district from TRAINING years only
    freq = ev[ev.start.dt.year <= TRAIN_END].groupby("district").size() / (TRAIN_END - START_YEAR + 1)
    df["district_flood_freq"] = df["district"].map(freq).fillna(0.0)
    st, st_cols = static_features(known)
    if st is not None:
        df = df.merge(st, on="district", how="left")
        print("   static GIS features:", st_cols)
    names = dict(zip(districts.district, districts.district_name))
    df["district_name"] = df["district"].map(names)
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET, index=False, compression="gzip")
    t = f"flood_next_{HORIZON}d"
    summary = {"rows": int(len(df)), "districts": int(df.district.nunique()),
               "years": f"{df.date.dt.year.min()}-{df.date.dt.year.max()}", "events": int(len(ev)),
               "positive_rows": int(df[t].sum()), "static_features": st_cols}
    (DATASET.parent / "district_daily_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved", DATASET)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
