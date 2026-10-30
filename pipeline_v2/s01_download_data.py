"""Step 1: download the long-term datasets (resumable: files already downloaded are skipped).

  * India Flood Inventory v3/v4 (Zenodo): real flood events 1967-2023 with districts -> the target
  * IMD gridded daily rainfall 0.25 deg (imdpune.gov.in via imdlib), 2000-2023     -> main driver
  * NASA POWER daily weather (humidity, temperature) for each district centre       -> extra features

    python -m pipeline_v2.s01_download_data
"""
import sys
import time

import requests

from pipeline_v2.common import END_YEAR, IFI_CSV, IMD_DIR, POWER_DIR, START_YEAR, V2, load_districts

IFI_URLS = [
    "https://zenodo.org/records/16994648/files/India_Flood_Inventory_v3.csv?download=1",
    "https://zenodo.org/api/records/16994648/files/India_Flood_Inventory_v3.csv/content",
]
POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,WS2M"
             "&community=AG&longitude={lon:.3f}&latitude={lat:.3f}&start={s}0101&end={e}1231&format=CSV")


def get(url, timeout=120):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
            print(f"   HTTP {r.status_code} ({len(r.content)} bytes)")
        except Exception as e:
            print("   ", type(e).__name__, str(e)[:120])
        time.sleep(3 * (attempt + 1))
    return None


def flood_inventory():
    if IFI_CSV.exists() and IFI_CSV.stat().st_size > 100_000:
        print("[1] India Flood Inventory: already downloaded")
        return True
    print("[1] Downloading India Flood Inventory (1.8 MB)...")
    for u in IFI_URLS:
        data = get(u)
        if data:
            IFI_CSV.write_bytes(data)
            print("   saved", IFI_CSV)
            return True
    print("   FAILED. Download it by hand from https://zenodo.org/records/16994648 "
          f"(India_Flood_Inventory_v3.csv) and save it as {IFI_CSV}")
    return False


def imd_rain():
    import imdlib as imd
    IMD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[2] IMD daily rainfall {START_YEAR}-{END_YEAR} (about 25 MB per year)...")
    ok = 0
    for y in range(START_YEAR, END_YEAR + 1):
        f = IMD_DIR / "rain" / f"{y}.grd"
        if f.exists() and f.stat().st_size > 5_000_000:
            ok += 1
            continue
        for attempt in range(3):
            try:
                imd.get_data("rain", y, y, fn_format="yearwise", file_dir=str(IMD_DIR))
                break
            except Exception as e:
                print(f"   {y}: retry ({type(e).__name__})")
                time.sleep(5)
        if f.exists():
            ok += 1
            print(f"   {y} ok")
        else:
            print(f"   {y} FAILED (run this step again later; finished years are kept)")
    print(f"   {ok} of {END_YEAR - START_YEAR + 1} years ready")
    return ok


def nasa_power():
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    d = load_districts()
    print(f"[3] NASA POWER daily weather for {len(d)} districts...")
    ok = 0
    for _, row in d.iterrows():
        f = POWER_DIR / f"{row.district}.csv"
        if f.exists() and f.stat().st_size > 50_000:
            ok += 1
            continue
        c = row.geometry.representative_point()
        data = get(POWER_URL.format(lon=c.x, lat=c.y, s=START_YEAR, e=END_YEAR), timeout=180)
        if data:
            f.write_bytes(data)
            ok += 1
            print(f"   {row.district_name} ok")
        else:
            print(f"   {row.district_name} failed (optional; the model works without it)")
    print(f"   {ok} of {len(d)} districts ready")


if __name__ == "__main__":
    V2.mkdir(parents=True, exist_ok=True)
    have_ifi = flood_inventory()
    years = imd_rain()
    if "--no-weather" not in sys.argv:
        nasa_power()
    if not have_ifi or years < 10:
        print("\nNot enough data yet: fix the failed downloads above, then run this step again.")
        sys.exit(1)
    print("\nDownloads complete.")
