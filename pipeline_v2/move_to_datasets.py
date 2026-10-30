"""One-time tidy-up: put every dataset folder inside one 'Datasets' folder and update the code paths.

  data/            -> Datasets/data/
  processed_data/  -> Datasets/processed_data/
  DEM/             -> Datasets/DEM/
  LULC/            -> Datasets/LULC/

Every code file it changes is backed up first in _backups/before_datasets_<time>/.
Run it with the backend and frontend STOPPED and no File Explorer window open inside those folders.

    python -m pipeline_v2.move_to_datasets
"""
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "Datasets"
FOLDERS = ["data", "processed_data", "DEM", "LULC"]
BACKUP = ROOT / "_backups" / f"before_datasets_{datetime.now():%Y%m%d_%H%M}"
SKIP = {"node_modules", "dist", "_backups", "_updates", "Datasets", "__pycache__", ".git", "venv", ".venv",
        "outputs", "database"} | set(FOLDERS)
EXT = {".py", ".ipynb", ".js", ".jsx", ".bat", ".ps1", ".cfg", ".toml", ".ini", ".txt"}
Q = r'(\\?["\'])'                                   # a quote, also the escaped \" used inside notebooks


def rules(name):
    n = re.escape(name)
    return [
        # ROOT / "data"     (also when "data" is on its own line after a /)
        (re.compile(r'(/\s*)' + Q + n + r'\2(?![\w/\\])'), lambda m: f'{m.group(1)}{m.group(2)}Datasets{m.group(2)} / {m.group(2)}{name}{m.group(2)}'),
        # os.path.join(BASE_DIR, "processed_data", ...)
        (re.compile(r'(,\s*)' + Q + n + r'\2(?=\s*[,)])'), lambda m: f'{m.group(1)}{m.group(2)}Datasets{m.group(2)}, {m.group(2)}{name}{m.group(2)}'),
        # "data/rainfall/x.csv"  or  "data\\rainfall"  (a path string starting with the folder name)
        (re.compile(Q + r'(\.?[/\\]{0,2})' + n + r'(?=[/\\])'), lambda m: f'{m.group(1)}{m.group(2)}Datasets/{name}'),
    ]


ALL_RULES = [r for f in FOLDERS for r in rules(f)]


def fix(text):
    for rx, rep in ALL_RULES:
        text = rx.sub(rep, text)
    return text


def code_files():
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() in EXT and p.name != "move_to_datasets.py":
                yield p


def main():
    present = [f for f in FOLDERS if (ROOT / f).exists()]
    if not present and DS.exists():
        print("Already done: everything is inside the Datasets folder.")
        return
    clash = [f for f in present if (DS / f).exists()]
    if clash:
        sys.exit(f"Stop: both {clash} and Datasets/{clash} exist. Nothing changed.")

    DS.mkdir(exist_ok=True)
    moved = []
    for f in present:
        try:
            (ROOT / f).rename(DS / f)
            moved.append(f)
        except OSError as e:
            for g in moved:                                   # undo, so the project is never half-moved
                (DS / g).rename(ROOT / g)
            sys.exit(f"Could not move '{f}' (a program is using it): {e}\n"
                     "Close VS Code tabs / File Explorer windows inside it, stop the backend, then run again. Nothing changed.")
    print("1. Moved into Datasets/:", ", ".join(moved))

    changed = []
    for p in code_files():
        s = p.read_text(encoding="utf-8", errors="ignore")
        s2 = fix(s)
        if s2 != s:
            dst = BACKUP / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            p.write_text(s2, encoding="utf-8")
            changed.append(str(p.relative_to(ROOT)))
    print(f"2. Updated paths in {len(changed)} files:")
    for c in changed:
        print("     ", c)
    print("   Backup of the original files:", BACKUP)

    (DS / "README.md").write_text(README, encoding="utf-8")
    print("3. Wrote Datasets/README.md")

    left = []
    bare = re.compile(Q + r'(' + "|".join(FOLDERS) + r')\1')
    for p in code_files():
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if bare.search(line) and "Datasets" not in line:
                left.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:120]}")
    print("4. Lines still naming an old folder without Datasets:", len(left))
    for x in left:
        print("     ", x)


README = """# Datasets

All data used by Flood Guard AI is kept in this one folder. Large files are not uploaded to GitHub;
`python -m pipeline_v2.s01_download_data` downloads the long-term data again.

| Folder | What it is | Source | Used by |
|---|---|---|---|
| `data/v2/imd/` | IMD gridded daily rainfall 0.25 deg, 2000-2023 | India Meteorological Department (imdlib) | v2 flood model |
| `data/v2/nasa_power/` | Daily humidity, temperature, wind per district | NASA POWER | v2 flood model |
| `data/v2/India_Flood_Inventory_v3.csv` | Reported flood events 1967-2023 (labels) | India Flood Inventory v3 (Zenodo) | v2 flood model |
| `data/boundaries/` | District boundaries (GADM 4.1) | GADM | district matching, maps |
| `data/` (other folders) | Original project data: rainfall, water level, weather, population | see the project README | earlier models, dashboard |
| `DEM/` | Elevation model | SRTM / Copernicus DEM | GIS risk map |
| `LULC/` | Land use / land cover | ESA WorldCover | GIS risk map |
| `processed_data/` | Cleaned tables and the district-day dataset `district_daily_v2.csv.gz` | built by the pipelines | models, API |
"""

if __name__ == "__main__":
    main()
