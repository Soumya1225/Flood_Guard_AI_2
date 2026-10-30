"""Connect the v2 flood-warning model to the dashboard (backups are made first; safe to run twice).

  backend/main.py  : includes the router from backend/flood_v2_api.py  (/api/v2/...)
  frontend App.jsx : shows the <FloodWarningV2 /> panel under the existing GIS panel

    python -m pipeline_v2.integrate_v2
"""
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "backend" / "main.py"
APP = ROOT / "frontend" / "src" / "App.jsx"
BACKUP = ROOT / "_backups" / f"before_v2_{datetime.now():%Y%m%d_%H%M}"
MARK = "# --- Flood Guard AI v2 (real flood events) ---"
BACKEND_BLOCK = f'''

{MARK}
try:
    from backend.flood_v2_api import router as flood_v2_router
except ImportError:
    from flood_v2_api import router as flood_v2_router
app.include_router(flood_v2_router)
'''


def main():
    m = MAIN.read_text(encoding="utf-8")
    a = APP.read_text(encoding="utf-8")
    if MARK not in m or "FloodWarningV2" not in a:
        BACKUP.mkdir(parents=True, exist_ok=True)
        for f in (MAIN, APP):
            shutil.copy2(f, BACKUP / f.name)
        print("Backups of the original files:", BACKUP)
    if MARK not in m:
        tail = m.find('\nif __name__ == "__main__"')
        m = (m.rstrip() + BACKEND_BLOCK) if tail < 0 else (m[:tail].rstrip() + BACKEND_BLOCK + m[tail:])
        MAIN.write_text(m, encoding="utf-8")
        print("backend/main.py: v2 router added")
    else:
        print("backend/main.py: already connected")

    if "FloodWarningV2" not in a:
        imp = 'import EmergencyGISIntelligence from "./EmergencyGISIntelligence";'
        if imp not in a:
            raise SystemExit("Could not find the EmergencyGISIntelligence import in App.jsx; nothing changed.")
        a = a.replace(imp, imp + '\nimport FloodWarningV2 from "./FloodWarningV2";', 1)
        start = a.index("<EmergencyGISIntelligence")
        end = a.index("/>", start) + 2
        indent = a[a.rfind("\n", 0, start) + 1:start]
        a = a[:end] + "\n" + indent + "<FloodWarningV2 />" + a[end:]
        APP.write_text(a, encoding="utf-8")
        print("frontend/src/App.jsx: FloodWarningV2 panel added under the GIS panel")
    else:
        print("frontend/src/App.jsx: already connected")


if __name__ == "__main__":
    main()
