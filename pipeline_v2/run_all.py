"""Run the whole v2 pipeline: download -> dataset -> models -> GIS risk map.

    python -m pipeline_v2.run_all            (from the project folder; downloads are resumable)
"""
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for step in ["pipeline_v2.s01_download_data", "pipeline_v2.s02_build_dataset", "pipeline_v2.s03_train_models"]:
    print("\n" + "=" * 70 + f"\n{step}\n" + "=" * 70)
    r = subprocess.run([sys.executable, "-m", step] + sys.argv[1:], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"Stopped at {step}. Fix the message above and run again (finished downloads are kept).")
print("\n" + "=" * 70 + "\ngis/06_build_risk_map.py\n" + "=" * 70)
runpy.run_path(str(ROOT / "gis" / "06_build_risk_map.py"), run_name="__main__")
print("\nALL DONE. Results: models/v2_reports/ (metrics_v2.json, model_comparison.csv, roc_pr_curves.png)")
