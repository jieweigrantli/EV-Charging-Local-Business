#!/usr/bin/env python
"""
run_all.py -- reproduce every table, figure, Source Data file and
Supplementary Data file of the paper from data/filtered_dataset.csv (full
dataset), or a reduced version from data/filtered_dataset_public.csv (Dryad).

    python run_all.py            # full pipeline (about 3-4 minutes)
    python run_all.py --quick    # skip the Monte Carlo convergence simulation
    python run_all.py --no-map   # skip the map (needs geopandas + one download)

Set EVCLB_DATA=/path/to/dataset.csv to run on a different dataset.  Runs on
the public file write to outputs_public/.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
STEPS = [
    ("01_descriptives.py", "Descriptive statistics, Table 1, data for Figures 1, 2, 4 and S6"),
    ("02_two_part_model.py", "Two-part hurdle model, Tables 2 / S2 / S4, robustness (Supplementary Data 1), Figure 3 data"),
    ("03_heterogeneity.py", "Stratified logits by reason for charging, Tables 3 / S3, Wald tests"),
    ("04_counterfactual.py", "Charge-duration counterfactuals (Supplementary Figure 5)"),
    ("05_convergence.py", "Monte Carlo convergence analysis (Supplementary Figures 1-2)"),
    ("06_figures.py", "All figures"),
    ("07_map.py", "Supplementary Figure 3 (map)"),
    ("08_source_data.py", "Source Data and Supplementary Data workbooks"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="skip 05_convergence.py")
ap.add_argument("--no-map", action="store_true", help="skip 07_map.py")
args = ap.parse_args()

t0 = time.time()
for script, desc in STEPS:
    if args.quick and script.startswith("05"):
        print(f"--- skipping {script}")
        continue
    if args.no_map and script.startswith("07"):
        print(f"--- skipping {script}")
        continue
    print(f"\n=== {script}: {desc}")
    r = subprocess.run([sys.executable, script], cwd=CODE)
    if r.returncode != 0:
        sys.exit(f"{script} failed with exit code {r.returncode}")
sys.path.insert(0, str(CODE))
from common import OUT_DIR  # noqa: E402
print(f"\nAll steps finished in {time.time() - t0:.0f} s. Outputs are in {OUT_DIR}.")
