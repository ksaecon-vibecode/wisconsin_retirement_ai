"""
============================================================
Phase 2 NLP Pipeline — Master Runner
============================================================
Project: Wisconsin Retirement AI
Script:  code/02_nlp_pipeline/run_pipeline.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PURPOSE:
    Run the complete NLP pipeline in the correct sequence.
    Calls all four scripts in order and reports final status.

USAGE:
    conda activate wi_retirement_ai
    cd wisconsin_retirement_ai
    python code/02_nlp_pipeline/run_pipeline.py

    OR run individual tracks:
    python code/02_nlp_pipeline/02a_preprocessing.py
    python code/02_nlp_pipeline/02b_track_a_ai_density.py
    python code/02_nlp_pipeline/02c_track_b_topic_model.py
    python code/02_nlp_pipeline/02d_track_c_distress.py

SEQUENCE:
    02a → 02b → 02c → 02d → NLP merge to master dataset

RUNTIME ESTIMATE:
    02a preprocessing:     5-15 min (spaCy lemmatization)
    02b Track A:           2-5 min (keyword) + 20-60 min (semantic, optional)
    02c Track B:           5-20 min (LDA topic modeling)
    02d Track C:           2-5 min (VADER)
    Total (keyword only):  ~15-45 min
    Total (with semantic): ~45-90 min
============================================================
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

print("=" * 60)
print("Phase 2 NLP Pipeline — Master Runner")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

SCRIPTS = [
    ("02a_preprocessing.py",      "Text Preprocessing"),
    ("02b_track_a_ai_density.py", "Track A: AI Complaint Density"),
    ("02c_track_b_topic_model.py","Track B: LDA Topic Model"),
    ("02d_track_c_distress.py",   "Track C: Financial Distress Index"),
]

results = {}

for script_name, label in SCRIPTS:
    script_path = PIPELINE_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Script:  {script_name}")
    print(f"Time:    {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        results[script_name] = "MISSING"
        continue

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,   # Show output in real time
        text=True,
        cwd=str(PROJECT_ROOT)
    )

    if result.returncode == 0:
        results[script_name] = "SUCCESS"
        print(f"\n  {label}: COMPLETE")
    else:
        results[script_name] = "FAILED"
        print(f"\n  {label}: FAILED (return code {result.returncode})")
        print("  Check the error output above.")
        print("  You can fix and re-run individual scripts without")
        print("  restarting the whole pipeline.")

# ── Final status report ───────────────────────────────────────
print(f"\n{'='*60}")
print("PIPELINE STATUS REPORT")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

all_ok = True
for script, status in results.items():
    icon = "OK" if status == "SUCCESS" else "FAIL" if status == "FAILED" else "MISSING"
    print(f"  [{icon}] {script}")
    if status != "SUCCESS":
        all_ok = False

# Check output files
print("\nOutput files:")
expected_outputs = [
    "cfpb_wisconsin_preprocessed.parquet",
    "cfpb_ai_density_county_year.parquet",
    "cfpb_topic_prevalence_county_year.parquet",
    "cfpb_distress_index_county_year.parquet",
]

all_outputs_present = True
for fname in expected_outputs:
    fpath = PROC_DIR / fname
    if fpath.exists():
        size = fpath.stat().st_size / 1024
        print(f"  [OK]   {fname} ({size:.1f} KB)")
    else:
        print(f"  [MISS] {fname} — not yet produced")
        all_outputs_present = False

if all_ok and all_outputs_present:
    print("\n" + "=" * 60)
    print("ALL TRACKS COMPLETE.")
    print("Next step: Run the NLP merge to add outputs to master dataset.")
    print("  source('code/01_data_acquisition/01h_master_merge_nlp.R')")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("Some tracks incomplete. Fix errors above and re-run.")
    print("You can run individual tracks without restarting all.")
    print("=" * 60)
