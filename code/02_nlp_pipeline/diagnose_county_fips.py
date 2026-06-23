"""
============================================================
DIAGNOSTIC: Inspect CFPB county FIPS column
============================================================
Run this in PowerShell from project root:
    python code/02_nlp_pipeline/diagnose_county_fips.py

This tells us exactly what is in primary_county_fips
so we can fix the aggregation correctly.
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

print("=" * 60)
print("DIAGNOSTIC: CFPB County FIPS Inspection")
print("=" * 60)

# ── 1. Load CFPB with county file ────────────────────────────
cfpb_path = PROC_DIR / "cfpb_wisconsin_with_county.parquet"
print(f"\nLoading: {cfpb_path.name}")
df = pd.read_parquet(cfpb_path)
print(f"Total rows: {len(df):,}")
print(f"All columns: {list(df.columns)}")

# ── 2. Find county FIPS column ────────────────────────────────
county_candidates = [c for c in df.columns if 'county' in c.lower() or 'fips' in c.lower()]
print(f"\nCounty/FIPS related columns: {county_candidates}")

for col in county_candidates:
    print(f"\n--- Column: '{col}' ---")
    print(f"  dtype: {df[col].dtype}")
    print(f"  null count: {df[col].isna().sum():,} of {len(df):,} ({df[col].isna().mean()*100:.1f}%)")
    non_null = df[col].dropna()
    if len(non_null) > 0:
        print(f"  non-null count: {len(non_null):,}")
        print(f"  sample values (first 10): {non_null.head(10).tolist()}")
        print(f"  unique values count: {non_null.nunique():,}")
        # Show value length distribution
        if df[col].dtype == object:
            lengths = non_null.astype(str).str.len().value_counts()
            print(f"  value length distribution: {lengths.to_dict()}")
            # Show values starting with 55
            wi_vals = non_null[non_null.astype(str).str.startswith('55')]
            print(f"  values starting with '55': {len(wi_vals):,}")
            print(f"  sample '55' values: {wi_vals.head(5).tolist()}")
        else:
            # Numeric — check range
            print(f"  min: {non_null.min()}, max: {non_null.max()}")
            # Check if they look like FIPS codes
            if non_null.dtype in ['float64', 'int64']:
                wi_vals = non_null[(non_null >= 55000) & (non_null <= 55999)]
                print(f"  values in WI FIPS range (55000-55999): {len(wi_vals):,}")
                print(f"  sample WI values: {wi_vals.head(5).tolist()}")

# ── 3. Also check ZIP column ──────────────────────────────────
zip_col = next((c for c in df.columns if 'zip' in c.lower()), None)
if zip_col:
    print(f"\n--- ZIP column: '{zip_col}' ---")
    print(f"  dtype: {df[zip_col].dtype}")
    print(f"  null count: {df[zip_col].isna().sum():,}")
    non_null = df[zip_col].dropna()
    print(f"  sample values: {non_null.head(10).tolist()}")

# ── 4. Check HUD primary crosswalk ───────────────────────────
hud_path = PROC_DIR / "hud_zip_county_primary.parquet"
if hud_path.exists():
    print(f"\n--- HUD primary crosswalk ---")
    df_hud = pd.read_parquet(hud_path)
    print(f"  Rows: {len(df_hud):,}")
    print(f"  Columns: {list(df_hud.columns)}")
    print(f"  Sample: {df_hud.head(3).to_dict('records')}")

    # Check overlap between CFPB ZIPs and HUD ZIPs
    if zip_col and 'zip' in df_hud.columns:
        cfpb_zips = set(df[zip_col].dropna().astype(str).str.strip())
        hud_zips  = set(df_hud['zip'].astype(str).str.strip())
        overlap   = cfpb_zips & hud_zips
        print(f"\n  CFPB unique ZIPs: {len(cfpb_zips):,}")
        print(f"  HUD unique ZIPs:  {len(hud_zips):,}")
        print(f"  Overlap:          {len(overlap):,}")
        print(f"  CFPB ZIPs NOT in HUD: {len(cfpb_zips - hud_zips):,}")
        print(f"  Sample CFPB ZIPs not in HUD: {list(cfpb_zips - hud_zips)[:5]}")

print("\n" + "=" * 60)
print("Diagnostic complete. Paste this output back.")
print("=" * 60)
