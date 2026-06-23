"""
Diagnostic: Check year column in CFPB files
Run: python code/02_nlp_pipeline/diagnose_year.py
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

print("=" * 60)
print("DIAGNOSTIC: Year Column in CFPB Files")
print("=" * 60)

for fname in [
    "cfpb_wisconsin_raw.parquet",
    "cfpb_wisconsin_with_county.parquet",
    "cfpb_wisconsin_preprocessed.parquet",
]:
    path = PROC_DIR / fname
    if not path.exists():
        print(f"\n{fname}: NOT FOUND")
        continue

    df = pd.read_parquet(path)
    print(f"\n--- {fname} ---")
    print(f"  Rows: {len(df):,}")

    # Year column
    year_cols = [c for c in df.columns if 'year' in c.lower()]
    print(f"  Year-related columns: {year_cols}")

    for yc in year_cols:
        vals = pd.to_numeric(df[yc], errors='coerce').dropna()
        if len(vals) > 0:
            print(f"  '{yc}': {vals.nunique()} unique values, "
                  f"range {int(vals.min())}-{int(vals.max())}, "
                  f"nulls={df[yc].isna().sum():,}")
            print(f"    Distribution: {dict(vals.value_counts().sort_index().head(15))}")

    # County column
    county_cols = [c for c in df.columns if 'county' in c.lower() or 'fips' in c.lower()]
    for cc in county_cols:
        non_null = df[cc].notna().sum()
        print(f"  '{cc}': {non_null:,} non-null")

    # Complaint ID
    id_cols = [c for c in df.columns if 'complaint_id' in c.lower() or c == 'id']
    print(f"  ID columns: {id_cols}")
    if id_cols:
        print(f"  Sample IDs: {df[id_cols[0]].head(3).tolist()}")

print("\n" + "=" * 60)
print("Diagnostic complete.")
