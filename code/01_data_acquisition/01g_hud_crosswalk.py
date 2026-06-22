"""
============================================================
Workstream 1G: HUD ZIP-to-County FIPS Crosswalk
============================================================
Project: Wisconsin Retirement AI
Script:  code/01_data_acquisition/01g_hud_crosswalk.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PURPOSE:
    Download the HUD USPS ZIP-to-County crosswalk files for
    each year 2011-2025. These files map each ZIP code to one
    or more county FIPS codes, with a residential address
    ratio used to assign ZIPs that span multiple counties.

    Two uses in this project:
    1. Map CFPB complaint ZIP codes to Wisconsin county FIPS
    2. Map NFCS respondent ZIP codes (if restricted access
       granted) to Wisconsin county FIPS

MANUAL DOWNLOAD REQUIRED:
    HUD crosswalk files require a free account registration.

    Steps:
    1. Go to: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
    2. Register for a free account if you do not have one
    3. Select "ZIP-County" crosswalk
    4. Download for each quarter: Q4 of each year 2011-2025
       (Q4 is preferred — captures end-of-year ZIP boundaries)
    5. Save each file as:
       data/raw/hud/zip_county_{YEAR}Q4.xlsx
       Example: data/raw/hud/zip_county_2021Q4.xlsx

    Alternatively, download via the HUD API (requires token):
    https://www.huduser.gov/portal/dataset/uspszip-api.html

INPUT:   data/raw/hud/zip_county_{YEAR}Q4.xlsx (one per year)
OUTPUT:  data/processed/hud_zip_county_crosswalk.parquet
         data/processed/cfpb_wisconsin_with_county.parquet

NOTES:
    - ZIP codes that span multiple counties are assigned to the
      county receiving the largest share of residential addresses
      (res_ratio column in HUD file).
    - This is the standard approach in the county-level
      economic geography literature.
    - ZIP code boundaries change over time, so we use the
      year-matched crosswalk file for each CFPB complaint year.
============================================================
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 0. Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_HUD      = PROJECT_ROOT / "data" / "raw" / "hud"
RAW_CFPB     = PROJECT_ROOT / "data" / "processed"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DOCS_DIR     = PROJECT_ROOT / "docs"

PROC_DIR.mkdir(parents=True, exist_ok=True)
RAW_HUD.mkdir(parents=True, exist_ok=True)

# ── 1. Wisconsin county FIPS codes (for validation) ──────────
WI_COUNTY_FIPS = {
    "55001", "55003", "55005", "55007", "55009", "55011",
    "55013", "55015", "55017", "55019", "55021", "55023",
    "55025", "55027", "55029", "55031", "55033", "55035",
    "55037", "55039", "55041", "55043", "55045", "55047",
    "55049", "55051", "55053", "55055", "55057", "55059",
    "55061", "55063", "55065", "55067", "55069", "55071",
    "55073", "55075", "55077", "55078", "55079", "55081",
    "55083", "55085", "55087", "55089", "55091", "55093",
    "55095", "55097", "55099", "55101", "55103", "55105",
    "55107", "55109", "55111", "55113", "55115", "55117",
    "55119", "55121", "55123", "55125", "55127", "55129",
    "55131", "55133", "55135", "55137", "55139", "55141"
}

YEARS = list(range(2011, 2026))

print("=" * 60)
print("Workstream 1G: HUD ZIP-County Crosswalk")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── 2. Load crosswalk files ───────────────────────────────────
def load_crosswalk_year(year: int) -> pd.DataFrame | None:
    """Load HUD crosswalk for a given year, return None if missing."""
    # Try Q4 first, then Q3, Q2, Q1
    for quarter in ["Q4", "Q3", "Q2", "Q1"]:
        path = RAW_HUD / f"zip_county_{year}{quarter}.xlsx"
        if path.exists():
            try:
                df = pd.read_excel(path, dtype=str)
                df.columns = df.columns.str.strip().str.lower()

                # Standardize column names across HUD schema versions
                # HUD has changed column names over time
                col_map = {}
                for col in df.columns:
                    if 'zip' in col and 'code' in col or col == 'zip':
                        col_map[col] = 'zip'
                    elif col in ('county', 'cnty', 'fips'):
                        col_map[col] = 'county_fips'
                    elif 'res_ratio' in col or col == 'res_ratio':
                        col_map[col] = 'res_ratio'
                    elif col == 'tot_ratio':
                        col_map[col] = 'tot_ratio'

                df = df.rename(columns=col_map)

                # Keep only columns we need
                keep = [c for c in ['zip', 'county_fips', 'res_ratio', 'tot_ratio']
                        if c in df.columns]
                df = df[keep].copy()
                df['crosswalk_year'] = year
                df['crosswalk_quarter'] = quarter

                print(f"  {year}{quarter}: {len(df):,} ZIP-county pairs loaded")
                return df

            except Exception as e:
                print(f"  ERROR loading {path.name}: {e}")

    return None


# Check for files
found_years = []
missing_years = []

for yr in YEARS:
    found = any(
        (RAW_HUD / f"zip_county_{yr}{q}.xlsx").exists()
        for q in ["Q4", "Q3", "Q2", "Q1"]
    )
    if found:
        found_years.append(yr)
    else:
        missing_years.append(yr)

if missing_years:
    print(f"\n⚠️  Missing crosswalk files for years: {missing_years}")
    print("Download from: https://www.huduser.gov/portal/datasets/usps_crosswalk.html")
    print("Save as: data/raw/hud/zip_county_{YEAR}Q4.xlsx\n")

if not found_years:
    print("\nNo HUD crosswalk files found.")
    print("Creating placeholder output with instructions.")
    placeholder = pd.DataFrame({
        'zip': pd.Series(dtype=str),
        'county_fips': pd.Series(dtype=str),
        'res_ratio': pd.Series(dtype=float),
        'crosswalk_year': pd.Series(dtype=int)
    })
    out_path = PROC_DIR / "hud_zip_county_crosswalk.parquet"
    placeholder.to_parquet(out_path, index=False)
    print(f"Placeholder saved: {out_path}")
    print("Re-run after downloading HUD files.")
    exit()

# ── 3. Load all available years ───────────────────────────────
print(f"\nLoading {len(found_years)} crosswalk years...")
all_cw = []
for yr in found_years:
    df_yr = load_crosswalk_year(yr)
    if df_yr is not None:
        all_cw.append(df_yr)

df_crosswalk = pd.concat(all_cw, ignore_index=True)
print(f"\nTotal ZIP-county pairs loaded: {len(df_crosswalk):,}")

# ── 4. Build primary assignment crosswalk ─────────────────────
# For each ZIP-year, assign to the county with the highest residential ratio.
# This handles ZIPs that span county boundaries.

print("\nBuilding primary ZIP-to-county assignment (max res_ratio)...")

# Convert res_ratio to numeric
df_crosswalk['res_ratio'] = pd.to_numeric(df_crosswalk['res_ratio'], errors='coerce')

# For each ZIP-year, keep the county with the highest res_ratio
df_primary = (
    df_crosswalk
    .sort_values('res_ratio', ascending=False)
    .groupby(['zip', 'crosswalk_year'])
    .first()
    .reset_index()
    [['zip', 'crosswalk_year', 'county_fips', 'res_ratio']]
    .rename(columns={'county_fips': 'primary_county_fips',
                     'res_ratio': 'primary_res_ratio'})
)

print(f"Unique ZIP-year pairs: {len(df_primary):,}")
print(f"Unique ZIPs: {df_primary['zip'].nunique():,}")

# ── 5. Save full crosswalk and primary assignment ─────────────
cw_path = PROC_DIR / "hud_zip_county_crosswalk.parquet"
df_crosswalk.to_parquet(cw_path, index=False)
print(f"\nFull crosswalk saved: {cw_path}")

primary_path = PROC_DIR / "hud_zip_county_primary.parquet"
df_primary.to_parquet(primary_path, index=False)
print(f"Primary assignment saved: {primary_path}")

# ── 6. Apply crosswalk to CFPB Wisconsin data ─────────────────
cfpb_path = PROC_DIR / "cfpb_wisconsin_raw.parquet"
if cfpb_path.exists():
    print("\nApplying crosswalk to CFPB Wisconsin data...")
    df_cfpb = pd.read_parquet(cfpb_path)

    if 'zip_clean' in df_cfpb.columns and 'year' in df_cfpb.columns:
        # Merge: match each complaint to crosswalk for its complaint year
        df_cfpb = df_cfpb.merge(
            df_primary.rename(columns={'crosswalk_year': 'year'}),
            left_on=['zip_clean', 'year'],
            right_on=['zip', 'year'],
            how='left'
        ).drop(columns=['zip'], errors='ignore')

        matched     = df_cfpb['primary_county_fips'].notna().sum()
        unmatched   = df_cfpb['primary_county_fips'].isna().sum()
        wi_matched  = df_cfpb['primary_county_fips'].isin(WI_COUNTY_FIPS).sum()

        print(f"  Complaints matched to county: {matched:,} ({matched/len(df_cfpb)*100:.1f}%)")
        print(f"  Unmatched (no ZIP or no crosswalk): {unmatched:,}")
        print(f"  Matched to WI county: {wi_matched:,}")

        # Save CFPB with county
        out_cfpb = PROC_DIR / "cfpb_wisconsin_with_county.parquet"
        df_cfpb.to_parquet(out_cfpb, index=False)
        print(f"  Saved: {out_cfpb}")
    else:
        print("  CFPB file missing 'zip_clean' or 'year' column.")
        print("  Run 01b_cfpb_filter.py first.")
else:
    print(f"\n  CFPB processed file not found: {cfpb_path}")
    print("  Run 01b_cfpb_filter.py first, then re-run this script.")

print("\n" + "=" * 60)
print("Workstream 1G complete.")
print("=" * 60)
