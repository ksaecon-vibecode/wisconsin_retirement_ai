"""
============================================================
Workstream 1C: FCC Form 477 Broadband Data — FINAL v4
============================================================
Project: Wisconsin Retirement AI
Script:  code/01_data_acquisition/01c_fcc_broadband.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

DATA FORMAT (confirmed from actual files):
    Both files are WIDE format with identical column structure:
    Year, Month, FIPS, State, County, State_Name, County_Name,
    Housing_Units, Tier_1, Tier_2, Tier_3, Tier_4

    Values = number of residential fixed broadband ISPs
             offering service at or above each speed threshold.

    SPEED THRESHOLDS BY FILE:
    ┌──────────┬──────────────────────┬──────────────────────┐
    │  Column  │  Primary 2014-2025   │  Historical 2008-2013│
    ├──────────┼──────────────────────┼──────────────────────┤
    │  Tier_1  │  > 200 kbps          │  > 200 kbps          │
    │  Tier_2  │  >= 10/1 Mbps        │  >= 3/0.768 Mbps     │
    │  Tier_3  │  >= 25/3 Mbps ★      │  >= 10/1 Mbps        │
    │  Tier_4  │  >= 100/10 Mbps      │  >= 25/3 Mbps ★      │
    └──────────┴──────────────────────┴──────────────────────┘
    ★ = 25/3 Mbps threshold (FCC broadband standard = our IV)

    KEY FINDING: 25/3 Mbps is measured in BOTH files, just under
    different column names. This allows a consistent broadband IV
    across ALL NFCS waves including 2009 and 2012.

INSTRUMENT CONSTRUCTION:
    bb_providers_25_3 = number of ISPs offering >= 25/3 Mbps
    Source: Tier_3 from primary file (2014+)
            Tier_4 from historical file (2008-2013)
    Higher values = more ISP competition = stronger broadband
    infrastructure = better AI financial tool access environment.

OUTPUT:
    data/processed/fcc_broadband_county_year.parquet
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 0. Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FCC      = PROJECT_ROOT / "data" / "raw" / "fcc"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DOCS_DIR     = PROJECT_ROOT / "docs"

PROC_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

WISCONSIN_FIPS_PREFIX = "55"

# ── 1. Speed tier column mapping ─────────────────────────────
# Maps the 25/3 Mbps column name for each file era
TIER_MAP = {
    'primary':    {'bb_25_3': 'tier_3', 'bb_100_10': 'tier_4'},
    'historical': {'bb_25_3': 'tier_4', 'bb_100_10': None},
}

print("=" * 60)
print("Workstream 1C: FCC Broadband Tier Data — Final v4")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print()
print("Schema confirmed:")
print("  Primary (2014-2025):    Tier_3 = 25/3 Mbps ISP count")
print("  Historical (2008-2013): Tier_4 = 25/3 Mbps ISP count")
print("  Consistent IV series possible across all NFCS waves.")


# ── 2. Helper functions ───────────────────────────────────────
def find_file(stem):
    """Find file by stem, trying .csv and .xlsx extensions."""
    for ext in ['.csv', '.xlsx', '.xls', '']:
        p = RAW_FCC / f"{stem}{ext}"
        if p.exists():
            return p
    matches = list(RAW_FCC.glob(f"{stem}*"))
    return matches[0] if matches else None


def read_file_safe(path):
    """Read CSV or Excel, trying multiple encodings for CSV."""
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(path, dtype=str)
        print(f"  Format: Excel")
        return df
    # Try CSV with multiple encodings
    for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False,
                             encoding=enc)
            print(f"  Format: CSV  |  Encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path.name} with any standard encoding.")


def process_fcc_file(stem, era):
    """
    Load, clean, filter to Wisconsin, and extract broadband tiers
    for one FCC file.

    Parameters:
        stem : str  — file name without extension
        era  : str  — 'primary' or 'historical'

    Returns:
        pd.DataFrame with columns:
            county_fips, year, bb_providers_25_3,
            bb_providers_100_10 (primary only), era
    """
    print(f"\n{'='*50}")
    print(f"Processing: {stem}  [{era}]")
    print(f"{'='*50}")

    path = find_file(stem)
    if path is None:
        print(f"  NOT FOUND: {stem}")
        return None

    print(f"  File: {path.name}  ({path.stat().st_size/1024:.0f} KB)")
    df = read_file_safe(path)

    # Standardize column names to lowercase
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ', '_').str.replace('-', '_'))

    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # ── Standardize FIPS to 5-digit string ───────────────────
    # The FIPS column in these files is a raw integer like 55001
    # (state FIPS 55 + county FIPS 001 concatenated)
    # We need it as a zero-padded 5-character string
    fips_col = next(
        (c for c in df.columns if c in ['fips', 'county_fips', 'geoid']),
        None
    )
    if fips_col is None:
        print(f"  ERROR: No FIPS column found. Columns: {list(df.columns)}")
        return None

    df['county_fips'] = df[fips_col].astype(str).str.strip().str.zfill(5)

    # ── Filter to Wisconsin ───────────────────────────────────
    df_wi = df[df['county_fips'].str.startswith(WISCONSIN_FIPS_PREFIX)].copy()
    print(f"  Wisconsin rows: {len(df_wi):,}")

    if len(df_wi) == 0:
        print(f"  ERROR: No Wisconsin rows found.")
        print(f"  Sample FIPS values: {df['county_fips'].head(5).tolist()}")
        return None

    # ── Parse year ────────────────────────────────────────────
    if 'year' not in df_wi.columns:
        print("  ERROR: No year column.")
        return None
    df_wi['year'] = pd.to_numeric(df_wi['year'], errors='coerce')

    # ── Parse period and prefer December ─────────────────────
    # December = end-of-year snapshot, most stable for annual merge
    # June data used as fallback for years where December unavailable
    if 'month' in df_wi.columns:
        df_wi['month_num'] = pd.to_numeric(df_wi['month'], errors='coerce')
        periods = sorted(df_wi['month_num'].dropna().unique().tolist())
        print(f"  Periods (months) available: {periods}")

        df_dec = df_wi[df_wi['month_num'] == 12]
        df_jun = df_wi[df_wi['month_num'] == 6]

        dec_years = set(df_dec['year'].dropna().unique())
        jun_fill  = df_jun[~df_jun['year'].isin(dec_years)]

        df_wi = pd.concat([df_dec, jun_fill], ignore_index=True)
        print(f"  Rows after preferring December: {len(df_wi):,}")
        print(f"  Years: {sorted(df_wi['year'].dropna().unique().tolist())}")

    # ── Extract broadband tier values ─────────────────────────
    tier_cols = TIER_MAP[era]

    # 25/3 Mbps provider count — our main IV variable
    col_25_3 = tier_cols['bb_25_3']      # 'tier_3' or 'tier_4'
    col_100_10 = tier_cols['bb_100_10']  # 'tier_4' or None

    if col_25_3 not in df_wi.columns:
        print(f"  ERROR: Column '{col_25_3}' not found.")
        print(f"  Available columns: {list(df_wi.columns)}")
        return None

    df_wi['bb_providers_25_3'] = pd.to_numeric(
        df_wi[col_25_3], errors='coerce'
    )

    if col_100_10 and col_100_10 in df_wi.columns:
        df_wi['bb_providers_100_10'] = pd.to_numeric(
            df_wi[col_100_10], errors='coerce'
        )
    else:
        df_wi['bb_providers_100_10'] = np.nan

    # ── Aggregate to county-year ──────────────────────────────
    # Each county should already have one row per year after
    # the December filter, but we aggregate to be safe
    df_out = (
        df_wi.groupby(['county_fips', 'year'])
        .agg(
            bb_providers_25_3   = ('bb_providers_25_3',   'max'),
            bb_providers_100_10 = ('bb_providers_100_10', 'max'),
            housing_units       = ('housing_units',        'first')
            if 'housing_units' in df_wi.columns else
            ('county_fips', 'count')
        )
        .reset_index()
    )

    df_out['era'] = era
    print(f"\n  Output: {len(df_out):,} county-year rows")
    print(f"  Counties: {df_out['county_fips'].nunique()}")
    print(f"  bb_providers_25_3 sample:")
    print(f"    Mean={df_out['bb_providers_25_3'].mean():.2f}, "
          f"Max={df_out['bb_providers_25_3'].max():.0f}, "
          f"Min={df_out['bb_providers_25_3'].min():.0f}")

    return df_out


# ── 3. Process both files ─────────────────────────────────────
df_primary    = process_fcc_file("fcc_tier_data_2014_2025",           "primary")
df_historical = process_fcc_file("fcc_tier_data_historical_2008_2013", "historical")

# Combine
frames = [df for df in [df_primary, df_historical] if df is not None]
if not frames:
    print("\nERROR: No FCC data processed. Check files and re-run.")
    exit()

df_fcc = pd.concat(frames, ignore_index=True)
df_fcc = df_fcc.sort_values(['county_fips', 'year']).reset_index(drop=True)


# ── 4. Map to NFCS survey waves ───────────────────────────────
# Each NFCS respondent is matched to the FCC year closest to
# their survey wave year, using December data where available.
#
# NFCS wave -> FCC year used:
#   2009 -> FCC Dec 2009 (historical, Tier_4 = 25/3 Mbps)
#   2012 -> FCC Dec 2012 (historical, Tier_4 = 25/3 Mbps)
#   2015 -> FCC Dec 2015 (primary,    Tier_3 = 25/3 Mbps)
#   2018 -> FCC Dec 2018 (primary,    Tier_3 = 25/3 Mbps)
#   2021 -> FCC Dec 2021 (primary,    Tier_3 = 25/3 Mbps)
#   2024 -> FCC Jun 2024 (primary,    Tier_3 = 25/3 Mbps)

nfcs_to_fcc = {
    2009: 2009,
    2012: 2012,
    2015: 2015,
    2018: 2018,
    2021: 2021,
    2024: 2024,
}

wave_map_df = pd.DataFrame([
    {'fcc_year': v, 'nfcs_wave': k}
    for k, v in nfcs_to_fcc.items()
])

df_fcc = df_fcc.merge(
    wave_map_df,
    left_on  = 'year',
    right_on = 'fcc_year',
    how      = 'left'
).drop(columns=['fcc_year'], errors='ignore')


# ── 5. Final summary ──────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL FCC BROADBAND DATASET")
print("=" * 60)
print(f"Total county-year rows: {len(df_fcc):,}")
print(f"Unique counties:        {df_fcc['county_fips'].nunique()}")
print(f"Years covered:          "
      f"{sorted(df_fcc['year'].dropna().unique().tolist())}")
print(f"\nData by era:")
print(df_fcc['era'].value_counts().to_string())

print(f"\nBroadband providers at 25/3 Mbps — Wisconsin counties by year:")
summary = (
    df_fcc[df_fcc['bb_providers_25_3'].notna()]
    .groupby('year')['bb_providers_25_3']
    .agg(['mean', 'min', 'max', 'count'])
    .round(2)
)
print(summary.to_string())

print(f"\nNFCS wave coverage:")
for wave in [2009, 2012, 2015, 2018, 2021, 2024]:
    wave_data = df_fcc[df_fcc['nfcs_wave'] == wave]
    n_counties = wave_data['county_fips'].nunique()
    has_iv = wave_data['bb_providers_25_3'].notna().sum()
    print(f"  NFCS {wave}: {n_counties} counties, "
          f"{has_iv} with 25/3 Mbps IV data")


# ── 6. Save ───────────────────────────────────────────────────
out_path = PROC_DIR / "fcc_broadband_county_year.parquet"
df_fcc.to_parquet(out_path, index=False, engine='pyarrow')
print(f"\nSaved: {out_path}")
print(f"Size:  {out_path.stat().st_size / 1024:.1f} KB")

# Save log
log_path = DOCS_DIR / "fcc_processing_log.txt"
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("FCC Broadband Processing Log\n")
    f.write(f"Generated: {datetime.now()}\n\n")
    f.write("SCHEMA USED:\n")
    f.write("  Primary (2014-2025):    Tier_3 = 25/3 Mbps ISP count\n")
    f.write("  Historical (2008-2013): Tier_4 = 25/3 Mbps ISP count\n\n")
    f.write(f"Total rows: {len(df_fcc):,}\n")
    f.write(f"Counties:   {df_fcc['county_fips'].nunique()}\n")
    f.write(f"Years:      {sorted(df_fcc['year'].dropna().unique().tolist())}\n")
print(f"Log:   {log_path}")

print("\n" + "=" * 60)
print("Workstream 1C complete.")
print("=" * 60)
