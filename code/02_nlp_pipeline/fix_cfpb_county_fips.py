"""
============================================================
FIX: Rebuild CFPB County FIPS using 3-digit ZIP prefix
============================================================
Project: Wisconsin Retirement AI
Script:  code/02_nlp_pipeline/fix_cfpb_county_fips.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PROBLEM:
    CFPB redacts the last 2 digits of ZIP codes for privacy:
    '530XX', '542XX', '548XX' etc.
    Only 103 of 27,348 Wisconsin complaints had unredacted ZIPs
    that matched the HUD crosswalk, leaving 27,245 without county.

SOLUTION:
    Build a 3-digit ZIP prefix crosswalk from the HUD data.
    The first 3 digits of any ZIP code reliably identify a small
    geographic area. For each 3-digit prefix, we find the county
    FIPS that receives the most residential addresses across all
    ZIP codes sharing that prefix. This is the standard approach
    for CFPB data with redacted ZIPs.

    For full unredacted ZIPs (like '53149'), we use the exact
    5-digit match first and fall back to prefix only if needed.

INPUT:
    data/processed/hud_zip_county_primary.parquet
    data/processed/cfpb_wisconsin_raw.parquet

OUTPUT:
    data/processed/cfpb_wisconsin_with_county.parquet  (rebuilt)
    data/processed/hud_zip3_county_crosswalk.parquet   (new)
    docs/cfpb_county_fix_log.txt
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DOCS_DIR     = PROJECT_ROOT / "docs"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 60)
log("FIX: Rebuild CFPB County FIPS (3-digit ZIP prefix)")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 1. Load HUD crosswalk ─────────────────────────────────────
log("\n[1/5] Loading HUD ZIP-to-county crosswalk...")
hud_path = PROC_DIR / "hud_zip_county_primary.parquet"
if not hud_path.exists():
    raise FileNotFoundError(f"HUD crosswalk not found: {hud_path}")

df_hud = pd.read_parquet(hud_path)
df_hud.columns = df_hud.columns.str.lower()
log(f"  HUD rows: {len(df_hud):,}")
log(f"  HUD columns: {list(df_hud.columns)}")
log(f"  Sample: {df_hud.head(2).to_dict('records')}")

# Standardize ZIP to string, ensure 5-digit zero-padded
df_hud['zip_str'] = df_hud['zip'].astype(str).str.strip().str.zfill(5)
df_hud['zip3']    = df_hud['zip_str'].str[:3]
df_hud['county_fips_str'] = df_hud['primary_county_fips'].astype(str).str.strip().str.zfill(5)

log(f"  ZIP3 prefixes: {df_hud['zip3'].nunique():,} unique")
log(f"  County FIPS: {df_hud['county_fips_str'].nunique():,} unique")


# ── 2. Build 3-digit prefix crosswalk ────────────────────────
log("\n[2/5] Building 3-digit ZIP prefix crosswalk...")

# For Wisconsin ZIPs (530-549), map prefix to dominant county
# Strategy: for each prefix, use the county that appears most
# frequently across all 5-digit ZIPs sharing that prefix
# weighted by residential ratio

# Filter to Wisconsin prefixes
wi_prefixes = [str(p) for p in range(530, 550)]
df_wi_hud = df_hud[df_hud['zip3'].isin(wi_prefixes)].copy()
log(f"  Wisconsin-prefix rows in HUD: {len(df_wi_hud):,}")
log(f"  Wisconsin ZIP3 prefixes found: {sorted(df_wi_hud['zip3'].unique())}")

# For each ZIP3, find the modal (most common) county FIPS
# Use res_ratio as weight if available
res_col = next((c for c in df_hud.columns
                if 'res' in c.lower() and 'ratio' in c.lower()), None)

if res_col:
    log(f"  Using '{res_col}' as weight")
    df_wi_hud['res_ratio_num'] = pd.to_numeric(df_wi_hud[res_col], errors='coerce').fillna(0)
    # Sum residential ratio by prefix-county pair across all years
    zip3_county = (
        df_wi_hud.groupby(['zip3', 'county_fips_str'])['res_ratio_num']
        .sum()
        .reset_index()
    )
    # For each prefix, pick the county with highest total res_ratio
    zip3_primary = (
        zip3_county
        .sort_values('res_ratio_num', ascending=False)
        .groupby('zip3')
        .first()
        .reset_index()
        .rename(columns={'county_fips_str': 'county_fips_from_zip3'})
        [['zip3', 'county_fips_from_zip3']]
    )
else:
    log("  No res_ratio column found — using modal county")
    zip3_primary = (
        df_wi_hud.groupby(['zip3', 'county_fips_str'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .groupby('zip3')
        .first()
        .reset_index()
        .rename(columns={'county_fips_str': 'county_fips_from_zip3'})
        [['zip3', 'county_fips_from_zip3']]
    )

log(f"\n  3-digit prefix crosswalk built:")
log(f"  Entries: {len(zip3_primary)}")
log(f"\n  Wisconsin ZIP3 -> Primary County:")
for _, row in zip3_primary.sort_values('zip3').iterrows():
    log(f"    {row['zip3']}XX -> {row['county_fips_from_zip3']}")

# Save 3-digit crosswalk
zip3_path = PROC_DIR / "hud_zip3_county_crosswalk.parquet"
zip3_primary.to_parquet(zip3_path, index=False, engine='pyarrow')
log(f"\n  Saved: {zip3_path}")


# ── 3. Load raw CFPB Wisconsin file ──────────────────────────
log("\n[3/5] Loading raw CFPB Wisconsin file...")

# Use the raw file (before county assignment) for a clean rebuild
cfpb_raw_path = PROC_DIR / "cfpb_wisconsin_raw.parquet"
if not cfpb_raw_path.exists():
    # Fall back to the with_county file and drop the bad county columns
    log("  Raw file not found — using cfpb_wisconsin_with_county.parquet")
    cfpb_path = PROC_DIR / "cfpb_wisconsin_with_county.parquet"
    df_cfpb = pd.read_parquet(cfpb_path)
    # Drop old county assignment columns
    drop_cols = [c for c in df_cfpb.columns
                 if 'county' in c.lower() or 'res_ratio' in c.lower()]
    df_cfpb = df_cfpb.drop(columns=drop_cols, errors='ignore')
    log(f"  Dropped old county columns: {drop_cols}")
else:
    df_cfpb = pd.read_parquet(cfpb_raw_path)

log(f"  CFPB rows: {len(df_cfpb):,}")
log(f"  CFPB columns: {list(df_cfpb.columns)}")

# Identify ZIP column
zip_col = next(
    (c for c in df_cfpb.columns if c == 'zip_clean'),
    next((c for c in df_cfpb.columns if 'zip' in c.lower()), None)
)
log(f"  ZIP column: '{zip_col}'")


# ── 4. Extract 3-digit prefix from CFPB ZIPs ─────────────────
log("\n[4/5] Extracting ZIP prefixes from CFPB data...")

def extract_zip3(zip_val):
    """
    Extract 3-digit ZIP prefix, handling redacted ZIPs.
    '53149'  -> '531'   (full ZIP)
    '530XX'  -> '530'   (redacted, extract first 3 digits)
    '530xx'  -> '530'   (lowercase redaction)
    None/NaN -> None
    """
    if pd.isna(zip_val):
        return None
    s = str(zip_val).strip().upper()
    # Extract first 3 digits
    digits = ''
    for ch in s:
        if ch.isdigit():
            digits += ch
            if len(digits) == 3:
                return digits
    return None if len(digits) < 3 else digits

df_cfpb['zip3'] = df_cfpb[zip_col].apply(extract_zip3)
df_cfpb['zip5_clean'] = (
    df_cfpb[zip_col].astype(str)
    .str.replace(r'[^0-9]', '', regex=True)
    .str[:5]
    .where(df_cfpb[zip_col].astype(str).str.replace(r'[^0-9]', '', regex=True).str.len() == 5)
)

zip3_found = df_cfpb['zip3'].notna().sum()
zip5_found = df_cfpb['zip5_clean'].notna().sum()
log(f"  ZIP3 prefix extracted: {zip3_found:,} of {len(df_cfpb):,}")
log(f"  ZIP5 full (unredacted): {zip5_found:,} of {len(df_cfpb):,}")
log(f"  ZIP3 prefix distribution:")
z3_counts = df_cfpb['zip3'].value_counts()
for prefix, cnt in z3_counts.items():
    log(f"    {prefix}XX: {cnt:,}")


# ── 5. Assign county FIPS ─────────────────────────────────────
log("\n[5/5] Assigning county FIPS to all complaints...")

# Step A: Try exact 5-digit match first (for unredacted ZIPs)
df_hud_5 = df_hud[['zip_str', 'county_fips_str']].drop_duplicates('zip_str')
df_cfpb = df_cfpb.merge(
    df_hud_5.rename(columns={
        'zip_str': 'zip5_clean',
        'county_fips_str': 'county_fips_from_zip5'
    }),
    on='zip5_clean',
    how='left'
)

zip5_matched = df_cfpb['county_fips_from_zip5'].notna().sum()
log(f"  Matched via exact ZIP5: {zip5_matched:,}")

# Step B: Fill remaining with 3-digit prefix crosswalk
df_cfpb = df_cfpb.merge(zip3_primary, on='zip3', how='left')

zip3_matched = df_cfpb['county_fips_from_zip3'].notna().sum()
log(f"  Matched via ZIP3 prefix: {zip3_matched:,}")

# Step C: Combine — prefer exact ZIP5, fallback to ZIP3 prefix
df_cfpb['primary_county_fips'] = df_cfpb['county_fips_from_zip5'].fillna(
    df_cfpb['county_fips_from_zip3']
)

total_matched = df_cfpb['primary_county_fips'].notna().sum()
log(f"\n  FINAL county FIPS assignment:")
log(f"  Total matched: {total_matched:,} of {len(df_cfpb):,} ({total_matched/len(df_cfpb)*100:.1f}%)")
log(f"  Unique counties: {df_cfpb['primary_county_fips'].nunique()}")

# County distribution
log("\n  Complaints by county (top 15):")
county_counts = df_cfpb['primary_county_fips'].value_counts().head(15)
for fips, cnt in county_counts.items():
    log(f"    {fips}: {cnt:,} complaints")

# Verify all assigned counties are Wisconsin (55xxx)
non_wi = df_cfpb[
    df_cfpb['primary_county_fips'].notna() &
    ~df_cfpb['primary_county_fips'].astype(str).str.startswith('55')
]
log(f"\n  Non-WI counties assigned (should be 0): {len(non_wi):,}")

# ── 6. Clean up and save ──────────────────────────────────────
# Remove intermediate columns
df_cfpb = df_cfpb.drop(
    columns=['county_fips_from_zip5', 'county_fips_from_zip3', 'zip5_clean'],
    errors='ignore'
)

out_path = PROC_DIR / "cfpb_wisconsin_with_county.parquet"
df_cfpb.to_parquet(out_path, index=False, engine='pyarrow')
log(f"\nRebuilt CFPB file saved: {out_path}")
log(f"Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
log(f"Shape: {df_cfpb.shape[0]:,} rows x {df_cfpb.shape[1]} columns")

# ── 7. Write log ─────────────────────────────────────────────
log_path = DOCS_DIR / "cfpb_county_fix_log.txt"
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log: {log_path}")

log("\n" + "=" * 60)
log("County FIPS fix complete.")
log("Next: Re-run Track A and Track C scripts.")
log("=" * 60)
