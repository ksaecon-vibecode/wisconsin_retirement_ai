"""
============================================================
FIX: Rebuild year column in CFPB Wisconsin files
============================================================
Project: Wisconsin Retirement AI
Script:  code/02_nlp_pipeline/fix_cfpb_year.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PROBLEM:
    The 'year' column in cfpb_wisconsin_raw.parquet has only
    109 non-null values (2025-2026). The other 27,239 rows
    have null years because date parsing failed in 01b_cfpb_filter.py.

    Root cause: 'date_received' column contains dates in a format
    that pandas could not parse with the default settings, OR
    the column name differs slightly in this CFPB download.

FIX:
    Re-parse the date from whatever date column exists in the
    raw file, trying multiple formats until one works.
    Rebuild 'year', 'month', and 'post_2020' columns.
    Overwrite all three CFPB processed files with correct years.

RUN ORDER:
    python code/02_nlp_pipeline/fix_cfpb_year.py
    python code/02_nlp_pipeline/02b_track_a_ai_density.py
    python code/02_nlp_pipeline/02d_track_c_distress.py
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 60)
log("FIX: Rebuild CFPB Year Column")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 1. Load the raw file ──────────────────────────────────────
raw_path = PROC_DIR / "cfpb_wisconsin_raw.parquet"
log(f"\nLoading: {raw_path.name}")
df = pd.read_parquet(raw_path)
log(f"  Rows: {len(df):,}")
log(f"  Columns: {list(df.columns)}")

# ── 2. Find the date column ───────────────────────────────────
log("\nSearching for date columns...")
date_candidates = [c for c in df.columns
                   if 'date' in c.lower() or 'received' in c.lower()]
log(f"  Date-related columns: {date_candidates}")

for dc in date_candidates:
    sample = df[dc].dropna().head(5).tolist()
    log(f"  '{dc}' sample values: {sample}")

# ── 3. Re-parse dates ─────────────────────────────────────────
log("\nRe-parsing dates...")

# Identify the primary date column
# CFPB standard column is 'date_received' or 'date_received_parsed'
date_col = next(
    (c for c in df.columns
     if c.lower() in ['date_received', 'date received',
                       'date_received_parsed', 'datereceived']),
    None
)
if date_col is None:
    # Try any column with 'date' in the name
    date_col = next(
        (c for c in df.columns if 'date' in c.lower()), None
    )

if date_col is None:
    log("ERROR: No date column found.")
    log(f"Available columns: {list(df.columns)}")
    exit()

log(f"  Using date column: '{date_col}'")
log(f"  Sample raw values: {df[date_col].dropna().head(10).tolist()}")
log(f"  Null count before: {df[date_col].isna().sum():,}")

# Try multiple date format parsers
def parse_dates_robust(series):
    """Try multiple date formats, return parsed datetime series."""
    # Try 1: pandas auto-inference (handles most formats)
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True,
                                errors='coerce')
        n_parsed = parsed.notna().sum()
        if n_parsed > len(series) * 0.5:
            print(f"    Auto-inference: {n_parsed:,} parsed")
            return parsed
    except Exception:
        pass

    # Try 2: MM/DD/YYYY (common US format)
    try:
        parsed = pd.to_datetime(series, format='%m/%d/%Y', errors='coerce')
        n_parsed = parsed.notna().sum()
        if n_parsed > len(series) * 0.5:
            print(f"    MM/DD/YYYY: {n_parsed:,} parsed")
            return parsed
    except Exception:
        pass

    # Try 3: YYYY-MM-DD (ISO format)
    try:
        parsed = pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')
        n_parsed = parsed.notna().sum()
        if n_parsed > len(series) * 0.5:
            print(f"    YYYY-MM-DD: {n_parsed:,} parsed")
            return parsed
    except Exception:
        pass

    # Try 4: mixed format (slowest but most flexible)
    try:
        parsed = pd.to_datetime(series, format='mixed', errors='coerce')
        n_parsed = parsed.notna().sum()
        print(f"    Mixed format: {n_parsed:,} parsed")
        return parsed
    except Exception:
        pass

    # Try 5: dayfirst=True
    try:
        parsed = pd.to_datetime(series, dayfirst=False, errors='coerce')
        n_parsed = parsed.notna().sum()
        print(f"    dayfirst=False: {n_parsed:,} parsed")
        return parsed
    except Exception:
        pass

    return pd.to_datetime(series, errors='coerce')

log(f"\n  Trying date parsers:")
df['date_parsed'] = parse_dates_robust(df[date_col])

n_parsed = df['date_parsed'].notna().sum()
log(f"\n  Dates successfully parsed: {n_parsed:,} of {len(df):,}")

if n_parsed < 100:
    log("\n  WARNING: Very few dates parsed. Checking raw values...")
    log(f"  Raw value examples: {df[date_col].head(20).tolist()}")
    log("  Trying string extraction...")

    # Last resort: extract 4-digit year from string
    year_extracted = df[date_col].astype(str).str.extract(r'(20\d{2})', expand=False)
    log(f"  Years extractable by regex: {year_extracted.notna().sum():,}")
    log(f"  Unique years: {sorted(year_extracted.dropna().unique())}")

    if year_extracted.notna().sum() > 1000:
        df['year']  = pd.to_numeric(year_extracted, errors='coerce')
        df['month'] = np.nan
        log("  Using regex year extraction as fallback")
    else:
        log("  CRITICAL: Cannot parse dates. Check raw CFPB download.")
        log("  The file may be corrupted or use an unexpected format.")
        exit()
else:
    # Extract year and month from parsed dates
    df['year']  = df['date_parsed'].dt.year.astype('Int64')
    df['month'] = df['date_parsed'].dt.month.astype('Int64')
    log(f"  Year range: {df['year'].min()} - {df['year'].max()}")

# ── 4. Build post_2020 flag ───────────────────────────────────
df['post_2020'] = (df['year'] >= 2020).astype('Int64')
df.loc[df['year'].isna(), 'post_2020'] = pd.NA

# ── 5. Show year distribution ─────────────────────────────────
log("\nYear distribution after fix:")
year_counts = df['year'].value_counts().sort_index()
for yr, cnt in year_counts.items():
    bar = '#' * (cnt // 200)
    log(f"  {int(yr)}: {cnt:>6,}  {bar}")

pre_2020  = (df['year'] < 2020).sum()
post_2020 = (df['year'] >= 2020).sum()
null_year = df['year'].isna().sum()
log(f"\n  Pre-2020:  {pre_2020:,}")
log(f"  Post-2020: {post_2020:,}")
log(f"  Null year: {null_year:,}")

# ── 6. Drop the temporary parse column ───────────────────────
df = df.drop(columns=['date_parsed'], errors='ignore')

# ── 7. Save corrected raw file ────────────────────────────────
log("\nSaving corrected files...")
df.to_parquet(raw_path, index=False, engine='pyarrow')
log(f"  Saved: {raw_path.name}")

# ── 8. Also update cfpb_wisconsin_with_county.parquet ─────────
county_path = PROC_DIR / "cfpb_wisconsin_with_county.parquet"
if county_path.exists():
    df_county = pd.read_parquet(county_path)
    # Drop old year/month/post_2020 and merge in corrected versions
    df_county = df_county.drop(
        columns=['year', 'month', 'post_2020', 'date_parsed'],
        errors='ignore'
    )
    df_county = df_county.merge(
        df[['complaint_id', 'year', 'month', 'post_2020']],
        on='complaint_id', how='left'
    )
    df_county.to_parquet(county_path, index=False, engine='pyarrow')
    log(f"  Saved: {county_path.name}")

# ── 9. Also update preprocessed file ─────────────────────────
prep_path = PROC_DIR / "cfpb_wisconsin_preprocessed.parquet"
if prep_path.exists():
    df_prep = pd.read_parquet(prep_path)
    df_prep = df_prep.drop(
        columns=['year', 'month', 'post_2020', 'date_parsed'],
        errors='ignore'
    )
    df_prep = df_prep.merge(
        df[['complaint_id', 'year', 'month', 'post_2020']],
        on='complaint_id', how='left'
    )
    df_prep.to_parquet(prep_path, index=False, engine='pyarrow')
    log(f"  Saved: {prep_path.name}")

log("\n" + "=" * 60)
log("Year fix complete.")
log(f"Year distribution now covers {df['year'].nunique()} unique years.")
log("Next: Re-run Track A and Track C scripts.")
log("=" * 60)
