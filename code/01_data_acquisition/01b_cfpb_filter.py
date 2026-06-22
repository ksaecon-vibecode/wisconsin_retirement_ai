"""
============================================================
Workstream 1B: CFPB Consumer Complaint Database
Wisconsin Filter and Initial Processing
============================================================
Project: Wisconsin Retirement AI
Script:  code/01_data_acquisition/01b_cfpb_filter.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PURPOSE:
    1. Unzip the raw CFPB complaints file
    2. Filter to Wisconsin complaints only
    3. Filter to records with non-empty narratives
    4. Add temporal flag (pre/post 2020)
    5. Save cleaned Wisconsin file to data/processed/

INPUT:
    data/raw/cfpb/complaints.csv.zip
    (Download from: https://www.consumerfinance.gov/data-research/
     consumer-complaints/#download-the-data)

OUTPUT:
    data/processed/cfpb_wisconsin_raw.parquet
    docs/cfpb_filter_log.txt

RUNTIME:
    ~5-10 minutes depending on machine. The national file is large
    (~600MB compressed, ~3-4GB uncompressed). Script streams the
    file in chunks to avoid loading everything into memory.
============================================================
"""

import os
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── 0. Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cfpb"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

ZIP_PATH = RAW_DIR / "complaints.csv.zip"
OUT_PATH = PROC_DIR / "cfpb_wisconsin_raw.parquet"
LOG_PATH = DOCS_DIR / "cfpb_filter_log.txt"

PROC_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Validate input ───────────────────────────────────────
if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"\n\nCFPB zip file not found at:\n  {ZIP_PATH}\n\n"
        "Please download the full complaints database from:\n"
        "  https://www.consumerfinance.gov/data-research/"
        "consumer-complaints/#download-the-data\n"
        "Save as: data/raw/cfpb/complaints.csv.zip\n"
    )

log_lines = []


def log(msg):
    """Print and record log messages."""
    print(msg)
    log_lines.append(msg)


log("=" * 60)
log("CFPB Wisconsin Filter — Starting")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 2. Identify CSV file inside zip ─────────────────────────
log("\n[1/5] Inspecting zip archive...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    names = z.namelist()
    log(f"  Files in archive: {names}")
    csv_name = next((n for n in names if n.endswith(".csv")), None)
    if csv_name is None:
        raise ValueError("No CSV file found inside the zip archive.")
    log(f"  Target CSV: {csv_name}")

# ── 3. Stream and filter to Wisconsin ───────────────────────
log("\n[2/5] Streaming CSV and filtering to Wisconsin...")

# CFPB column names (stable as of 2025 schema)
CFPB_COLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
]

CHUNK_SIZE = 100_000  # rows per chunk
wi_chunks = []
total_rows = 0
wi_rows = 0
wi_narrative_rows = 0

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    with z.open(csv_name) as f:
        reader = pd.read_csv(
            f,
            dtype=str,  # Read everything as string first
            chunksize=CHUNK_SIZE,
            low_memory=False,
            on_bad_lines="skip",
        )
        for i, chunk in enumerate(reader):
            total_rows += len(chunk)

            # Standardize column names
            chunk.columns = chunk.columns.str.strip()

            # Filter Wisconsin
            state_col = "State" if "State" in chunk.columns else chunk.columns[8]
            wi_chunk = chunk[chunk[state_col].str.strip().str.upper() == "WI"].copy()
            wi_rows += len(wi_chunk)

            if len(wi_chunk) > 0:
                wi_chunks.append(wi_chunk)

            if (i + 1) % 10 == 0:
                log(
                    f"  Processed {total_rows:,} rows so far | "
                    f"WI rows found: {wi_rows:,}"
                )

log(f"\n  Total national rows processed: {total_rows:,}")
log(f"  Wisconsin rows found:          {wi_rows:,}")

# ── 4. Concatenate and clean ─────────────────────────────────
log("\n[3/5] Concatenating Wisconsin chunks and cleaning...")

if not wi_chunks:
    raise ValueError("No Wisconsin rows found. Check the State column in the CSV.")

df = pd.concat(wi_chunks, ignore_index=True)

# Standardize column names: lowercase, underscores
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("?", "", regex=False)
    .str.replace("/", "_", regex=False)
)

log(f"  Columns after cleaning: {list(df.columns)}")

# ── 5. Identify narrative column ────────────────────────────
# Column name varies slightly across CFPB schema versions
narrative_col = next(
    (c for c in df.columns if "narrative" in c or "complaint_text" in c), None
)
if narrative_col is None:
    raise ValueError(
        f"Cannot find narrative column. Available columns: {list(df.columns)}"
    )
log(f"  Narrative column identified: '{narrative_col}'")

# ── 6. Filter to records with non-empty narratives ───────────
log("\n[4/5] Filtering to records with non-empty narratives...")

df["has_narrative"] = (
    df[narrative_col].notna()
    & (df[narrative_col].str.strip() != "")
    & (df[narrative_col].str.strip().str.lower() != "nan")
)
wi_no_narrative = (~df["has_narrative"]).sum()
df_narr = df[df["has_narrative"]].copy()
wi_narrative_rows = len(df_narr)

log(f"  WI rows with narrative:    {wi_narrative_rows:,}")
log(f"  WI rows without narrative: {wi_no_narrative:,}")
log(f"  Share with narrative:      {wi_narrative_rows / wi_rows * 100:.1f}%")

# ── 7. Parse and validate dates ──────────────────────────────
log("\n[5/5] Parsing dates and adding temporal flags...")

date_col = next(
    (
        c
        for c in df_narr.columns
        if "date_received" in c or "date received" in c.replace("_", " ")
    ),
    None,
)
if date_col:
    df_narr["date_received_parsed"] = pd.to_datetime(df_narr[date_col], errors="coerce")
    df_narr["year"] = df_narr["date_received_parsed"].dt.year
    df_narr["month"] = df_narr["date_received_parsed"].dt.month
    df_narr["post_2020"] = (df_narr["year"] >= 2020).astype(int)

    # Date range
    min_date = df_narr["date_received_parsed"].min()
    max_date = df_narr["date_received_parsed"].max()
    log(f"  Date range: {min_date.date()} to {max_date.date()}")

    # Year distribution
    year_counts = df_narr["year"].value_counts().sort_index()
    log(f"\n  Complaints by year (Wisconsin, with narrative):")
    for yr, cnt in year_counts.items():
        bar = "#" * (cnt // 100)
        log(f"    {yr}: {cnt:>6,}  {bar}")

    # Post-2020 breakdown
    pre = (df_narr["post_2020"] == 0).sum()
    post = (df_narr["post_2020"] == 1).sum()
    log(f"\n  Pre-2020 complaints:  {pre:,}")
    log(f"  Post-2020 complaints: {post:,}")
else:
    log("  WARNING: Date column not found. Temporal flags not added.")

# ── 8. ZIP to county mapping placeholder ────────────────────
# HUD crosswalk merge happens in script 01g_hud_crosswalk.py
# For now, preserve the ZIP code column for that later step.
zip_col = next((c for c in df_narr.columns if "zip" in c), None)
if zip_col:
    # Standardize ZIP codes to 5 digits
    df_narr["zip_clean"] = (
        df_narr[zip_col]
        .str.strip()
        .str.extract(r"(\d{5})", expand=False)  # Keep only 5-digit ZIPs
    )
    valid_zip = df_narr["zip_clean"].notna().sum()
    log(f"\n  ZIP codes present and valid 5-digit: {valid_zip:,} of {len(df_narr):,}")

# ── 9. Product distribution ──────────────────────────────────
prod_col = next((c for c in df_narr.columns if c == "product"), None)
if prod_col:
    log(f"\n  Top 10 complaint products (Wisconsin):")
    top_products = df_narr[prod_col].value_counts().head(10)
    for prod, cnt in top_products.items():
        log(f"    {cnt:>6,}  {prod}")

# ── 10. Save output ──────────────────────────────────────────
log(f"\n  Saving to: {OUT_PATH}")
df_narr.to_parquet(OUT_PATH, index=False, engine="pyarrow")
log(f"  File saved. Shape: {df_narr.shape[0]:,} rows x {df_narr.shape[1]} columns")

# ── 11. Summary statistics ───────────────────────────────────
log("\n" + "=" * 60)
log("FILTER SUMMARY")
log("=" * 60)
log(f"National complaints (all states):          {total_rows:,}")
log(f"Wisconsin complaints (all):                {wi_rows:,}")
log(f"Wisconsin complaints (with narrative):     {wi_narrative_rows:,}")
log(
    f"Share of national with WI narrative:       {wi_narrative_rows / total_rows * 100:.2f}%"
)
log(f"\nOutput file: {OUT_PATH.name}")
log(f"Output size: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# ── 12. Write log file ───────────────────────────────────────
# Use utf-8 encoding explicitly to handle all Unicode characters on Windows
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"\nLog written to: {LOG_PATH}")
log("=" * 60)
log("Workstream 1B complete.")
log("=" * 60)
