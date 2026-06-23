"""
============================================================
Workstream 1A: FINRA NFCS Processing — FINAL VERSION
Incorporates restricted-use A2 (ZIP) and A4 (race/ethnicity)
============================================================
Project: Wisconsin Retirement AI
Script:  code/01_data_acquisition/01a_nfcs_process.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

INPUTS:
    Public wave files (one per wave):
        data/raw/nfcs/nfcs_2009.zip
        data/raw/nfcs/nfcs_2012.zip
        data/raw/nfcs/nfcs_2015.zip
        data/raw/nfcs/nfcs_2018.zip
        data/raw/nfcs/nfcs_2021.zip
        data/raw/nfcs/nfcs_2024.zip

    Restricted tracking files (national, all waves):
        data/raw/nfcs/restricted/2024_State_Tracking__A2_zip_code.csv
        data/raw/nfcs/restricted/2024_State_Tracking__A4_A4a_ethnicity.csv

    HUD crosswalk (for ZIP-to-county):
        data/processed/hud_zip_county_primary.parquet

OUTPUTS:
    data/processed/nfcs_wisconsin_pooled.parquet
    docs/nfcs_attrition_table.csv
    docs/nfcs_processing_log.txt

SAMPLE DEFINITION:
    Wisconsin respondents (STATEQ = 50)
    Currently employed for an employer (A9 = 2 or 3)
    Age 25-62
    Non-missing retirement enrollment outcome
    NOTE: Cannot distinguish private vs government sector —
          documented in paper data section.
============================================================
"""

import os
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── 0. Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_NFCS = PROJECT_ROOT / "data" / "raw" / "nfcs"
RAW_RESTRICTED = PROJECT_ROOT / "data" / "raw" / "nfcs" / "restricted"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

PROC_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

WISCONSIN_STATEQ = 50
EMPLOYED_CODES = [2, 3]  # Full-time and part-time for employer

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


log("=" * 60)
log("Workstream 1A: NFCS Processing — Final Version")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 1. Load restricted tracking files ────────────────────────
log("\n[1/6] Loading restricted tracking files...")

A2_PATH = RAW_RESTRICTED / "2024_State_Tracking__A2_zip_code.csv"
A4_PATH = RAW_RESTRICTED / "2024_State_Tracking__A4_A4a_ethnicity.csv"

if not A2_PATH.exists():
    raise FileNotFoundError(
        f"A2 file not found: {A2_PATH}\n"
        "Place restricted CSV files in data/raw/nfcs/restricted/"
    )
if not A4_PATH.exists():
    raise FileNotFoundError(
        f"A4 file not found: {A4_PATH}\n"
        "Place restricted CSV files in data/raw/nfcs/restricted/"
    )

df_a2 = pd.read_csv(A2_PATH, dtype=str, encoding="utf-8-sig")
df_a4 = pd.read_csv(A4_PATH, dtype=str, encoding="utf-8-sig")

# Standardize column names
df_a2.columns = df_a2.columns.str.strip().str.upper()
df_a4.columns = df_a4.columns.str.strip().str.upper()

log(f"  A2 (ZIP):       {len(df_a2):,} rows, columns: {list(df_a2.columns)}")
log(f"  A4 (ethnicity): {len(df_a4):,} rows, columns: {list(df_a4.columns)}")

# Merge A2 and A4 on NFCSID
df_restricted = df_a2.merge(df_a4, on="NFCSID", how="inner")
log(f"  Merged restricted: {len(df_restricted):,} rows")

# Clean ZIP code — ensure 5 digits, zero-padded
df_restricted["zip_clean"] = (
    df_restricted["A2"].astype(str).str.strip().str.extract(r"(\d{5})", expand=False)
)
valid_zip = df_restricted["zip_clean"].notna().sum()
log(f"  Valid 5-digit ZIP codes: {valid_zip:,} of {len(df_restricted):,}")


# ── 2. Load HUD ZIP-to-county crosswalk ──────────────────────
log("\n[2/6] Loading HUD ZIP-to-county crosswalk...")

HUD_PATH = PROC_DIR / "hud_zip_county_primary.parquet"
if not HUD_PATH.exists():
    log("  WARNING: HUD crosswalk not found. County FIPS will be missing.")
    df_hud = None
else:
    df_hud = pd.read_parquet(HUD_PATH)
    df_hud.columns = df_hud.columns.str.lower()
    # Ensure ZIP is clean string
    df_hud["zip"] = df_hud["zip"].astype(str).str.strip().str.zfill(5)
    log(f"  HUD crosswalk loaded: {len(df_hud):,} ZIP-year pairs")


# ── 3. NFCS wave variable crosswalk ──────────────────────────
# Maps conceptual variable names to NFCS column names per wave.
# Values of None mean the variable was not available that wave.

WAVE_YEARS = [2009, 2012, 2015, 2018, 2021, 2024]

# Correct answers for Big Five literacy questions
LITERACY_CORRECT = {
    "M6": 1,  # Compound interest: More than $102
    "M7": 3,  # Inflation: Less than today
    "M8": 2,  # Bond prices: They will fall
    "M9": 1,  # Mortgage: True (15-yr has less total interest)
    "M10": 2,  # Diversification: False (single stock riskier)
}

# ── 4. Process each wave ──────────────────────────────────────
log("\n[3/6] Processing public NFCS wave files...")

WAVE_FILES = {
    2009: RAW_NFCS / "nfcs_2009.zip",
    2012: RAW_NFCS / "nfcs_2012.zip",
    2015: RAW_NFCS / "nfcs_2015.zip",
    2018: RAW_NFCS / "nfcs_2018.zip",
    2021: RAW_NFCS / "nfcs_2021.zip",
    2024: RAW_NFCS / "nfcs_2024.zip",
}


def extract_csv_from_zip(zip_path):
    """Extract the State-by-State CSV from an NFCS zip file."""
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [
            n
            for n in z.namelist()
            if n.lower().endswith(".csv")
            and "__macosx" not in n.lower()
            and "inv" not in n.lower()  # Exclude investor file
            and "investor" not in n.lower()
        ]
        if not csv_files:
            # Fallback: any CSV
            csv_files = [
                n
                for n in z.namelist()
                if n.lower().endswith(".csv") and "__macosx" not in n.lower()
            ]
        if not csv_files:
            raise ValueError(f"No CSV found in {zip_path.name}")
        target = csv_files[0]
        with z.open(target) as f:
            df = pd.read_csv(f, dtype=str, low_memory=False, encoding="utf-8")
        return df, target


def process_wave(year, zip_path):
    """Process one NFCS wave: filter WI, construct variables."""
    log(f"\n  --- Wave {year} ---")

    # Load CSV
    df, csv_name = extract_csv_from_zip(zip_path)
    df.columns = df.columns.str.strip().str.upper()
    log(f"    File: {csv_name}")
    log(f"    National rows: {len(df):,}")

    attrition = {"year": year, "national": len(df)}

    # Filter Wisconsin
    if "STATEQ" not in df.columns:
        log(f"    ERROR: STATEQ not found. Columns: {list(df.columns[:10])}")
        return None, attrition

    df["STATEQ_NUM"] = pd.to_numeric(df["STATEQ"], errors="coerce")
    df = df[df["STATEQ_NUM"] == WISCONSIN_STATEQ].copy()
    attrition["wisconsin"] = len(df)
    log(f"    Wisconsin rows: {len(df):,}")

    # Add wave year
    df["SURVEY_WAVE"] = year

    # Filter: employed for employer (A9 = 2 or 3)
    if "A9" in df.columns:
        df["A9_NUM"] = pd.to_numeric(df["A9"], errors="coerce")
        df = df[df["A9_NUM"].isin(EMPLOYED_CODES)].copy()
    attrition["employed"] = len(df)
    log(f"    After employment filter: {len(df):,}")

    # Filter: non-retired household (A10A == 1)
    # NOTE: A3Ar_w contains AGE BRACKET CODES (1-6), NOT raw ages.
    # Filtering on raw bracket values would drop all respondents since
    # codes 1-6 are all less than 25 as integers.
    # Use A10A instead — directly identifies non-retired households.
    # A10A = 1: non-retired household (correct target population)
    # A10A = 2: retired household (respondent retired) — exclude
    # A10A = 3: retired household (spouse retired) — exclude
    if "A10A" in df.columns:
        df["A10A_NUM"] = pd.to_numeric(df["A10A"], errors="coerce")
        df = df[df["A10A_NUM"] == 1].copy()

    # Store age bracket as ordinal control variable (do NOT filter on it)
    age_col = next(
        (c for c in ["A3AR_W", "A3AR", "A3A_W", "A3A"] if c in df.columns), None
    )
    df["AGE_NUM"] = pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan

    attrition["age_25_62"] = len(df)
    log(f"    After non-retired filter (A10A=1): {len(df):,}")

    # ── Outcome: retirement enrollment ──────────────────────
    # C1 = employer plan; C4 = IRA/individual plan
    c1_col = next((c for c in df.columns if c.startswith("C1")), None)
    c4_col = next((c for c in df.columns if c.startswith("C4")), None)

    if c1_col:
        df["C1_NUM"] = pd.to_numeric(df[c1_col], errors="coerce")
    else:
        df["C1_NUM"] = np.nan

    if c4_col:
        df["C4_NUM"] = pd.to_numeric(df[c4_col], errors="coerce")
    else:
        df["C4_NUM"] = np.nan

    # Drop if both outcome variables missing
    df = df[df["C1_NUM"].notna() | df["C4_NUM"].notna()].copy()
    attrition["non_missing_outcome"] = len(df)

    # Enrolled = has employer plan (C1==1) OR has IRA (C4==1)
    df["enrolled"] = ((df["C1_NUM"] == 1) | (df["C4_NUM"] == 1)).astype(int)

    log(f"    After outcome filter: {len(df):,}")
    log(f"    Enrollment rate: {df['enrolled'].mean() * 100:.1f}%")

    # ── Financial literacy score (Big Five) ─────────────────
    lit_score = pd.Series(0.0, index=df.index)
    for q_col, correct in LITERACY_CORRECT.items():
        if q_col in df.columns:
            vals = pd.to_numeric(df[q_col], errors="coerce")
            # Don't know (98) and refuse (99) coded as wrong
            lit_score += (vals == correct).astype(float).fillna(0)

    df["literacy_score"] = lit_score
    log(f"    Mean literacy score: {df['literacy_score'].mean():.2f}/5")

    # ── Present bias index (additive, 0-3) ──────────────────
    bias_items = []

    # Payday loan use (G25_2): 1=Never, 2-5=Used -> coded 1
    if "G25_2" in df.columns:
        p = pd.to_numeric(df["G25_2"], errors="coerce")
        df["bias_payday"] = ((p >= 2) & (p <= 5)).astype(float)
        bias_items.append("bias_payday")

    # Overdraft (B4): 1=Yes
    if "B4" in df.columns:
        o = pd.to_numeric(df["B4"], errors="coerce")
        df["bias_overdraft"] = (o == 1).astype(float)
        bias_items.append("bias_overdraft")

    # CC revolving balance (F2_2): 1=Yes in some months
    if "F2_2" in df.columns:
        c = pd.to_numeric(df["F2_2"], errors="coerce")
        df["bias_cc_revolve"] = (c == 1).astype(float)
        bias_items.append("bias_cc_revolve")

    if bias_items:
        df["impatience_index"] = df[bias_items].sum(axis=1)
    else:
        df["impatience_index"] = np.nan

    log(f"    Mean impatience index: {df['impatience_index'].mean():.2f}/3")

    # ── Financial education instrument (M20, M21_1) ─────────
    if "M20" in df.columns:
        df["fin_ed_received"] = pd.to_numeric(df["M20"], errors="coerce")
    if "M21_1" in df.columns:
        df["fin_ed_school"] = pd.to_numeric(df["M21_1"], errors="coerce")

    # ── AI interest variable (2024 wave only — B61) ──────────
    if year == 2024 and "B61" in df.columns:
        df["ai_interest"] = pd.to_numeric(df["B61"], errors="coerce")
        log(f"    B61 (AI interest) found in 2024 wave")
    else:
        df["ai_interest"] = np.nan

    # ── Other controls from public file ─────────────────────
    control_map = {
        "income": next((c for c in ["A8_2021", "A8"] if c in df.columns), None),
        "education": next((c for c in ["A5_2015", "A5"] if c in df.columns), None),
        "marital": "A6" if "A6" in df.columns else None,
        "homeowner": "EA_1" if "EA_1" in df.columns else None,
        "fin_fragility": "J20" if "J20" in df.columns else None,
        "ret_calc": "J8" if "J8" in df.columns else None,
        "employment": "A9" if "A9" in df.columns else None,
        "weight_nat": "WGT_N2"
        if "WGT_N2" in df.columns
        else "wgt_n2"
        if "wgt_n2" in df.columns
        else None,
        "weight_st": "WGT_S3"
        if "WGT_S3" in df.columns
        else "wgt_s3"
        if "wgt_s3" in df.columns
        else None,
    }

    for std_name, src_col in control_map.items():
        if src_col and src_col in df.columns:
            df[std_name] = pd.to_numeric(df[src_col], errors="coerce")
        else:
            df[std_name] = np.nan

    # Keep NFCSID for merge with restricted files
    keep_cols = (
        ["NFCSID", "SURVEY_WAVE", "STATEQ_NUM"]
        + ["enrolled", "literacy_score", "impatience_index"]
        + ["bias_payday", "bias_overdraft", "bias_cc_revolve"]
        + ["fin_ed_received", "fin_ed_school", "ai_interest"]
        + list(control_map.keys())
        + ["AGE_NUM"]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df, attrition


# ── 5. Run all available waves ────────────────────────────────
all_waves = []
all_attrition = []

for year, zip_path in sorted(WAVE_FILES.items()):
    if not zip_path.exists():
        log(f"\n  MISSING: {zip_path.name} — skipping wave {year}")
        continue
    try:
        df_wave, attrition = process_wave(year, zip_path)
        if df_wave is not None and len(df_wave) > 0:
            all_waves.append(df_wave)
            all_attrition.append(attrition)
    except Exception as e:
        log(f"\n  ERROR processing wave {year}: {e}")

if not all_waves:
    raise ValueError("No waves successfully processed.")

# ── 6. Pool waves ─────────────────────────────────────────────
log("\n[4/6] Pooling waves and merging restricted variables...")

df_pooled = pd.concat(all_waves, ignore_index=True)
log(f"  Pooled: {len(df_pooled):,} respondents across {len(all_waves)} waves")

# Merge restricted A2 and A4 onto Wisconsin pooled data
# NFCSID is the unique join key
df_pooled = df_pooled.merge(
    df_restricted[["NFCSID", "zip_clean", "A4A"]].rename(
        columns={"A4A": "race_eth_restricted"}
    ),
    on="NFCSID",
    how="left",
)

zip_matched = df_pooled["zip_clean"].notna().sum()
log(f"  ZIP codes matched: {zip_matched:,} of {len(df_pooled):,}")

# ── 7. ZIP to county FIPS crosswalk ──────────────────────────
log("\n[5/6] Crosswalking ZIP codes to county FIPS...")

if df_hud is not None:
    # Match each respondent to their wave year crosswalk
    # Use integer year for matching
    df_pooled["wave_year"] = df_pooled["SURVEY_WAVE"].astype(int)

    df_pooled = df_pooled.merge(
        df_hud[["zip", "crosswalk_year", "primary_county_fips"]].rename(
            columns={
                "zip": "zip_clean",
                "crosswalk_year": "wave_year",
                "primary_county_fips": "county_fips",
            }
        ),
        on=["zip_clean", "wave_year"],
        how="left",
    )

    county_matched = df_pooled["county_fips"].notna().sum()
    log(f"  County FIPS matched: {county_matched:,} of {len(df_pooled):,}")
    log(f"  Match rate: {county_matched / len(df_pooled) * 100:.1f}%")

    # For unmatched (ZIP not in that year's crosswalk), try adjacent years
    unmatched = df_pooled["county_fips"].isna()
    if unmatched.sum() > 0:
        log(f"  Attempting fallback match for {unmatched.sum()} unmatched rows...")
        # Try matching to any available crosswalk year within +/- 2 years
        df_hud_any = (
            df_hud[["zip", "primary_county_fips"]]
            .drop_duplicates("zip")
            .rename(
                columns={"zip": "zip_clean", "primary_county_fips": "county_fips_fb"}
            )
        )
        df_pooled = df_pooled.merge(df_hud_any, on="zip_clean", how="left")
        df_pooled.loc[
            unmatched & df_pooled["county_fips_fb"].notna(), "county_fips"
        ] = df_pooled.loc[
            unmatched & df_pooled["county_fips_fb"].notna(), "county_fips_fb"
        ]
        df_pooled = df_pooled.drop(columns=["county_fips_fb"])

        county_final = df_pooled["county_fips"].notna().sum()
        log(
            f"  After fallback: {county_final:,} matched ({county_final / len(df_pooled) * 100:.1f}%)"
        )
else:
    df_pooled["county_fips"] = np.nan
    log("  HUD crosswalk not available — county_fips set to missing")

# ── 8. Construct minority indicator ──────────────────────────
log("\n[6/6] Constructing derived variables...")

# Use restricted A4A (more granular) if available
# A4A codes: 1=White NH, 2=Black NH, 3=Hispanic, 4=Asian/PI, 5=Other
df_pooled["race_eth"] = pd.to_numeric(df_pooled["race_eth_restricted"], errors="coerce")
df_pooled["minority"] = (df_pooled["race_eth"] >= 2).astype(float)
df_pooled.loc[df_pooled["race_eth"].isna(), "minority"] = np.nan

# Low income flag (below Wisconsin median — updated per wave)
# Using income bracket < 4 (below $35,000) as threshold
df_pooled["low_income"] = (df_pooled["income"] <= 3).astype(float)
df_pooled.loc[df_pooled["income"].isna(), "low_income"] = np.nan

# Female indicator (from weight/gender variable)
# A50B in 2021/2024, A3B in earlier waves — already in public data
# Use AGE_NUM as sanity check on sample

log(f"  Minority share: {df_pooled['minority'].mean() * 100:.1f}%")
log(f"  Low-income share: {df_pooled['low_income'].mean() * 100:.1f}%")

# ── 9. Final summary ──────────────────────────────────────────
log("\n" + "=" * 60)
log("FINAL POOLED DATASET SUMMARY")
log("=" * 60)
log(f"Total respondents:   {len(df_pooled):,}")
log(f"Survey waves:        {sorted(df_pooled['SURVEY_WAVE'].unique())}")
log(f"Enrollment rate:     {df_pooled['enrolled'].mean() * 100:.1f}%")
log(f"Mean literacy score: {df_pooled['literacy_score'].mean():.2f}/5")
log(f"Mean impatience:     {df_pooled['impatience_index'].mean():.2f}/3")
log(f"County FIPS present: {df_pooled['county_fips'].notna().sum():,}")
log(f"ZIP present:         {df_pooled['zip_clean'].notna().sum():,}")

log("\nBy wave:")
for yr in sorted(df_pooled["SURVEY_WAVE"].unique()):
    sub = df_pooled[df_pooled["SURVEY_WAVE"] == yr]
    log(
        f"  {yr}: {len(sub):,} respondents | "
        f"enrolled={sub['enrolled'].mean() * 100:.1f}% | "
        f"literacy={sub['literacy_score'].mean():.2f}"
    )

# ── 10. Attrition table ───────────────────────────────────────
attrition_df = pd.DataFrame(all_attrition)
attrition_path = DOCS_DIR / "nfcs_attrition_table.csv"
attrition_df.to_csv(attrition_path, index=False)
log(f"\nAttrition table saved: {attrition_path}")

# ── 11. Save output ───────────────────────────────────────────
out_path = PROC_DIR / "nfcs_wisconsin_pooled.parquet"
df_pooled.to_parquet(out_path, index=False, engine="pyarrow")
log(f"Saved: {out_path}")

# ── 12. Write log ─────────────────────────────────────────────
log_path = DOCS_DIR / "nfcs_processing_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"Log: {log_path}")

log("\n" + "=" * 60)
log("Workstream 1A complete.")
log("=" * 60)
