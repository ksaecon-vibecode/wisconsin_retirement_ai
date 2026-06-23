"""
============================================================
Phase 2 NLP Pipeline — Track C: Financial Distress Index
============================================================
Project: Wisconsin Retirement AI
Script:  code/02_nlp_pipeline/02d_track_c_distress.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PURPOSE:
    Apply VADER sentiment analysis to all 27,348 Wisconsin
    CFPB narratives and aggregate to a county-year Financial
    Distress Index. This enters all four Probit models as a
    regression control capturing general financial stress
    levels in each county beyond what unemployment rate and
    poverty rate capture.

    Financial Distress Index = mean(-VADER_compound)
    Higher values = more negative sentiment = more distress.

    VADER is chosen because:
    - Designed for short, informal text (consumer complaints)
    - No training required — works out of the box
    - Fast on CPU — processes 27,348 narratives in minutes
    - Well-validated in prior CFPB sentiment research
      (Osman & Sabit 2022)

INPUT:
    data/processed/cfpb_wisconsin_preprocessed.parquet

OUTPUT:
    data/processed/cfpb_distress_index_county_year.parquet
    docs/track_c_log.txt
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 0. Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DOCS_DIR     = PROJECT_ROOT / "docs"

INPUT_PATH  = PROC_DIR / "cfpb_wisconsin_preprocessed.parquet"
OUTPUT_PATH = PROC_DIR / "cfpb_distress_index_county_year.parquet"
LOG_PATH    = DOCS_DIR / "track_c_log.txt"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 60)
log("Track C: Financial Distress Index (VADER Sentiment)")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 1. Load VADER ─────────────────────────────────────────────
log("\nLoading VADER sentiment analyzer...")
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    log("  VADER loaded successfully")
    USE_VADER = True
except ImportError:
    log("  VADER not installed.")
    log("  Install: pip install vaderSentiment")
    log("  Using fallback: negative word count heuristic")
    USE_VADER = False
    analyzer = None

# Fallback: simple negative word list if VADER unavailable
NEGATIVE_WORDS = {
    'fraud', 'scam', 'unauthorized', 'stolen', 'lost', 'wrong',
    'error', 'mistake', 'problem', 'issue', 'complaint', 'dispute',
    'denied', 'refused', 'failed', 'impossible', 'terrible', 'horrible',
    'awful', 'bad', 'worse', 'worst', 'unfair', 'deceptive', 'misleading',
    'harassing', 'harassment', 'threatening', 'illegal', 'violation',
    'damage', 'damaged', 'harm', 'harmed', 'hurt', 'suffering',
    'frustrated', 'angry', 'upset', 'distressed', 'stress', 'stressful',
}

def get_sentiment(text: str) -> dict:
    """Get sentiment scores for one text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

    if USE_VADER:
        return analyzer.polarity_scores(text)
    else:
        # Fallback: simple negative word proportion
        words = text.lower().split()
        if len(words) == 0:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        neg_prop = neg_count / len(words)
        # Scale to VADER-like compound score (-1 to 0 range for negative text)
        compound = -min(neg_prop * 5, 1.0)
        return {'compound': compound, 'pos': 0.0,
                'neg': neg_prop, 'neu': 1 - neg_prop}


# ── 2. Load corpus ────────────────────────────────────────────
log(f"\nLoading corpus from: {INPUT_PATH.name}")
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed file not found: {INPUT_PATH}\n"
        "Run 02a_preprocessing.py first."
    )

df = pd.read_parquet(INPUT_PATH)
log(f"  Rows: {len(df):,}")

# Identify narrative column — use original text for sentiment
# (preprocessing removes punctuation which VADER uses for intensity)
narrative_col = next(
    (c for c in df.columns
     if 'narrative' in c.lower() and 'consumer' in c.lower()),
    next(
        (c for c in df.columns if 'narrative' in c.lower()),
        None
    )
)
if narrative_col is None:
    raise ValueError(f"Narrative column not found. Columns: {list(df.columns)}")

log(f"  Narrative column: '{narrative_col}'")
log(f"  Using {'VADER' if USE_VADER else 'fallback negative word'} scoring")


# ── 3. Apply sentiment scoring ────────────────────────────────
log(f"\nScoring {len(df):,} narratives...")

try:
    from tqdm import tqdm
    tqdm.pandas(desc="  Scoring")
    scores = df[narrative_col].progress_apply(
        lambda x: get_sentiment(str(x) if pd.notna(x) else '')
    )
except ImportError:
    log("  (Running without progress bar...)")
    scores = df[narrative_col].apply(
        lambda x: get_sentiment(str(x) if pd.notna(x) else '')
    )

# Extract individual score components
df['vader_compound'] = [s['compound'] for s in scores]
df['vader_pos']      = [s['pos']      for s in scores]
df['vader_neg']      = [s['neg']      for s in scores]
df['vader_neu']      = [s['neu']      for s in scores]

# Financial Distress Score = -compound
# (compound ranges -1 to +1; negating makes higher = more distress)
df['distress_score'] = -df['vader_compound']

log(f"\nSentiment summary:")
log(f"  Mean compound score:   {df['vader_compound'].mean():.4f}")
log(f"  Mean distress score:   {df['distress_score'].mean():.4f}")
log(f"  Pct highly negative:   {(df['vader_compound'] < -0.5).mean()*100:.1f}%")
log(f"  Pct highly positive:   {(df['vader_compound'] > 0.5).mean()*100:.1f}%")
log(f"  Pct near neutral:      {(df['vader_compound'].abs() < 0.05).mean()*100:.1f}%")

# Sample: most distressed and least distressed narratives
log("\nSample — top 3 most distressed narratives:")
most_distressed = df.nlargest(3, 'distress_score')
for i, (_, row) in enumerate(most_distressed.iterrows()):
    log(f"  [{i+1}] Score={row['distress_score']:.3f}: "
        f"{str(row[narrative_col])[:100]}...")

log("\nSample — top 3 least distressed (most positive):")
least_distressed = df.nsmallest(3, 'distress_score')
for i, (_, row) in enumerate(least_distressed.iterrows()):
    log(f"  [{i+1}] Score={row['distress_score']:.3f}: "
        f"{str(row[narrative_col])[:100]}...")


# ── 4. Aggregate to county-year ───────────────────────────────
log("\nAggregating to county-year Financial Distress Index...")

# Join updated county FIPS from the fixed cfpb_wisconsin_with_county.parquet
log("  Loading fixed county FIPS from cfpb_wisconsin_with_county.parquet...")
FIXED_CFPB_PATH = PROC_DIR / "cfpb_wisconsin_with_county.parquet"

if FIXED_CFPB_PATH.exists():
    df_fixed = pd.read_parquet(FIXED_CFPB_PATH)
    id_col_fixed = next((c for c in df_fixed.columns if 'complaint_id' in c.lower()), None)
    id_col_pre   = next((c for c in df.columns if 'complaint_id' in c.lower()), None)
    if id_col_fixed and id_col_pre:
        df_fixed[id_col_fixed] = df_fixed[id_col_fixed].astype(str)
        df[id_col_pre]         = df[id_col_pre].astype(str)
        df = df.drop(columns=["primary_county_fips"], errors="ignore")
        df = df.merge(
            df_fixed[[id_col_fixed, "primary_county_fips"]].rename(
                columns={id_col_fixed: id_col_pre}),
            on=id_col_pre, how="left"
        )
        log(f"  County FIPS joined: {df['primary_county_fips'].notna().sum():,} matched")
    else:
        log(f"  complaint_id not found (fixed={id_col_fixed}, pre={id_col_pre})")
else:
    log(f"  Fixed CFPB file not found: {FIXED_CFPB_PATH}")

year_col = "year" if "year" in df.columns else None

df_wi = df[df["primary_county_fips"].notna()].copy()
df_wi["county_fips"] = df_wi["primary_county_fips"].astype(str).str.strip().str.zfill(5)
df_wi = df_wi[df_wi["county_fips"].str.match(r"^\d{5}$")].copy()
log(f"  Rows with valid county FIPS: {len(df_wi):,}")

if year_col and len(df_wi) > 0:
    df_wi["year_num"] = pd.to_numeric(df_wi[year_col], errors="coerce")

    county_year = df_wi.groupby(["county_fips", "year_num"]).agg(
        n_complaints        = (narrative_col, "count"),
        distress_index      = ("distress_score", "mean"),
        mean_vader_compound = ("vader_compound", "mean"),
        mean_vader_neg      = ("vader_neg", "mean"),
        mean_vader_pos      = ("vader_pos", "mean"),
        pct_highly_negative = ("vader_compound",
                               lambda x: (x < -0.5).mean()),
    ).reset_index().rename(columns={"year_num": "year"})

    log(f"\n  County-year cells: {len(county_year):,}")
    log(f"  Counties:          {county_year['county_fips'].nunique()}")
    log(f"  Years:             {sorted(county_year['year'].dropna().unique().tolist())}")

    log("\n  Mean distress index by year:")
    yr_dist = county_year.groupby("year")["distress_index"].agg(
        ["mean", "min", "max"]).round(4)
    for yr, row in yr_dist.iterrows():
        log(f"    {int(yr)}: mean={row['mean']:.4f} "
            f"min={row['min']:.4f} max={row['max']:.4f}")

    county_year.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    log(f"\nSaved: {OUTPUT_PATH}")
    log(f"Size:  {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
else:
    log("  Cannot aggregate — no rows with valid county FIPS and year.")


# ── 5. Write log ─────────────────────────────────────────────
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log: {LOG_PATH}")

log("\n" + "=" * 60)
log("Track C complete.")
log("=" * 60)
